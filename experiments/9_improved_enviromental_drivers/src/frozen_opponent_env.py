"""
Frozen Opponent Intersection Environment.

Replaces IDM traffic with frozen model-driven MDPVehicles. The environment is
single-agent from the outside (one ego vehicle), but internally manages N frozen
opponent vehicles that use a pre-trained model (e.g., from experiment 7) to
select actions each step.

Key design:
  - Extends RandomSpawnIntersectionEnv (single ego, random spawn/destination)
  - No IDM vehicles at all — all traffic is model-driven
  - Frozen vehicles are MDPVehicles with actions computed via the frozen network
  - The training pipeline is identical to experiment 7 (single-agent DQN)

Registered as "intersection-frozen-opponents-v0".
"""
import numpy as np
import gymnasium as gym

import torch

from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv

from src.random_spawn_intersection import RandomSpawnIntersectionEnv
from src.ego_centric_transform import ego_centric_transform


class FrozenOpponentIntersectionEnv(RandomSpawnIntersectionEnv):
    """Single-agent intersection with frozen model-driven traffic instead of IDM."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "enviromental_driver_model_path": None,  # Path to frozen .tar checkpoint
            "frozen_agent_config": None,              # Agent config dict (network arch)
        })
        return config

    def _reset(self) -> None:
        """Build road, spawn ego, then spawn initial frozen vehicles."""
        self._make_road()

        # Load frozen network (once, or on first reset).
        # On the very first reset (triggered by gym.make before config is applied),
        # the frozen model path isn't set yet — fall back to parent IDM behavior.
        if not self._load_frozen_net_if_needed():
            super()._make_vehicles()
            return

        # Spawn initial frozen traffic (replaces IDM _make_vehicles logic)
        self._frozen_vehicles = []

        # Configure vehicle type parameters (same as parent _make_vehicles)
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        n_vehicles = self.config["initial_vehicle_count"]
        for t in range(max(n_vehicles - 1, 0)):
            self._spawn_frozen_vehicle(
                longitudinal=np.linspace(0, 80, n_vehicles)[t]
            )

        # Simulate a few steps so vehicles spread out
        simulation_steps = 3
        for _ in range(simulation_steps):
            for _ in range(self.config["simulation_frequency"]):
                # Act frozen vehicles before each sim step
                self._act_frozen_vehicles()
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])

        # Challenger vehicle (frozen, not IDM)
        self._spawn_frozen_vehicle(
            longitudinal=60,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0.0,
        )

        # Spawn ego (controlled vehicle) — same as parent
        self.controlled_vehicles = []
        spawn_entry = self.config["spawn_entry"]
        if spawn_entry is None:
            spawn_entry = self.np_random.integers(0, 4)

        ego_lane = self.road.network.get_lane(
            (f"o{spawn_entry}", f"ir{spawn_entry}", 0)
        )

        if self.config["destination"] is not None:
            destination = self.config["destination"]
        else:
            other_exits = [i for i in range(4) if i != spawn_entry]
            dest_idx = self.np_random.choice(other_exits)
            destination = f"o{dest_idx}"

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(60.0 + 5.0 * self.np_random.normal(1.0), 0.0),
            speed=ego_lane.speed_limit,
            heading=ego_lane.heading_at(60.0),
        )
        try:
            ego_vehicle.plan_route_to(destination)
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                ego_lane.speed_limit
            )
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                ego_vehicle.speed_index
            )
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        # Remove any vehicle too close to ego
        for v in list(self.road.vehicles):
            if (
                v is not ego_vehicle
                and np.linalg.norm(v.position - ego_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)
                if v in self._frozen_vehicles:
                    self._frozen_vehicles.remove(v)

    def step(self, action):
        """Override IntersectionEnv.step to drive frozen vehicles before simulation."""
        # 1. Compute and apply actions for frozen vehicles
        self._act_frozen_vehicles()

        # 2. Apply ego action (standard single-agent)
        self.controlled_vehicles[0].act(
            self.action_type.actions[action]
        )

        # 3. Simulate physics
        self.time += 1 / self.config["policy_frequency"]
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            if frame < frames - 1:
                self._automatic_rendering()

        # 4. Clear exited/crashed vehicles
        self._clear_vehicles()

        # 5. Maybe spawn new frozen vehicle (replaces IDM spawn)
        self._maybe_spawn_frozen_vehicle()

        # 6. Update frozen list (remove vehicles cleared from road)
        self._frozen_vehicles = [
            v for v in self._frozen_vehicles if v in self.road.vehicles
        ]

        # 7. Observe, reward, done (single-agent)
        obs = self.observation_type.observe()
        reward = self._agent_reward(action, self.vehicle)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "crashed": self.vehicle.crashed,
            "is_success": self.has_arrived(self.vehicle),
            "rewards": self._agent_rewards(action, self.vehicle),
            "num_frozen_vehicles": len(self._frozen_vehicles),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _agent_rewards(self, action, vehicle):
        """Compute reward components dict (for info)."""
        try:
            return super()._agent_rewards(action, vehicle)
        except AttributeError:
            # Fallback if parent doesn't have _agent_rewards
            return {"arrived_reward": 1.0 if self.has_arrived(vehicle) else 0.0}

    # ------------------------------------------------------------------
    # Frozen network
    # ------------------------------------------------------------------
    def _load_frozen_net_if_needed(self):
        """Load the frozen EgoAttentionNetwork from checkpoint (once).

        Returns True if the frozen net is ready, False if config is not yet applied
        (e.g., during the initial gym.make reset before configure() is called).
        """
        if hasattr(self, "_frozen_net") and self._frozen_net is not None:
            return True

        model_path = self.config.get("enviromental_driver_model_path")
        agent_config = self.config.get("frozen_agent_config")

        if model_path is None or agent_config is None:
            return False

        # Build the network from agent config
        from rl_agents.agents.deep_q_network.pytorch import DQNAgent

        # Create a minimal env-like object for DQNAgent to read spaces from
        # We need observation_space and action_space
        n_features = len(
            self.config.get("observation", {}).get(
                "features",
                ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "cos_d", "sin_d"],
            )
        )
        vehicles_count = self.config.get("observation", {}).get("vehicles_count", 15)
        n_actions = len(self.config.get("action", {}).get("target_speeds", [0, 4.5, 9]))

        import gymnasium.spaces as spaces
        dummy_obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(vehicles_count, n_features), dtype=np.float32,
        )
        dummy_act_space = spaces.Discrete(n_actions)

        class _DummyEnv:
            observation_space = dummy_obs_space
            action_space = dummy_act_space

        dummy_agent = DQNAgent(_DummyEnv(), agent_config)
        dummy_agent.load(model_path)
        self._frozen_net = dummy_agent.value_net
        self._frozen_net.eval()

        # Also store the device
        self._frozen_device = next(self._frozen_net.parameters()).device
        return True

    # ------------------------------------------------------------------
    # Frozen vehicle actions
    # ------------------------------------------------------------------
    def _act_frozen_vehicles(self):
        """Compute and apply actions for all active frozen vehicles."""
        for vehicle in self._frozen_vehicles:
            if vehicle.crashed:
                vehicle.color = None  # Let default red show through
                continue
            if not hasattr(vehicle, 'route') or vehicle.route is None:
                continue

            # Get observation from this vehicle's perspective
            obs = self._observe_from(vehicle)

            # Apply ego-centric transform
            obs = ego_centric_transform(obs, vehicle)

            # Get greedy action from frozen network
            action_idx = self._frozen_action(obs)

            # Apply action to the MDPVehicle
            vehicle.act(self.action_type.actions[action_idx])

    def _observe_from(self, vehicle):
        """Build KinematicObservation from a specific vehicle's perspective."""
        # Temporarily swap the observer vehicle
        original_observer = self.observation_type.observer_vehicle
        self.observation_type.observer_vehicle = vehicle
        obs = self.observation_type.observe()
        # Restore original observer
        self.observation_type.observer_vehicle = original_observer
        return obs

    def _frozen_action(self, obs):
        """Get greedy action from frozen network (no exploration)."""
        with torch.no_grad():
            state = torch.tensor(
                obs, dtype=torch.float32, device=self._frozen_device
            ).unsqueeze(0)
            q_values = self._frozen_net(state)
            return q_values.argmax(dim=-1).item()

    # ------------------------------------------------------------------
    # Frozen vehicle spawning
    # ------------------------------------------------------------------
    def _spawn_frozen_vehicle(self, longitudinal=60, go_straight=False,
                              position_deviation=1.0, speed_deviation=0.0):
        """Spawn a frozen MDPVehicle with a random route.

        Mirrors _spawn_vehicle() but creates MDPVehicle instead of IDMVehicle.
        """
        # Pick a random entry
        entry = self.np_random.integers(0, 4)
        lane_id = (f"o{entry}", f"ir{entry}", 0)

        try:
            lane = self.road.network.get_lane(lane_id)
        except KeyError:
            return  # Entry doesn't exist

        # Position along the lane
        pos_long = longitudinal + 5.0 * position_deviation * self.np_random.normal()
        position = lane.position(pos_long, 0)
        heading = lane.heading_at(pos_long)
        speed = lane.speed_limit + speed_deviation * self.np_random.normal()
        speed = max(speed, 0)

        # Clearance check — abort if any vehicle is within 15m (same as _spawn_vehicle)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - position) < 15:
                return

        # Create MDPVehicle (same class as ego)
        vehicle = self.action_type.vehicle_class(
            self.road,
            position,
            speed=speed,
            heading=heading,
        )

        # Pick a random destination (not the entry)
        if go_straight:
            # Go straight through the intersection
            destination = f"o{(entry + 2) % 4}"
        else:
            other_exits = [i for i in range(4) if i != entry]
            dest_idx = self.np_random.choice(other_exits)
            destination = f"o{dest_idx}"

        try:
            vehicle.plan_route_to(destination)
            vehicle.speed_index = vehicle.speed_to_index(speed)
            vehicle.target_speed = vehicle.index_to_speed(vehicle.speed_index)
        except AttributeError:
            pass

        # Light blue like IDM vehicles so they're visually distinct from the ego
        vehicle.color = (100, 200, 255)

        self.road.vehicles.append(vehicle)
        self._frozen_vehicles.append(vehicle)

    def _maybe_spawn_frozen_vehicle(self):
        """Spawn a new frozen vehicle with spawn_probability."""
        if self.np_random.random() < self.config["spawn_probability"]:
            self._spawn_frozen_vehicle()


gym.register(
    id="intersection-frozen-opponents-v0",
    entry_point="src.frozen_opponent_env:FrozenOpponentIntersectionEnv",
)
