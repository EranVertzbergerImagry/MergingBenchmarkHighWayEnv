"""
Unified Intersection Environment (Experiment 9).

Supports both IDM and frozen-model traffic, with single or multi-agent control.
Unifies functionality from experiments 7 (IDM + single ego), 8 (IDM + multi-agent),
and 9 (frozen model traffic).

Config-driven behavior:
  - enviromental_driver_model_path: None or path → frozen model traffic
  - enviromental_driver_model_path absent/IDM mode → IDM traffic (parent behavior)
  - controlled_vehicles: 1 → single-agent, >1 → multi-agent with staggered spawning

Registered as "intersection-frozen-opponents-v0" (backward compat)
and "intersection-unified-v0".
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch

from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv

from src.random_spawn_intersection import RandomSpawnIntersectionEnv
from src.ego_centric_transform import ego_centric_transform


class FrozenOpponentIntersectionEnv(RandomSpawnIntersectionEnv):
    """Intersection env with configurable traffic model and multi-agent support."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "enviromental_driver_model_path": None,  # Path to frozen .tar, or None for IDM
            "frozen_agent_config": None,              # Agent config dict (network arch)
            "controlled_vehicles": 1,                 # Number of controlled vehicles
            "spawn_clearance": 20,                    # Min distance for safe agent spawn
            "spawn_window_fraction": 0.6,             # Agents spawn in first N% of episode
        })
        return config

    @property
    def _use_frozen_traffic(self):
        """True if using frozen model traffic, False if using IDM."""
        return (
            self.config.get("enviromental_driver_model_path") is not None
            and self.config.get("frozen_agent_config") is not None
        )

    @property
    def _is_multi_agent(self):
        return self.config["controlled_vehicles"] > 1

    # ------------------------------------------------------------------
    # Spaces (multi-agent override)
    # ------------------------------------------------------------------
    def define_spaces(self) -> None:
        """Override to create tuple observation/action spaces for multi-agent."""
        super().define_spaces()
        if not self._is_multi_agent:
            return

        n = self.config["controlled_vehicles"]
        self._single_obs_space = self.observation_space
        self._single_act_space = self.action_space

        self.observation_space = spaces.Tuple(tuple(
            self._single_obs_space for _ in range(n)
        ))
        self.action_space = spaces.Tuple(tuple(
            self._single_act_space for _ in range(n)
        ))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        """Build road, spawn NPC traffic, spawn or prepare ego vehicle(s)."""
        self._make_road()
        self._frozen_vehicles = []

        # Load frozen network if needed (skip on first gym.make reset)
        use_frozen = self._use_frozen_traffic and self._load_frozen_net_if_needed()

        if use_frozen:
            self._spawn_frozen_npc_traffic()
        else:
            self._spawn_idm_npc_traffic()

        if self._is_multi_agent:
            self._reset_multi_agent()
        else:
            self._reset_single_agent(use_frozen)

    def _spawn_idm_npc_traffic(self):
        """Spawn IDM NPC vehicles (experiment 7 behavior)."""
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        n_vehicles = self.config["initial_vehicle_count"]
        simulation_steps = 3
        for t in range(max(n_vehicles - 1, 0)):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            for _ in range(self.config["simulation_frequency"]):
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1.0,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0.0,
        )

    def _spawn_frozen_npc_traffic(self):
        """Spawn frozen model-driven NPC vehicles (experiment 9 behavior)."""
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        n_vehicles = self.config["initial_vehicle_count"]
        for t in range(max(n_vehicles - 1, 0)):
            self._spawn_frozen_vehicle(
                longitudinal=np.linspace(0, 80, n_vehicles)[t]
            )

        simulation_steps = 3
        for _ in range(simulation_steps):
            for _ in range(self.config["simulation_frequency"]):
                self._act_frozen_vehicles()
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])

        # Challenger vehicle (frozen)
        self._spawn_frozen_vehicle(
            longitudinal=60,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0.0,
        )

    def _reset_single_agent(self, use_frozen):
        """Spawn a single controlled vehicle."""
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

        # Remove any NPC vehicle too close to ego
        for v in list(self.road.vehicles):
            if (
                v is not ego_vehicle
                and np.linalg.norm(v.position - ego_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)
                if v in self._frozen_vehicles:
                    self._frozen_vehicles.remove(v)

    def _reset_multi_agent(self):
        """Prepare staggered spawn slots for N controlled vehicles (exp 8 pattern)."""
        n = self.config["controlled_vehicles"]
        duration = self.config["duration"]
        window = duration * self.config["spawn_window_fraction"]

        self._agent_spawn_times = sorted(
            self.np_random.uniform(0, window, size=n).tolist()
        )
        self._agent_entries = self.np_random.integers(0, 4, size=n).tolist()
        self._agent_destinations = []
        for entry in self._agent_entries:
            other_exits = [i for i in range(4) if i != entry]
            dest_idx = self.np_random.choice(other_exits)
            self._agent_destinations.append(f"o{dest_idx}")

        self._agent_spawned = [False] * n
        self._agent_done = [False] * n
        self.controlled_vehicles = []
        self._agent_vehicle_idx = [None] * n

    # ------------------------------------------------------------------
    # Multi-agent reset/step helpers
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if not self._is_multi_agent:
            return super().reset(seed=seed, options=options)

        import gymnasium
        gymnasium.Env.reset(self, seed=seed, options=options)

        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()

        obs = self._observe_all()
        info = self._build_multi_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def _try_spawn_agent(self, agent_idx):
        """Try to spawn a controlled vehicle for the given agent slot."""
        entry = self._agent_entries[agent_idx]
        destination = self._agent_destinations[agent_idx]
        clearance = self.config["spawn_clearance"]

        ego_lane = self.road.network.get_lane(
            (f"o{entry}", f"ir{entry}", 0)
        )
        spawn_pos = ego_lane.position(60.0 + 5.0 * self.np_random.normal(1.0), 0.0)

        for v in self.road.vehicles:
            if np.linalg.norm(v.position - spawn_pos) < clearance:
                return False

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            spawn_pos,
            speed=ego_lane.speed_limit,
            heading=ego_lane.heading_at(60.0),
        )
        try:
            ego_vehicle.plan_route_to(destination)
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        self._agent_vehicle_idx[agent_idx] = len(self.controlled_vehicles) - 1
        self._agent_spawned[agent_idx] = True
        return True

    def _observe_single(self, vehicle):
        """Build a Kinematics observation from one vehicle's perspective."""
        self.observation_type.observer_vehicle = vehicle
        return self.observation_type.observe()

    def _observe_all(self):
        """Build tuple of observations, one per agent slot."""
        n = self.config["controlled_vehicles"]
        zero_obs = np.zeros(self._single_obs_space.shape)
        obs_list = []

        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                vidx = self._agent_vehicle_idx[i]
                obs_list.append(self._observe_single(self.controlled_vehicles[vidx]))
            else:
                obs_list.append(zero_obs.copy())

        if self.controlled_vehicles:
            self.observation_type.observer_vehicle = self.controlled_vehicles[0]

        return tuple(obs_list)

    def _compute_multi_rewards(self, action):
        """Per-agent reward tuple. Unspawned agents get 0.0."""
        n = self.config["controlled_vehicles"]
        rewards = []
        for i in range(n):
            if self._agent_spawned[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                rewards.append(self._agent_reward(action, vehicle))
            else:
                rewards.append(0.0)
        return tuple(rewards)

    def _build_multi_info(self):
        """Build info dict with per-agent status."""
        n = self.config["controlled_vehicles"]
        agents_info = []
        for i in range(n):
            ai = {
                "agent_id": i,
                "spawned": self._agent_spawned[i],
                "done": self._agent_done[i],
                "entry": f"o{self._agent_entries[i]}",
                "destination": self._agent_destinations[i],
                "crashed": False,
                "arrived": False,
            }
            if self._agent_spawned[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                ai["crashed"] = vehicle.crashed
                ai["arrived"] = self.has_arrived(vehicle)
            agents_info.append(ai)

        any_crashed = any(a["crashed"] for a in agents_info)

        return {
            "agents": agents_info,
            "crashed": any_crashed,
            "is_success": all(
                a["arrived"] for a in agents_info if a["spawned"]
            ),
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        if self._is_multi_agent:
            return self._step_multi(action)
        else:
            return self._step_single(action)

    def _step_single(self, action):
        """Single-agent step (original exp 9 behavior)."""
        use_frozen = self._use_frozen_traffic

        # 1. Drive frozen vehicles (if applicable)
        if use_frozen:
            self._act_frozen_vehicles()

        # 2. Apply ego action
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

        # 5. Spawn new NPC traffic
        if use_frozen:
            self._maybe_spawn_frozen_vehicle()
            self._frozen_vehicles = [
                v for v in self._frozen_vehicles if v in self.road.vehicles
            ]
        else:
            self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

        # 6. Observe, reward, done
        obs = self.observation_type.observe()
        reward = self._agent_reward(action, self.vehicle)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "crashed": self.vehicle.crashed,
            "is_success": self.has_arrived(self.vehicle),
            "rewards": self._agent_rewards_dict(action, self.vehicle),
        }
        if use_frozen:
            info["num_frozen_vehicles"] = len(self._frozen_vehicles)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _step_multi(self, action):
        """Multi-agent step (exp 8 pattern)."""
        n = self.config["controlled_vehicles"]
        use_frozen = self._use_frozen_traffic

        # 1. Try to spawn agents whose time has come
        for i in range(n):
            if not self._agent_spawned[i] and self.time >= self._agent_spawn_times[i]:
                self._try_spawn_agent(i)

        # 2. Drive frozen NPC vehicles (if applicable)
        if use_frozen:
            self._act_frozen_vehicles()

        # 3. Apply actions to active agents
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                if not vehicle.crashed:
                    vehicle.act(self.action_type.actions[int(action[i])])

        # 4. Simulate physics
        self.time += 1 / self.config["policy_frequency"]
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            if frame < frames - 1:
                self._automatic_rendering()

        # 5. Clear vehicles & spawn new NPC traffic
        self._clear_vehicles()
        if use_frozen:
            self._maybe_spawn_frozen_vehicle()
            self._frozen_vehicles = [
                v for v in self._frozen_vehicles if v in self.road.vehicles
            ]
        else:
            self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

        # 6. Update done status for each agent
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                if vehicle.crashed or self.has_arrived(vehicle):
                    self._agent_done[i] = True

        # 7. Build observations and rewards
        obs = self._observe_all()
        agent_rewards = self._compute_multi_rewards(action)

        # 8. Termination
        terminated = self._is_terminated_multi()
        truncated = self._is_truncated()

        # 9. Info
        info = self._build_multi_info()
        info["agent_rewards"] = agent_rewards
        if use_frozen:
            info["num_frozen_vehicles"] = len(self._frozen_vehicles)

        # Scalar reward for gym compatibility (average of spawned agents)
        spawned_rewards = [r for i, r in enumerate(agent_rewards)
                          if self._agent_spawned[i]]
        reward = sum(spawned_rewards) / max(len(spawned_rewards), 1)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _is_terminated_multi(self):
        """Terminate when all spawned agents are done."""
        n = self.config["controlled_vehicles"]
        any_spawned = any(self._agent_spawned)
        if not any_spawned:
            return False
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                return False
        return True

    def _agent_rewards_dict(self, action, vehicle):
        """Compute reward components dict (for info)."""
        try:
            return super()._agent_rewards(action, vehicle)
        except AttributeError:
            return {"arrived_reward": 1.0 if self.has_arrived(vehicle) else 0.0}

    @property
    def vehicle(self):
        """Return first controlled vehicle for compatibility."""
        if self.controlled_vehicles:
            return self.controlled_vehicles[0]
        return None

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

        from rl_agents.agents.deep_q_network.pytorch import DQNAgent

        n_features = len(
            self.config.get("observation", {}).get(
                "features",
                ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "cos_d", "sin_d"],
            )
        )
        vehicles_count = self.config.get("observation", {}).get("vehicles_count", 15)
        n_actions = len(self.config.get("action", {}).get("target_speeds", [0, 4.5, 9]))

        import gymnasium.spaces as gspaces
        dummy_obs_space = gspaces.Box(
            low=-np.inf, high=np.inf,
            shape=(vehicles_count, n_features), dtype=np.float32,
        )
        dummy_act_space = gspaces.Discrete(n_actions)

        class _DummyEnv:
            observation_space = dummy_obs_space
            action_space = dummy_act_space

        dummy_agent = DQNAgent(_DummyEnv(), agent_config)
        dummy_agent.load(model_path)
        self._frozen_net = dummy_agent.value_net
        self._frozen_net.eval()
        self._frozen_device = next(self._frozen_net.parameters()).device
        return True

    # ------------------------------------------------------------------
    # Frozen vehicle actions
    # ------------------------------------------------------------------
    def _act_frozen_vehicles(self):
        """Compute and apply actions for all active frozen vehicles."""
        for vehicle in self._frozen_vehicles:
            if vehicle.crashed:
                vehicle.color = None
                continue
            if not hasattr(vehicle, 'route') or vehicle.route is None:
                continue

            obs = self._observe_from(vehicle)
            obs = ego_centric_transform(obs, vehicle)
            action_idx = self._frozen_action(obs)
            vehicle.act(self.action_type.actions[action_idx])

    def _observe_from(self, vehicle):
        """Build KinematicObservation from a specific vehicle's perspective."""
        original_observer = self.observation_type.observer_vehicle
        self.observation_type.observer_vehicle = vehicle
        obs = self.observation_type.observe()
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
        """Spawn a frozen MDPVehicle with a random route."""
        entry = self.np_random.integers(0, 4)
        lane_id = (f"o{entry}", f"ir{entry}", 0)

        try:
            lane = self.road.network.get_lane(lane_id)
        except KeyError:
            return

        pos_long = longitudinal + 5.0 * position_deviation * self.np_random.normal()
        position = lane.position(pos_long, 0)
        heading = lane.heading_at(pos_long)
        speed = lane.speed_limit + speed_deviation * self.np_random.normal()
        speed = max(speed, 0)

        for v in self.road.vehicles:
            if np.linalg.norm(v.position - position) < 15:
                return

        vehicle = self.action_type.vehicle_class(
            self.road,
            position,
            speed=speed,
            heading=heading,
        )

        if go_straight:
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

gym.register(
    id="intersection-unified-v0",
    entry_point="src.frozen_opponent_env:FrozenOpponentIntersectionEnv",
)
