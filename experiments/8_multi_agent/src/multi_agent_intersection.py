"""
Multi-agent intersection environment with staggered random spawning.

Extends RandomSpawnIntersectionEnv to support multiple controlled vehicles that
spawn at random times throughout the episode. Bypasses highway-env's
MultiAgentObservation/MultiAgentAction — handles tuple packing internally.

Key features:
  - N controlled vehicles (default 4), each spawning at a random time
  - Collision-safe spawn: defers placement if entry lane is occupied
  - Unborn/done agents get zero observations and contribute no reward
  - Returns tuple observations and accepts tuple actions

Registered as "intersection-multi-agent-v0".
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.observation import KinematicObservation, observation_factory

from src.random_spawn_intersection import RandomSpawnIntersectionEnv


class MultiAgentIntersectionEnv(RandomSpawnIntersectionEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "controlled_vehicles": 4,
            "spawn_clearance": 20,        # min distance to existing vehicles for safe spawn
            "spawn_window_fraction": 0.6, # agents spawn in first 60% of episode duration
        })
        return config

    def define_spaces(self) -> None:
        """Override to create tuple observation/action spaces."""
        super().define_spaces()
        n = self.config["controlled_vehicles"]

        # Single-agent spaces (used internally for each agent)
        self._single_obs_space = self.observation_space
        self._single_act_space = self.action_space

        # Multi-agent tuple spaces
        self.observation_space = spaces.Tuple(tuple(
            self._single_obs_space for _ in range(n)
        ))
        self.action_space = spaces.Tuple(tuple(
            self._single_act_space for _ in range(n)
        ))

    def _reset(self) -> None:
        """Reset: build road, spawn NPC traffic, prepare agent slots."""
        self._make_road()

        # Spawn NPC traffic only (not controlled vehicles)
        self._spawn_npc_traffic(self.config["initial_vehicle_count"])

        n = self.config["controlled_vehicles"]
        duration = self.config["duration"]
        window = duration * self.config["spawn_window_fraction"]

        # Pre-assign random spawn times, entries, destinations for each agent
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

        # Map from agent slot index to index in self.controlled_vehicles
        self._agent_vehicle_idx = [None] * n

    def _spawn_npc_traffic(self, n_vehicles):
        """Spawn NPC vehicles and challenger (same as parent _make_vehicles but without controlled vehicles)."""
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1.0,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0.0,
        )

    def _try_spawn_agent(self, agent_idx):
        """Try to spawn a controlled vehicle for the given agent slot. Returns True if spawned."""
        entry = self._agent_entries[agent_idx]
        destination = self._agent_destinations[agent_idx]
        clearance = self.config["spawn_clearance"]

        ego_lane = self.road.network.get_lane(
            (f"o{entry}", f"ir{entry}", 0)
        )
        spawn_pos = ego_lane.position(60.0 + 5.0 * self.np_random.normal(1.0), 0.0)

        # Check clearance against all existing vehicles
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - spawn_pos) < clearance:
                return False  # Lane not clear, defer

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

        # Restore observer_vehicle to first controlled vehicle (or None)
        if self.controlled_vehicles:
            self.observation_type.observer_vehicle = self.controlled_vehicles[0]

        return tuple(obs_list)

    def reset(self, *, seed=None, options=None):
        """Override to return tuple observations.

        We replicate AbstractEnv.reset() logic but skip its observe() call,
        because at reset time no controlled vehicles exist yet (they spawn
        during the episode). We use our own _observe_all() which returns
        zero-padded observations for unborn agents.
        """
        # Replicate AbstractEnv.reset() without the final observe()
        # (gym.Env.reset for seeding)
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

        # Build tuple observations (all zeros since no agents spawned yet)
        obs = self._observe_all()
        info = self._build_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """
        Step with tuple of actions.

        action: tuple of N ints (one per agent slot)
        """
        n = self.config["controlled_vehicles"]

        # 1. Try to spawn agents whose time has come
        for i in range(n):
            if not self._agent_spawned[i] and self.time >= self._agent_spawn_times[i]:
                self._try_spawn_agent(i)

        # 2. Apply actions to active agents
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                if not vehicle.crashed:
                    vehicle.act(self.action_type.actions[int(action[i])])

        # 3. Simulate physics (with intermediate rendering for video capture)
        self.time += 1 / self.config["policy_frequency"]
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            # Render intermediate frames for smooth video (last frame rendered by caller)
            if frame < frames - 1:
                self._automatic_rendering()

        # 4. Spawn new NPC traffic & clear exiting vehicles (from IntersectionEnv.step)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

        # 5. Update done status for each agent
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                vidx = self._agent_vehicle_idx[i]
                vehicle = self.controlled_vehicles[vidx]
                if vehicle.crashed or self.has_arrived(vehicle):
                    self._agent_done[i] = True

        # 6. Build observations
        obs = self._observe_all()

        # 7. Compute per-agent rewards
        agent_rewards = self._compute_rewards(action)

        # 8. Termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # 9. Info (includes per-agent rewards)
        info = self._build_info()
        info["agent_rewards"] = agent_rewards

        # Scalar reward for gym compatibility (average of spawned agents)
        spawned_rewards = [r for i, r in enumerate(agent_rewards)
                          if self._agent_spawned[i]]
        reward = sum(spawned_rewards) / max(len(spawned_rewards), 1)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_rewards(self, action):
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

    def _is_terminated(self):
        """Terminate when all spawned agents are done."""
        n = self.config["controlled_vehicles"]
        any_spawned = any(self._agent_spawned)
        if not any_spawned:
            return False  # Wait for first spawn
        # All spawned agents must be done
        for i in range(n):
            if self._agent_spawned[i] and not self._agent_done[i]:
                return False
        return True

    def _is_truncated(self):
        return self.time >= self.config["duration"]

    def _build_info(self):
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

        # Aggregate for rl-agents compatibility
        any_crashed = any(a["crashed"] for a in agents_info)
        any_arrived = any(a["arrived"] for a in agents_info)

        return {
            "agents": agents_info,
            "crashed": any_crashed,
            "is_success": all(
                a["arrived"] for a in agents_info if a["spawned"]
            ),
        }

    @property
    def vehicle(self):
        """Return first controlled vehicle for compatibility (rendering, etc)."""
        if self.controlled_vehicles:
            return self.controlled_vehicles[0]
        return None


gym.register(
    id="intersection-multi-agent-v0",
    entry_point="src.multi_agent_intersection:MultiAgentIntersectionEnv",
)
