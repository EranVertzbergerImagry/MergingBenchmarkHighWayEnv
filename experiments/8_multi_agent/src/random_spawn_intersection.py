"""
Custom intersection environment with configurable spawn entry and wrong-destination penalty.

Subclasses IntersectionEnv to add:
  - spawn_entry config: 0-3 for fixed entry, null for random (useful at inference)
  - destination always random from remaining 3 exits
  - wrong_destination_reward: penalty when arriving at wrong exit

Registered as "intersection-random-spawn-v0".
"""
import numpy as np
import gymnasium as gym

from highway_env import utils
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.vehicle.kinematics import Vehicle


class RandomSpawnIntersectionEnv(IntersectionEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "spawn_entry": 0,                # 0-3 for fixed, None for random
            "wrong_destination_reward": None, # None => use collision_reward
        })
        return config

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Same as parent but with configurable spawn entry and random destination
        excluding the ego's own entry.
        """
        # Configure other vehicles (same as parent)
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
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

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            # Determine spawn entry
            spawn_entry = self.config["spawn_entry"]
            if spawn_entry is None:
                spawn_entry = self.np_random.integers(0, 4)

            ego_lane = self.road.network.get_lane(
                (f"o{spawn_entry}", f"ir{spawn_entry}", 0)
            )

            # Pick random destination from exits that aren't the ego's entry
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
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Add wrong-destination penalty on top of parent reward."""
        reward = super()._agent_reward(action, vehicle)

        # Check if vehicle arrived at wrong destination
        if self.has_arrived(vehicle) and vehicle.route:
            # vehicle.lane_index = (from_node, to_node, lane_id)
            # vehicle.route[-1] = (from_node, to_node, lane_id) of planned final segment
            actual_exit = vehicle.lane_index[1]     # e.g. "o1"
            planned_exit = vehicle.route[-1][1]     # e.g. "o2"
            if actual_exit != planned_exit:
                penalty = self.config["wrong_destination_reward"]
                if penalty is None:
                    penalty = self.config["collision_reward"]
                reward = penalty

        return reward


gym.register(
    id="intersection-random-spawn-v0",
    entry_point="src.random_spawn_intersection:RandomSpawnIntersectionEnv",
)
