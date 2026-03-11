"""
Ego-centric observation wrapper.

Rotates all observations into the ego vehicle's heading frame so the policy
becomes heading-invariant.  Also injects the ego vehicle's destination
direction (cos_d, sin_d) into row 0 — other vehicles keep cos_d=sin_d=0
(we use observe_intentions=false, so only the ego knows its own destination).

Supports both single-agent (numpy array) and multi-agent (tuple of arrays)
observations. In multi-agent mode, each agent's observation is transformed
independently using that agent's own heading.

Expected feature order (9 features):
  [presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]
   0         1  2  3   4   5      6      7      8

After transformation the ego row becomes:
  presence=1, x=0, y=0, vx=forward_speed, vy≈0, cos_h=1, sin_h=0, cos_d=..., sin_d=...
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EgoCentricWrapper(gym.ObservationWrapper):
    """Rotate observations into the ego vehicle's heading frame."""

    # Feature column indices
    PRESENCE = 0
    X, Y = 1, 2
    VX, VY = 3, 4
    COS_H, SIN_H = 5, 6
    COS_D, SIN_D = 7, 8

    def __init__(self, env):
        super().__init__(env)
        # Override observation_space if it's a Tuple (multi-agent)
        if isinstance(self.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple(tuple(
                s for s in self.observation_space.spaces
            ))

    def observation(self, obs):
        if isinstance(obs, tuple):
            return self._observation_multi(obs)
        return self._transform_single(obs, self.env.unwrapped.vehicle)

    def _observation_multi(self, obs_tuple):
        """Transform each agent's observation independently."""
        env = self.env.unwrapped
        result = []
        for i, agent_obs in enumerate(obs_tuple):
            # Check if agent is spawned (has a vehicle)
            if (hasattr(env, '_agent_spawned')
                    and i < len(env._agent_spawned)
                    and env._agent_spawned[i]
                    and env._agent_vehicle_idx[i] is not None):
                vidx = env._agent_vehicle_idx[i]
                vehicle = env.controlled_vehicles[vidx]
                result.append(self._transform_single(agent_obs, vehicle))
            else:
                # Unborn/done: pass through zeros unchanged
                result.append(agent_obs)
        return tuple(result)

    def _transform_single(self, obs, vehicle):
        """Transform a single observation array into ego-centric frame."""
        obs = obs.copy()

        # Step 1: Inject ego's destination direction into row 0.
        dest_dir = vehicle.destination_direction
        obs[0, self.COS_D] = dest_dir[0]
        obs[0, self.SIN_D] = dest_dir[1]

        # Step 2: Ego-centric rotation by -ego_heading.
        ego_cos = obs[0, self.COS_H]
        ego_sin = obs[0, self.SIN_H]
        ego_x = obs[0, self.X]
        ego_y = obs[0, self.Y]

        for i in range(obs.shape[0]):
            if obs[i, self.PRESENCE] < 0.5:
                continue

            # Translate position to ego origin, then rotate by -theta
            dx = obs[i, self.X] - ego_x
            dy = obs[i, self.Y] - ego_y
            obs[i, self.X] = ego_cos * dx + ego_sin * dy
            obs[i, self.Y] = -ego_sin * dx + ego_cos * dy

            # Rotate velocity (no subtraction — ego needs its own speed)
            vx, vy = obs[i, self.VX], obs[i, self.VY]
            obs[i, self.VX] = ego_cos * vx + ego_sin * vy
            obs[i, self.VY] = -ego_sin * vx + ego_cos * vy

            # Rotate heading direction
            ch, sh = obs[i, self.COS_H], obs[i, self.SIN_H]
            obs[i, self.COS_H] = ego_cos * ch + ego_sin * sh
            obs[i, self.SIN_H] = -ego_sin * ch + ego_cos * sh

            # Rotate destination direction
            cd, sd = obs[i, self.COS_D], obs[i, self.SIN_D]
            obs[i, self.COS_D] = ego_cos * cd + ego_sin * sd
            obs[i, self.SIN_D] = -ego_sin * cd + ego_cos * sd

        return obs
