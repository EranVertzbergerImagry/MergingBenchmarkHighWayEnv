"""
Ego-centric observation wrapper.

Rotates all observations into the ego vehicle's heading frame so the policy
becomes heading-invariant.  Also injects the ego vehicle's destination
direction (cos_d, sin_d) into row 0 — other vehicles keep cos_d=sin_d=0
(we use observe_intentions=false, so only the ego knows its own destination).

Expected feature order (9 features):
  [presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]
   0         1  2  3   4   5      6      7      8

After transformation the ego row becomes:
  presence=1, x=0, y=0, vx=forward_speed, vy≈0, cos_h=1, sin_h=0, cos_d=..., sin_d=...
"""
import numpy as np
import gymnasium as gym


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

    def observation(self, obs):
        obs = obs.copy()

        # Step 1: Inject ego's destination direction into row 0.
        # With observe_intentions=false the env zeros cos_d/sin_d for other
        # vehicles, but the ego row (index 0) already has them from to_dict()
        # (called without observe_intentions flag).  We overwrite explicitly
        # for clarity / safety.
        ego_vehicle = self.env.unwrapped.vehicle
        dest_dir = ego_vehicle.destination_direction
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
