"""
Standalone ego-centric observation transform.

Replicates the rotation logic from EgoCentricWrapper._transform_single() so the
frozen opponent environment can transform observations for frozen vehicles without
going through the gymnasium wrapper.

Expected feature order (9 features):
  [presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]
   0         1  2  3   4   5      6      7      8
"""
import numpy as np


# Feature column indices
PRESENCE = 0
X, Y = 1, 2
VX, VY = 3, 4
COS_H, SIN_H = 5, 6
COS_D, SIN_D = 7, 8


def ego_centric_transform(obs, vehicle):
    """Transform a (V, 9) observation array into the given vehicle's heading frame.

    Also injects the vehicle's destination direction into row 0.
    Replicates EgoCentricWrapper.observation() logic.

    Args:
        obs: numpy array of shape (V, 9) — raw kinematics observation
        vehicle: the vehicle whose perspective to use (has .destination_direction,
                 heading encoded in obs[0, COS_H/SIN_H])

    Returns:
        Transformed observation array (copy, does not modify input).
    """
    obs = obs.copy()

    # Step 1: Inject vehicle's destination direction into row 0.
    dest_dir = vehicle.destination_direction
    obs[0, COS_D] = dest_dir[0]
    obs[0, SIN_D] = dest_dir[1]

    # Step 2: Ego-centric rotation by -ego_heading.
    ego_cos = obs[0, COS_H]
    ego_sin = obs[0, SIN_H]
    ego_x = obs[0, X]
    ego_y = obs[0, Y]

    for i in range(obs.shape[0]):
        if obs[i, PRESENCE] < 0.5:
            continue

        # Translate position to ego origin, then rotate by -theta
        dx = obs[i, X] - ego_x
        dy = obs[i, Y] - ego_y
        obs[i, X] = ego_cos * dx + ego_sin * dy
        obs[i, Y] = -ego_sin * dx + ego_cos * dy

        # Rotate velocity (no subtraction -- ego needs its own speed)
        vx, vy = obs[i, VX], obs[i, VY]
        obs[i, VX] = ego_cos * vx + ego_sin * vy
        obs[i, VY] = -ego_sin * vx + ego_cos * vy

        # Rotate heading direction
        ch, sh = obs[i, COS_H], obs[i, SIN_H]
        obs[i, COS_H] = ego_cos * ch + ego_sin * sh
        obs[i, SIN_H] = -ego_sin * ch + ego_cos * sh

        # Rotate destination direction
        cd, sd = obs[i, COS_D], obs[i, SIN_D]
        obs[i, COS_D] = ego_cos * cd + ego_sin * sd
        obs[i, SIN_D] = -ego_sin * cd + ego_cos * sd

    return obs
