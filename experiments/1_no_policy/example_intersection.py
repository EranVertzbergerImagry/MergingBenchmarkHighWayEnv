"""
Highway-env Intersection Example Script
Demonstrates the intersection environment where the ego vehicle
must navigate through a busy intersection.

Uses RecordVideo wrapper with set_record_video_wrapper() to capture
all intermediate simulation frames for smooth video playback.
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import os
import json
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    """Load config from default_config.json in script dir."""
    default_path = os.path.join(SCRIPT_DIR, "default_config.json")
    with open(default_path) as f:
        return json.load(f)


def apply_idm_params(idm_config):
    """Apply IDM behavior parameters to the vehicle class before env creation."""
    from highway_env.vehicle.behavior import IDMVehicle
    if "comfort_acc_max" in idm_config:
        IDMVehicle.COMFORT_ACC_MAX = idm_config["comfort_acc_max"]
    if "comfort_acc_min" in idm_config:
        IDMVehicle.COMFORT_ACC_MIN = idm_config["comfort_acc_min"]
    if "distance_wanted" in idm_config:
        IDMVehicle.DISTANCE_WANTED = idm_config["distance_wanted"]
    if "time_wanted" in idm_config:
        IDMVehicle.TIME_WANTED = idm_config["time_wanted"]


def main():
    config = load_config()
    if config.get("idm"):
        apply_idm_params(config["idm"])
    env_config = config["env"]

    # Clean up previous video folder
    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    # Create the intersection environment
    env = gym.make('intersection-v1', render_mode='rgb_array', config=env_config)

    # Wrap with RecordVideo to capture video
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record every episode
        name_prefix="intersection"
    )

    # Key fix: Allow environment to record intermediate simulation frames
    env.unwrapped.set_record_video_wrapper(env)

    print("=== Intersection Environment ===")
    print("")
    print("Goal: Navigate through the intersection to reach the destination.")
    print("Recording simulation with smooth framerate...")
    print("")

    obs, info = env.reset()

    # Reposition ego closer to the intersection (lane is 100m, default start is 60m)
    vehicle = env.unwrapped.controlled_vehicles[0]
    lane = vehicle.lane
    ego_position = 95.0  # meters along the incoming lane (100 = intersection edge)
    vehicle.position = lane.position(ego_position, 0)
    vehicle.heading = lane.heading_at(ego_position)
    vehicle.speed = 0.0
    vehicle.target_speed = 0.0
    
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 1000:
        # action = env.action_space.sample()
        action = 1  # IDLE - maintain speed   
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    crashed = info.get("crashed", False)
    # Check arrived via rewards dict (is_success may not exist)
    arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
    status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")

    print(f"Result: {status} | Steps: {steps} | Reward: {total_reward:.2f}")
    print("")

    env.close()

    # List generated video files
    print("Generated video files:")
    for f in sorted(os.listdir(video_folder)):
        filepath = os.path.join(video_folder, f)
        size = os.path.getsize(filepath) / 1024
        print(f"  {f} ({size:.1f} KB)")

    print("")
    print("Done!")

if __name__ == "__main__":
    main()
