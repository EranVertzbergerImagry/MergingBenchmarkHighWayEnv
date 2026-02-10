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
import shutil

def main():
    # Clean up previous video folder
    video_folder = "videos"
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)

    # Create the intersection environment
    env = gym.make('intersection-v1', render_mode='rgb_array', config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 2, 5, 10, 15, 20],  # Speed levels for FASTER/SLOWER
        },
        "duration": 100,              # Episode lasts 13 seconds (simulation time)                                                                                                                                 
        "destination": "o1",         # Target exit (o0=west, o1=north, o2=east, o3=south)                                                                                                                         
        "initial_vehicle_count": 10, # Start with 10 other vehicles                                                                                                                                               
        "spawn_probability": 0.6,    # 60% chance to spawn new vehicle each step  
        "policy_frequency": 5,  # 5 decisions per simulation second                                                                                                                                           
    })

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

    # Set custom initial speed (must set all three for it to stick)
    vehicle = env.unwrapped.vehicle
    vehicle.speed = 1.0                      # Current speed (m/s)
    vehicle.speed_index = 2                  # Index into target_speeds [0, 2, 5, 10, 15, 20]
    vehicle.target_speed = 3.0               # Must match target_speeds[speed_index]
    
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
