"""
Random Policy Evaluation for Highway-env Intersection

Runs a configurable number of episodes with a random action policy and
produces an evaluation.csv identical in format to the trained-agent
evaluation files.

Usage:
    python eval_random_policy.py                        # Evaluate with default config
    python eval_random_policy.py --config override.json  # Evaluate with overrides
    python eval_random_policy.py --demo                  # Record demo videos
    python eval_random_policy.py --demo --config override.json
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import os
import shutil
import argparse
import json
import csv
from datetime import datetime
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path=None):
    """Load config from file. Falls back to default_config.json in script dir."""
    default_path = os.path.join(SCRIPT_DIR, "default_config.json")
    with open(default_path) as f:
        config = json.load(f)

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if key == "env":
                for env_key, env_value in value.items():
                    config["env"][env_key] = env_value
            else:
                config[key] = value

    return config


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


def run_episodes(env, num_episodes):
    """Run episodes with random actions and return per-episode results."""
    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        crashed = info.get("crashed", False)
        arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
        results.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": total_reward,
            "crashed": crashed,
            "arrived": arrived,
        })
        status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")
        print(f"  Episode {ep + 1}: {status} | Steps: {steps} | Reward: {total_reward:.2f}")

    return results


def demo(env_config, num_episodes):
    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")

    print()
    print(f"=== Recording Random Policy Demo Videos ({num_episodes} episodes) ===")
    print()

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make("intersection-v1", render_mode="rgb_array", config=env_config)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix="random_intersection",
    )
    env.unwrapped.set_record_video_wrapper(env)

    run_episodes(env, num_episodes)
    env.close()

    print()
    print("Generated videos:")
    for f in sorted(os.listdir(video_folder)):
        size = os.path.getsize(os.path.join(video_folder, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")
    print()
    print("Done!")


def evaluate(env_config, num_episodes):
    data_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_csv = os.path.join(data_dir, f"random_policy_evaluation_{timestamp}.csv")

    print()
    print(f"=== Evaluating Random Policy ({num_episodes} episodes) ===")
    print()

    env = gym.make("intersection-v1", render_mode="rgb_array", config=env_config)
    results = run_episodes(env, num_episodes)
    env.close()

    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward", "crashed", "arrived"])
        writer.writeheader()
        writer.writerows(results)

    n_crashed = sum(r["crashed"] for r in results)
    n_arrived = sum(r["arrived"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])

    with open(eval_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SUMMARY", num_episodes, f"{avg_reward:.2f}",
            f"{100*n_crashed/num_episodes:.0f}%",
            f"{100*n_arrived/num_episodes:.0f}%",
        ])

    print()
    print(f"  Arrived: {n_arrived}/{num_episodes} ({100*n_arrived/num_episodes:.0f}%)  "
          f"Crashed: {n_crashed}/{num_episodes} ({100*n_crashed/num_episodes:.0f}%)  "
          f"Avg reward: {avg_reward:.2f}")
    print(f"  Saved to {eval_csv}")


def main():
    parser = argparse.ArgumentParser(description="Random policy evaluation for intersection env")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config with overrides")
    parser.add_argument("--demo", action="store_true", help="Record demo videos with random policy")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("idm"):
        apply_idm_params(config["idm"])
    env_config = config["env"]

    if args.demo:
        demo_episodes = config.get("demo_episodes", 3)
        demo(env_config, demo_episodes)
    else:
        num_episodes = config.get("eval_episodes", 100)
        evaluate(env_config, num_episodes)


if __name__ == "__main__":
    main()
