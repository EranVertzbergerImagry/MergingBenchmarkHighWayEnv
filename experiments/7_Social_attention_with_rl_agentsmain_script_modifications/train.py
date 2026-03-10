"""
Social Attention DQN — Experiment 7: Ego-Centric Random Destination Training

Extended from experiment 6 with:
  - Unified config (env + agent inline in one JSON)
  - Ego-centric observation wrapper (heading-invariant policy)
  - Random destination training (south entry, random exits)
  - Random spawn at inference (--random-spawn flag)

Usage:
    python train.py                                              # Train with defaults
    python train.py --config override.json                       # Override config
    python train.py --episodes 8000                              # Override episode count
    python train.py --test-only --recover-from checkpoint.tar    # Eval only
    python train.py --evaluation-only --recover-from ckpt.tar --random-spawn
    python train.py --demo-only --recover-from ckpt.tar          # Record videos
    python train.py --visualize                                  # Attention viz
"""
import os
import sys
import json
import csv
import argparse
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    from rl_agents.agents.common.factory import load_environment, load_agent
    from rl_agents.trainer.evaluation import Evaluation
    from rl_agents.configuration import Configurable
except ImportError:
    print(
        "ERROR: rl-agents is not installed.\n"
        "Install it with:\n"
        "    bash setup.sh\n"
    )
    sys.exit(1)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src import EgoCentricWrapper, TrainingTracker, generate_training_plot
import src  # noqa: F401 — triggers gym.register via src/__init__.py

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_agent_config(config_dict):
    """Resolve base_config paths if loading from a file-based agent config.

    For the unified config workflow, the agent dict is passed directly
    and typically has no base_config key.
    """
    if isinstance(config_dict, str):
        config_path = os.path.abspath(config_dict)
        with open(config_path) as f:
            config_dict = json.load(f)
        if "base_config" in config_dict:
            base_path = os.path.join(os.path.dirname(config_path), config_dict["base_config"])
            base_config = load_agent_config(base_path)
            del config_dict["base_config"]
            config_dict = Configurable.rec_update(base_config, config_dict)
    return config_dict


def load_config(config_path=None):
    """Load unified config from file. Falls back to default_config.json in script dir."""
    default_path = os.path.join(SCRIPT_DIR, "default_config.json")
    with open(default_path) as f:
        config = json.load(f)

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

    return config


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_env(env_config):
    """Create environment from the env config dict and wrap with EgoCentricWrapper."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=SCRIPT_DIR
    ) as f:
        json.dump(env_config, f)
        tmp_path = f.name
    try:
        env = load_environment(tmp_path)
    finally:
        os.unlink(tmp_path)

    env = EgoCentricWrapper(env)
    return env


def make_env_for_rendering(env_config, render_mode):
    """Create environment with a specific render_mode and ego-centric wrapper."""
    config = dict(env_config)
    env_id = config.pop("id")
    config.pop("import_module", None)
    import highway_env  # noqa: F401
    env = gym.make(env_id, render_mode=render_mode, config=config)
    env = EgoCentricWrapper(env)
    return env


def make_run_dir(config):
    """Create a timestamped run directory and save the merged config into it."""
    data_dir = os.path.join(SCRIPT_DIR, "data")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    description = config.get("description", "default")
    run_name = f"{timestamp}_{description}"
    run_dir = os.path.join(data_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return run_dir


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(config, run_dir, visualize=False):
    env_config = config["env"]
    agent_config = config["agent"]
    train_episodes = config["train_episodes"]

    episodes_csv = os.path.join(run_dir, "episodes.csv")
    plot_path = os.path.join(run_dir, "training_progress.html")

    print("=== Training with rl-agents framework ===")
    print(f"Env id:        {env_config['id']}")
    print(f"Episodes:      {train_episodes}")
    print(f"Run dir:       {run_dir}")
    print(f"Obs features:  {env_config['observation']['features']}")
    print()

    env = make_env(env_config)
    agent = load_agent(load_agent_config(agent_config), env)

    tracker = TrainingTracker(episodes_csv)

    evaluation = Evaluation(
        env,
        agent,
        directory=os.path.dirname(run_dir),
        run_directory=os.path.basename(run_dir),
        num_episodes=train_episodes,
        training=True,
        display_env=visualize,
        display_agent=visualize,
        display_rewards=False,
        step_callback_fn=tracker.step_callback,
    )

    evaluation.train()
    evaluation.close()

    print()
    print(f"  episodes.csv written to {episodes_csv}")

    generate_training_plot(
        episodes_csv, plot_path,
        description=config.get("description", ""),
    )

    return agent, env_config


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def run_episodes(agent, env, num_episodes):
    """Run episodes with agent in eval mode and return per-episode results."""
    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        try:
            vehicle = env.unwrapped.vehicle
            route = vehicle.route
            destination = route[-1][1] if route else "?"
        except (AttributeError, IndexError):
            destination = "?"

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        crashed = info.get("crashed", False)
        arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
        if not arrived:
            arrived = info.get("is_success", False)
        results.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": total_reward,
            "crashed": crashed,
            "arrived": arrived,
            "destination": destination,
        })
        status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")
        print(f"  Episode {ep + 1}: {status} | dest={destination} | Steps: {steps} | Reward: {total_reward:.2f}")

    return results


def evaluate_agent(agent, env_config, run_dir, num_episodes, eval_csv=None):
    """Post-training evaluation writing evaluation.csv with per-destination breakdown."""
    if eval_csv is None:
        eval_csv = os.path.join(run_dir, "evaluation.csv")

    print()
    print(f"=== Evaluating Trained Agent ({num_episodes} episodes) ===")
    print()

    env = make_env(env_config)
    agent.eval()
    results = run_episodes(agent, env, num_episodes)
    env.close()

    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "steps", "reward", "crashed", "arrived", "destination"]
        )
        writer.writeheader()
        writer.writerows(results)

    n_crashed = sum(r["crashed"] for r in results)
    n_arrived = sum(r["arrived"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])

    # Per-destination breakdown
    dest_stats = defaultdict(lambda: {"total": 0, "arrived": 0, "crashed": 0, "stall": 0})
    for r in results:
        d = r["destination"]
        dest_stats[d]["total"] += 1
        if r["arrived"]:
            dest_stats[d]["arrived"] += 1
        elif r["crashed"]:
            dest_stats[d]["crashed"] += 1
        else:
            dest_stats[d]["stall"] += 1

    with open(eval_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow([
            "SUMMARY", num_episodes, f"{avg_reward:.2f}",
            f"{100 * n_crashed / num_episodes:.0f}%",
            f"{100 * n_arrived / num_episodes:.0f}%",
        ])
        writer.writerow([])
        writer.writerow(["destination", "episodes", "arrived%", "crashed%", "stall%"])
        for dest in sorted(dest_stats.keys()):
            s = dest_stats[dest]
            n = s["total"]
            writer.writerow([
                dest, n,
                f"{100 * s['arrived'] / n:.0f}%",
                f"{100 * s['crashed'] / n:.0f}%",
                f"{100 * s['stall'] / n:.0f}%",
            ])

    print()
    print(
        f"  Arrived: {n_arrived}/{num_episodes} "
        f"({100 * n_arrived / num_episodes:.0f}%)  "
        f"Crashed: {n_crashed}/{num_episodes} "
        f"({100 * n_crashed / num_episodes:.0f}%)  "
        f"Avg reward: {avg_reward:.2f}"
    )
    print()
    print(f"  {'Dest':<6} {'Episodes':>8} {'Arrived':>8} {'Crashed':>8} {'Stall':>8}")
    print(f"  {'-'*38}")
    for dest in sorted(dest_stats.keys()):
        s = dest_stats[dest]
        n = s["total"]
        print(f"  {dest:<6} {n:>8} {100*s['arrived']/n:>7.0f}% {100*s['crashed']/n:>7.0f}% {100*s['stall']/n:>7.0f}%")
    print()
    print(f"  Saved to {eval_csv}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo(agent, env_config, num_episodes):
    """Record demo videos with attention overlay using RecordVideo wrapper."""
    from rl_agents.agents.common.graphics import AgentGraphics

    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")

    print()
    print(f"=== Recording Demo Videos ({num_episodes} episodes) ===")
    print()

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    env = make_env_for_rendering(env_config, render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix="rl_agents_ego_attn",
    )
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except AttributeError:
        pass

    agent.env = env
    agent.eval()

    from highway_env.envs.common.graphics import EnvViewer
    EnvViewer.agent_display = (
        lambda agent_surface, sim_surface:
            AgentGraphics.display(agent, agent_surface, sim_surface)
    )

    for ep in range(num_episodes):
        obs, info = env.reset()

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        crashed = info.get("crashed", False)
        arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
        if not arrived:
            arrived = info.get("is_success", False)
        status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")
        print(f"  Episode {ep + 1}: {status} | Steps: {steps} | Reward: {total_reward:.2f}")

    env.close()
    EnvViewer.agent_display = None

    print()
    print("Generated videos:")
    for f in sorted(os.listdir(video_folder)):
        size = os.path.getsize(os.path.join(video_folder, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Visualize (attention overlay in pygame window)
# ---------------------------------------------------------------------------
def visualize_agent(agent, env_config, num_episodes):
    """Run episodes with render_mode='human' and attention overlay."""
    from rl_agents.agents.common.graphics import AgentGraphics

    print()
    print(f"=== Visualizing Agent ({num_episodes} episodes) ===")
    print("  Close the pygame window or press Ctrl+C to stop.")
    print()

    env = make_env_for_rendering(env_config, render_mode="human")

    agent.env = env
    agent.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        if ep == 0:
            try:
                env.unwrapped.viewer.set_agent_display(
                    lambda agent_surface, sim_surface:
                        AgentGraphics.display(agent, agent_surface, sim_surface)
                )
            except AttributeError:
                print("  Warning: viewer does not support agent display overlay")
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        crashed = info.get("crashed", False)
        arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
        status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")
        print(f"  Episode {ep + 1}: {status} | Steps: {steps} | Reward: {total_reward:.2f}")

    env.close()
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Social Attention DQN — Experiment 7"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config with overrides",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override number of training episodes",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip training, evaluate/demo an existing checkpoint",
    )
    parser.add_argument(
        "--recover-from", type=str, default=None,
        help="Path to a .tar checkpoint (for --test-only or to resume training)",
    )
    parser.add_argument(
        "--demo-only", action="store_true",
        help="Skip training and evaluation, just record demo videos",
    )
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help="Skip training, evaluate existing checkpoint and save timestamped CSV",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable rl-agents built-in attention overlay (render_mode='human')",
    )
    parser.add_argument(
        "--random-spawn", action="store_true",
        help="Set spawn_entry=null for random spawn location (inference use)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.episodes is not None:
        config["train_episodes"] = args.episodes

    if args.random_spawn:
        config["env"]["spawn_entry"] = None

    env_config = config["env"]
    agent_config = config["agent"]

    if args.test_only or args.demo_only or args.evaluation_only:
        if not args.recover_from:
            parser.error("--test-only / --demo-only / --evaluation-only requires --recover-from <checkpoint.tar>")
        if not os.path.exists(args.recover_from):
            print(f"Error: Checkpoint not found: {args.recover_from}")
            return

        # Derive run_dir from checkpoint path
        p = os.path.dirname(os.path.abspath(args.recover_from))
        run_dir = p  # fallback
        while p != os.path.dirname(p):
            if os.path.basename(p) == "saved_models":
                run_dir = os.path.dirname(p)
                break
            p = os.path.dirname(p)

        print(f"Loading agent from {args.recover_from}")

        env = make_env(env_config)
        agent = load_agent(load_agent_config(agent_config), env)
        agent.load(args.recover_from)

        if args.visualize:
            visualize_agent(agent, env_config, config.get("demo_episodes", 3))
        elif args.demo_only:
            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_config, demo_episodes)
        elif args.evaluation_only:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            eval_csv = os.path.join(run_dir, f"evaluation_{timestamp}.csv")
            eval_episodes = config.get("eval_episodes", 100)
            evaluate_agent(agent, env_config, run_dir, eval_episodes,
                           eval_csv=eval_csv)
        else:
            eval_episodes = config.get("eval_episodes", 100)
            evaluate_agent(agent, env_config, run_dir, eval_episodes)

            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_config, demo_episodes)
    else:
        # Normal training flow
        run_dir = make_run_dir(config)

        agent, env_cfg = train(
            config, run_dir, visualize=args.visualize,
        )

        eval_episodes = config.get("eval_episodes", 100)
        evaluate_agent(agent, env_cfg, run_dir, eval_episodes)

        if config.get("demo_on_train_end", True):
            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_cfg, demo_episodes)


if __name__ == "__main__":
    main()
