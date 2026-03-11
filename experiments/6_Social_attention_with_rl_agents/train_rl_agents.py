"""
Social Attention DQN using the rl-agents framework

Reproduces the exact training setup from:
    "Social Attention for Autonomous Decision-Making in Dense Traffic"
    (Leurent & Mercat, 2019)

Uses the authors' own rl-agents library with built-in DQN,
EgoAttentionNetwork, and attention visualization — the same harness
used to produce the paper's results.

Usage:
    python train_rl_agents.py                                              # Train with defaults
    python train_rl_agents.py --config override.json                       # Override config
    python train_rl_agents.py --episodes 8000                              # Override episode count
    python train_rl_agents.py --test-only --recover-from checkpoint.tar    # Eval only
    python train_rl_agents.py --visualize                                  # Enable attention viz

Dependency:
    pip install git+https://github.com/eleurent/rl-agents
"""
import os
import sys
import json
import csv
import argparse
import shutil
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
        "    bash experiments/6_Social_attention_with_rl_agents/setup.sh\n"
    )
    sys.exit(1)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_agent_config(config_path):
    """Load agent JSON config, resolving base_config paths relative to each file's directory."""
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        agent_config = json.load(f)
    if "base_config" in agent_config:
        base_path = os.path.join(os.path.dirname(config_path), agent_config["base_config"])
        base_config = load_agent_config(base_path)
        del agent_config["base_config"]
        agent_config = Configurable.rec_update(base_config, agent_config)
    return agent_config


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path=None):
    """Load config from file. Falls back to default_config.json in script dir."""
    default_path = os.path.join(SCRIPT_DIR, "default_config.json")
    with open(default_path) as f:
        config = json.load(f)

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            config[key] = value

    return config


def make_run_dir(config):
    """Create a timestamped run directory and save the merged config into it."""
    data_dir = os.path.join(SCRIPT_DIR, "data")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    description = config.get("description", "default")
    run_name = f"{timestamp}_{description}"
    run_dir = os.path.join(data_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save merged config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return run_dir


# ---------------------------------------------------------------------------
# Training tracker  (step callback for rl-agents Evaluation)
# ---------------------------------------------------------------------------
class TrainingTracker:
    """Step callback that logs per-episode metrics to episodes.csv.

    Passed to ``Evaluation(step_callback_fn=tracker.step_callback)``.
    Accumulates per-step rewards and writes a row on episode end.
    Optionally updates a training_progress.html plot every ``plot_interval`` episodes.
    """

    def __init__(self, episodes_csv, plot_path=None, description="",
                 plot_interval=50):
        self.episodes_csv = episodes_csv
        self.plot_path = plot_path
        self.description = description
        self.plot_interval = plot_interval
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._current_ep = -1
        self._header_written = False
        self._last_plot_ep = 0

    def step_callback(self, episode, env, agent, transition, writer):
        obs, reward, terminated, truncated, info = transition

        if episode != self._current_ep:
            # New episode started — reset accumulators
            self._current_ep = episode
            self._ep_reward = 0.0
            self._ep_steps = 0

        self._ep_reward += reward
        self._ep_steps += 1

        if terminated or truncated:
            crashed = info.get("crashed", False)
            arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
            if not arrived:
                arrived = info.get("is_success", False)
            self._write_row(
                episode + 1, self._ep_reward, self._ep_steps,
                crashed, arrived,
            )
            self._maybe_update_plot(episode + 1)

    def _write_row(self, episode, reward, length, crashed, arrived):
        with open(self.episodes_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["episode", "reward", "length", "crashed", "arrived"])
                self._header_written = True
            writer.writerow([episode, f"{reward:.4f}", length, crashed, arrived])

    def _maybe_update_plot(self, episode):
        if self.plot_path and episode >= self._last_plot_ep + self.plot_interval:
            self._last_plot_ep = episode
            try:
                generate_training_plot(
                    self.episodes_csv, self.plot_path,
                    description=self.description, auto_refresh=True,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Training plot generation
# ---------------------------------------------------------------------------
def generate_training_plot(episodes_csv, plot_path, description="", window=50,
                           auto_refresh=False):
    """Read episodes.csv and generate an interactive Plotly HTML chart.

    Args:
        auto_refresh: If True, inject a <meta> tag that refreshes the browser
                      every 30 seconds (useful during training).
    """
    if not HAS_PLOTLY:
        print("  (plotly not installed — skipping training plot)")
        return

    episodes, rewards, lengths = [], [], []
    try:
        with open(episodes_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                lengths.append(int(row["length"]))
    except (OSError, KeyError):
        print("  (could not read episodes.csv — skipping training plot)")
        return

    if len(episodes) < 2:
        return

    episodes = np.array(episodes)
    rewards = np.array(rewards)
    lengths = np.array(lengths)

    def rolling_mean(data, w):
        if len(data) < w:
            return np.arange(1, len(data) + 1), data
        cs = np.cumsum(data)
        cs = np.insert(cs, 0, 0.0)
        means = (cs[w:] - cs[:-w]) / w
        x = np.arange(w, len(data) + 1)
        return x, means

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        subplot_titles=("Episode Reward", "Episode Length"),
        vertical_spacing=0.12,
    )

    # --- Episode Reward ---
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, mode="lines",
        name="Reward", opacity=0.3, line=dict(color="royalblue"),
        hovertemplate="Ep %{x}<br>Reward: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    rx, rm = rolling_mean(rewards, window)
    fig.add_trace(go.Scatter(
        x=rx, y=rm, mode="lines",
        name=f"Reward (mean {window})", line=dict(color="royalblue", width=2),
        hovertemplate="Ep %{x}<br>Mean: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # --- Episode Length ---
    fig.add_trace(go.Scatter(
        x=episodes, y=lengths, mode="lines",
        name="Length", opacity=0.3, line=dict(color="darkorange"),
        hovertemplate="Ep %{x}<br>Length: %{y}<extra></extra>",
    ), row=2, col=1)
    rx, rm = rolling_mean(lengths, window)
    fig.add_trace(go.Scatter(
        x=rx, y=rm, mode="lines",
        name=f"Length (mean {window})", line=dict(color="darkorange", width=2),
        hovertemplate="Ep %{x}<br>Mean: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Length", row=2, col=1)

    fig.update_layout(
        title=f"rl-agents EgoAttention DQN — {description}",
        height=700, width=1000,
        hovermode="x unified",
    )

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    html = fig.to_html()
    if auto_refresh:
        html = html.replace("<head>",
                            '<head><meta http-equiv="refresh" content="30">', 1)
    with open(plot_path, "w") as f:
        f.write(html)
    print(f"  Training plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(config, run_dir, visualize=False):
    env_config_path = os.path.join(SCRIPT_DIR, config["env_config_path"])
    agent_config_path = os.path.join(SCRIPT_DIR, config["agent_config_path"])
    train_episodes = config["train_episodes"]

    episodes_csv = os.path.join(run_dir, "episodes.csv")
    plot_path = os.path.join(run_dir, "training_progress.html")

    print("=== Training with rl-agents framework ===")
    print(f"Agent config:  {agent_config_path}")
    print(f"Env config:    {env_config_path}")
    print(f"Episodes:      {train_episodes}")
    print(f"Run dir:       {run_dir}")
    print()

    env = load_environment(env_config_path)
    agent = load_agent(load_agent_config(agent_config_path), env)

    tracker = TrainingTracker(
        episodes_csv,
        plot_path=plot_path,
        description=config.get("description", ""),
    )

    # Evaluation puts all outputs (checkpoints, TensorBoard, videos, metadata)
    # into directory/run_directory.
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

    # Final plot without auto-refresh
    generate_training_plot(
        episodes_csv, plot_path,
        description=config.get("description", ""),
    )

    return agent, env_config_path


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def run_episodes(agent, env, num_episodes):
    """Run episodes with agent in eval mode and return per-episode results."""
    results = []
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


def evaluate_agent(agent, env_config_path, run_dir, num_episodes, eval_csv=None):
    """Post-training evaluation writing evaluation.csv with SUMMARY row."""
    if eval_csv is None:
        eval_csv = os.path.join(run_dir, "evaluation.csv")

    print()
    print(f"=== Evaluating Trained Agent ({num_episodes} episodes) ===")
    print()

    env = load_environment(env_config_path)
    agent.eval()
    results = run_episodes(agent, env, num_episodes)
    env.close()

    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "steps", "reward", "crashed", "arrived"]
        )
        writer.writeheader()
        writer.writerows(results)

    n_crashed = sum(r["crashed"] for r in results)
    n_arrived = sum(r["arrived"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])

    with open(eval_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SUMMARY", num_episodes, f"{avg_reward:.2f}",
            f"{100 * n_crashed / num_episodes:.0f}%",
            f"{100 * n_arrived / num_episodes:.0f}%",
        ])

    print()
    print(
        f"  Arrived: {n_arrived}/{num_episodes} "
        f"({100 * n_arrived / num_episodes:.0f}%)  "
        f"Crashed: {n_crashed}/{num_episodes} "
        f"({100 * n_crashed / num_episodes:.0f}%)  "
        f"Avg reward: {avg_reward:.2f}"
    )
    print(f"  Saved to {eval_csv}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo(agent, env_config_path, num_episodes):
    """Record demo videos with attention overlay using RecordVideo wrapper."""
    from rl_agents.agents.common.graphics import AgentGraphics

    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")

    print()
    print(f"=== Recording Demo Videos ({num_episodes} episodes) ===")
    print()

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    # Create env with rgb_array rendering so videos capture the pygame surface
    with open(env_config_path) as f:
        env_config = json.load(f)
    env_id = env_config.pop("id")
    env_config.pop("import_module", None)
    import highway_env  # noqa: F401
    env = gym.make(env_id, render_mode="rgb_array", config=env_config)

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

    # Set attention overlay at the class level BEFORE first reset,
    # so all frames (including the first) have the same size.
    from highway_env.envs.common.graphics import EnvViewer
    EnvViewer.agent_display = (
        lambda agent_surface, sim_surface:
            AgentGraphics.display(agent, agent_surface, sim_surface)
    )

    results = []
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
def visualize_agent(agent, env_config_path, num_episodes):
    """Run episodes with render_mode='human' and attention overlay."""
    from rl_agents.agents.common.graphics import AgentGraphics

    print()
    print(f"=== Visualizing Agent ({num_episodes} episodes) ===")
    print("  Close the pygame window or press Ctrl+C to stop.")
    print()

    # Create env with human rendering
    with open(env_config_path) as f:
        env_config = json.load(f)
    env_id = env_config.pop("id")
    env_config.pop("import_module", None)
    import highway_env  # noqa: F401
    env = gym.make(env_id, render_mode="human", config=env_config)

    agent.env = env
    agent.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        # Set up attention overlay after first reset (viewer exists now)
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
        description="Social Attention DQN using the rl-agents framework"
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
    args = parser.parse_args()

    config = load_config(args.config)

    if args.episodes is not None:
        config["train_episodes"] = args.episodes

    env_config_path = os.path.join(SCRIPT_DIR, config["env_config_path"])
    agent_config_path = os.path.join(SCRIPT_DIR, config["agent_config_path"])

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

        env = load_environment(env_config_path)
        agent = load_agent(load_agent_config(agent_config_path), env)
        agent.load(args.recover_from)

        if args.visualize:
            visualize_agent(agent, env_config_path, config.get("demo_episodes", 3))
        elif args.demo_only:
            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_config_path, demo_episodes)
        elif args.evaluation_only:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            eval_csv = os.path.join(run_dir, f"evaluation_{timestamp}.csv")
            eval_episodes = config.get("eval_episodes", 100)
            evaluate_agent(agent, env_config_path, run_dir, eval_episodes,
                           eval_csv=eval_csv)
        else:
            eval_episodes = config.get("eval_episodes", 100)
            evaluate_agent(agent, env_config_path, run_dir, eval_episodes)

            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_config_path, demo_episodes)
    else:
        # Normal training flow
        run_dir = make_run_dir(config)

        agent, env_cfg_path = train(
            config, run_dir, visualize=args.visualize,
        )

        eval_episodes = config.get("eval_episodes", 100)
        evaluate_agent(agent, env_cfg_path, run_dir, eval_episodes)

        if config.get("demo_on_train_end", True):
            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_cfg_path, demo_episodes)


if __name__ == "__main__":
    main()
