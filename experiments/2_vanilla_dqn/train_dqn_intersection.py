"""
DQN Agent for Highway-env Intersection

Trains a DQN agent (stable-baselines3) to navigate the intersection
environment, then demonstrates the trained policy with video recording.

Usage:
    python train_dqn_intersection.py                        # Train with default config
    python train_dqn_intersection.py --config override.json  # Train with overrides
    python train_dqn_intersection.py --demo-only --run-dir data/runs/<run>
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
            if key == "env":
                # Shallow-merge env: only override top-level env keys,
                # leave observation/action sub-dicts untouched
                for env_key, env_value in value.items():
                    config["env"][env_key] = env_value
            else:
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


class TrainingPlotCallback(BaseCallback):
    """Saves a training progress plot each time SB3 logs verbose output."""

    def __init__(self, plot_path, episodes_csv, losses_csv,
                 log_interval=4, window=10):
        super().__init__()
        self.plot_path = plot_path
        self.episodes_csv = episodes_csv
        self.losses_csv = losses_csv
        self.log_interval = log_interval
        self.window = window
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_timesteps = []
        self.losses = []
        self.loss_steps = []
        self._ep_count = 0
        self._last_plot_ep = 0

    def _on_step(self):
        # Capture completed-episode stats from the Monitor wrapper
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.ep_rewards.append(ep_info["r"])
                self.ep_lengths.append(ep_info["l"])
                self.ep_timesteps.append(self.num_timesteps)
                self._ep_count += 1

        # Capture loss (recorded by DQN after each train step)
        loss = self.model.logger.name_to_value.get("train/loss")
        if loss is not None:
            if not self.losses or self.losses[-1] != loss:
                self.losses.append(loss)
                self.loss_steps.append(self.num_timesteps)

        # Update plot at same cadence as verbose output
        if (self._ep_count >= self._last_plot_ep + self.log_interval
                and len(self.ep_rewards) >= 2):
            self._last_plot_ep = self._ep_count
            self._save_plot()

        return True

    def _on_training_end(self):
        if self.ep_rewards:
            self._save_plot()

    # ------------------------------------------------------------------
    def _rolling_mean(self, data):
        arr = np.array(data, dtype=float)
        if len(arr) < self.window:
            return np.arange(1, len(arr) + 1), arr
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        means = (cs[self.window:] - cs[:-self.window]) / self.window
        x = np.arange(self.window, len(arr) + 1)
        return x, means

    def _save_plot(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        episodes = np.arange(1, len(self.ep_rewards) + 1)

        # --- ep_rew_mean ---
        axes[0].plot(episodes, self.ep_rewards, alpha=0.3, color="tab:blue")
        rx, rm = self._rolling_mean(self.ep_rewards)
        axes[0].plot(rx, rm, color="tab:blue",
                     label=f"rolling mean (w={self.window})")
        axes[0].set_ylabel("Episode Reward")
        axes[0].set_xlabel("Episode")
        axes[0].legend()
        axes[0].set_title("DQN Training Progress")
        axes[0].grid(True, alpha=0.3)

        # --- ep_len_mean ---
        axes[1].plot(episodes, self.ep_lengths, alpha=0.3, color="tab:orange")
        rx, rm = self._rolling_mean(self.ep_lengths)
        axes[1].plot(rx, rm, color="tab:orange",
                     label=f"rolling mean (w={self.window})")
        axes[1].set_ylabel("Episode Length")
        axes[1].set_xlabel("Episode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # --- loss ---
        if self.losses:
            axes[2].plot(self.loss_steps, self.losses, alpha=0.6,
                         color="tab:red")
        axes[2].set_ylabel("Loss")
        axes[2].set_xlabel("Timestep")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        fig.savefig(self.plot_path, dpi=100)
        plt.close(fig)
        self._save_csv()

    def _save_csv(self):
        os.makedirs(os.path.dirname(self.episodes_csv), exist_ok=True)

        with open(self.episodes_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length"])
            for i, (ts, r, l) in enumerate(
                zip(self.ep_timesteps, self.ep_rewards, self.ep_lengths), 1
            ):
                writer.writerow([i, ts, r, l])

        with open(self.losses_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "loss"])
            for ts, loss in zip(self.loss_steps, self.losses):
                writer.writerow([ts, loss])


def make_env(env_config):
    return gym.make("intersection-v1", render_mode="rgb_array", config=env_config)


def train(config, run_dir):
    env_config = config["env"]
    train_timesteps = config["train_timesteps"]
    eval_freq = config.get("eval_freq", 5000)

    model_path = os.path.join(run_dir, "models", "final_model")
    best_model_dir = os.path.join(run_dir, "models", "best")
    plot_path = os.path.join(run_dir, "training_progress.png")
    episodes_csv = os.path.join(run_dir, "episodes.csv")
    losses_csv = os.path.join(run_dir, "losses.csv")

    print("=== Training DQN on Intersection ===")
    print(f"Timesteps: {train_timesteps}")
    print(f"Run dir:   {run_dir}")
    print()

    env = make_env(env_config)
    eval_env = Monitor(make_env(env_config))

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15_000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        target_update_interval=50,
        verbose=1,
    )

    plot_cb = TrainingPlotCallback(plot_path, episodes_csv, losses_csv)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    model.learn(
        total_timesteps=train_timesteps,
        callback=CallbackList([plot_cb, eval_cb]),
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    env.close()
    eval_env.close()

    print()
    print(f"Final model saved to {model_path}.zip")
    print(f"Best model saved to {best_model_dir}/best_model.zip")
    print(f"Training plot saved to {plot_path}")

    best_path = os.path.join(best_model_dir, "best_model")
    if os.path.exists(best_path + ".zip"):
        return DQN.load(best_path)
    return model


def demo(model, env_config, run_dir):
    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")

    print()
    print(f"=== Demonstrating Trained Agent (3 episodes) ===")
    print()

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    env = make_env(env_config)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix="dqn_intersection",
    )
    env.unwrapped.set_record_video_wrapper(env)

    for ep in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
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
    print("Generated videos:")
    for f in sorted(os.listdir(video_folder)):
        size = os.path.getsize(os.path.join(video_folder, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")
    print()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="DQN agent for intersection env")
    parser.add_argument("--demo-only", action="store_true", help="Skip training, demo existing model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config with overrides")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to existing run dir (for --demo-only)")
    args = parser.parse_args()

    if args.demo_only:
        if args.run_dir:
            run_dir = args.run_dir
            run_config_path = os.path.join(run_dir, "config.json")
            with open(run_config_path) as f:
                config = json.load(f)
        else:
            config = load_config(args.config)
            run_dir = os.path.join(SCRIPT_DIR, "data")

        best_path = os.path.join(run_dir, "models", "best", "best_model")
        final_path = os.path.join(run_dir, "models", "final_model")
        if os.path.exists(best_path + ".zip"):
            print(f"Loading best model from {best_path}.zip")
            model = DQN.load(best_path)
        elif os.path.exists(final_path + ".zip"):
            print(f"Loading final model from {final_path}.zip")
            model = DQN.load(final_path)
        else:
            print(f"Error: No model found in {run_dir}/models/")
            return

        demo(model, config["env"], run_dir)
    else:
        config = load_config(args.config)
        run_dir = make_run_dir(config)
        model = train(config, run_dir)
        demo(model, config["env"], run_dir)


if __name__ == "__main__":
    main()
