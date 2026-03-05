"""
DQN with Social Attention for Highway-env Intersection

Trains a DQN agent using the EgoAttention architecture from:
    "Social Attention for Autonomous Decision-Making in Dense Traffic"
    (Leurent & Mercat, 2019)

The attention mechanism lets the ego vehicle learn which nearby vehicles
are most relevant for its decision, producing a permutation-invariant
and interpretable policy.

Usage:
    python train_social_attention_dqn.py                        # Train with default config
    python train_social_attention_dqn.py --config override.json  # Train with overrides
    python train_social_attention_dqn.py --demo-only --model-path data/runs/<run>/models/best/best_model.zip --config <config>
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from social_attention_model import SocialAttentionExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Social Attention model config  (matches ego_attention_2h.json from rl-agents)
# ---------------------------------------------------------------------------
POLICY_KWARGS = dict(
    features_extractor_class=SocialAttentionExtractor,
    features_extractor_kwargs=dict(
        embedding_layers=[64, 64],
        others_embedding_layers=[64, 64],
        attention_feature_size=64,
        attention_heads=2,
        self_attention=False,
        output_layers=[64, 64],
        presence_feature_idx=0,
    ),
    net_arch=[],  # no extra MLP — output_layer is inside the extractor
)


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


# ---------------------------------------------------------------------------
# Training plot callback  (same as experiment 2)
# ---------------------------------------------------------------------------
class TrainingPlotCallback(BaseCallback):
    """Saves a training progress plot each time SB3 logs verbose output."""

    def __init__(self, plot_path, episodes_csv, losses_csv, config,
                 log_interval=4, window=10):
        super().__init__()
        self.plot_path = plot_path
        self.episodes_csv = episodes_csv
        self.losses_csv = losses_csv
        self.config = config
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
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.ep_rewards.append(ep_info["r"])
                self.ep_lengths.append(ep_info["l"])
                self.ep_timesteps.append(self.num_timesteps)
                self._ep_count += 1

        loss = self.model.logger.name_to_value.get("train/loss")
        if loss is not None:
            if not self.losses or self.losses[-1] != loss:
                self.losses.append(loss)
                self.loss_steps.append(self.num_timesteps)

        if (self._ep_count >= self._last_plot_ep + self.log_interval
                and len(self.ep_rewards) >= 2):
            self._last_plot_ep = self._ep_count
            self._save_plot()

        return True

    def _on_training_end(self):
        if self.ep_rewards:
            self._save_plot(auto_refresh=False)

    def _rolling_mean(self, data):
        arr = np.array(data, dtype=float)
        if len(arr) < self.window:
            return np.arange(1, len(arr) + 1), arr
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        means = (cs[self.window:] - cs[:-self.window]) / self.window
        x = np.arange(self.window, len(arr) + 1)
        return x, means

    def _save_plot(self, auto_refresh=True):
        episodes = np.arange(1, len(self.ep_rewards) + 1)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=False,
            subplot_titles=("Episode Reward", "Episode Length", "Loss"),
            vertical_spacing=0.08,
        )

        # --- Episode Reward ---
        fig.add_trace(go.Scatter(
            x=episodes, y=self.ep_rewards, mode="lines",
            name="Reward", opacity=0.3, line=dict(color="royalblue"),
            hovertemplate="Ep %{x}<br>Reward: %{y:.2f}<extra></extra>",
        ), row=1, col=1)
        rx, rm = self._rolling_mean(self.ep_rewards)
        fig.add_trace(go.Scatter(
            x=rx, y=rm, mode="lines",
            name=f"Reward (mean {self.window})", line=dict(color="royalblue", width=2),
            hovertemplate="Ep %{x}<br>Mean: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # --- Episode Length ---
        fig.add_trace(go.Scatter(
            x=episodes, y=self.ep_lengths, mode="lines",
            name="Length", opacity=0.3, line=dict(color="darkorange"),
            hovertemplate="Ep %{x}<br>Length: %{y}<extra></extra>",
        ), row=2, col=1)
        rx, rm = self._rolling_mean(self.ep_lengths)
        fig.add_trace(go.Scatter(
            x=rx, y=rm, mode="lines",
            name=f"Length (mean {self.window})", line=dict(color="darkorange", width=2),
            hovertemplate="Ep %{x}<br>Mean: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

        # --- Loss ---
        if self.losses:
            fig.add_trace(go.Scatter(
                x=self.loss_steps, y=self.losses, mode="lines",
                name="Loss", opacity=0.6, line=dict(color="firebrick"),
                hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ), row=3, col=1)

        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Timestep", row=3, col=1)
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Length", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=3, col=1)

        env = self.config["env"]
        subtitle = (
            f"collision_reward={env['collision_reward']}  "
            f"high_speed_reward={env['high_speed_reward']}  "
            f"arrived_reward={env['arrived_reward']}"
        )
        desc = self.config.get("description", "default")
        fig.update_layout(
            title=(
                f"Social Attention DQN — {desc}<br>"
                f"<sup>{subtitle}</sup>"
            ),
            height=900, width=1000,
            hovermode="x unified",
        )

        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        html = fig.to_html()
        if auto_refresh:
            html = html.replace("<head>",
                                '<head><meta http-equiv="refresh" content="30">', 1)
        with open(self.plot_path, "w") as f:
            f.write(html)
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


# ---------------------------------------------------------------------------
# Eval log callback
# ---------------------------------------------------------------------------
class EvalMetricsWrapper(gym.Wrapper):
    """Tracks crash and arrival counts on the eval env."""

    def __init__(self, env):
        super().__init__(env)
        self.crash_count = 0
        self.arrive_count = 0
        self.episode_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.episode_count += 1
            if info.get("crashed", False):
                self.crash_count += 1
            if info.get("rewards", {}).get("arrived_reward", 0) > 0:
                self.arrive_count += 1
        return obs, reward, terminated, truncated, info


class EvalLogCallback(BaseCallback):
    """Logs each EvalCallback evaluation to training_eval_log.csv.

    Used as ``callback_after_eval`` on an EvalCallback so that
    ``self.parent`` is the EvalCallback instance.
    """

    def __init__(self, log_path, eval_metrics):
        super().__init__()
        self.log_path = log_path
        self.eval_metrics = eval_metrics
        self._header_written = False
        self._prev_best = -np.inf
        self._prev_episodes = 0
        self._prev_crashes = 0
        self._prev_arrivals = 0

    def _on_step(self):
        eval_cb = self.parent
        mean_reward = eval_cb.last_mean_reward
        std_reward = float(np.std(eval_cb.evaluations_results[-1]))
        mean_length = float(np.mean(eval_cb.evaluations_length[-1]))
        best_mean = eval_cb.best_mean_reward

        model_saved = "best_model" if best_mean > self._prev_best else ""
        self._prev_best = best_mean

        episodes = self.eval_metrics.episode_count - self._prev_episodes
        crashes = self.eval_metrics.crash_count - self._prev_crashes
        arrivals = self.eval_metrics.arrive_count - self._prev_arrivals
        self._prev_episodes = self.eval_metrics.episode_count
        self._prev_crashes = self.eval_metrics.crash_count
        self._prev_arrivals = self.eval_metrics.arrive_count

        crash_pct = f"{100 * crashes / episodes:.0f}%" if episodes > 0 else ""
        arrive_pct = f"{100 * arrivals / episodes:.0f}%" if episodes > 0 else ""

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow([
                    "timestep", "mean_reward", "std_reward",
                    "mean_length", "best_mean_reward", "model_saved",
                    "crash_rate", "arrival_rate",
                ])
                self._header_written = True
            writer.writerow([
                self.num_timesteps,
                f"{mean_reward:.2f}",
                f"{std_reward:.2f}",
                f"{mean_length:.1f}",
                f"{best_mean:.2f}",
                model_saved,
                crash_pct,
                arrive_pct,
            ])
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def make_env(env_config):
    return gym.make("intersection-v1", render_mode="rgb_array", config=env_config)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(config, run_dir):
    env_config = config["env"]
    train_timesteps = config["train_timesteps"]
    eval_freq = config.get("eval_freq", 5000)

    model_path = os.path.join(run_dir, "models", "social_attention_dqn")
    best_model_dir = os.path.join(run_dir, "models", "best")
    plot_path = os.path.join(run_dir, "training_progress.html")
    episodes_csv = os.path.join(run_dir, "episodes.csv")
    losses_csv = os.path.join(run_dir, "losses.csv")

    print("=== Training Social Attention DQN on Intersection ===")
    print(f"Timesteps: {train_timesteps}")
    print(f"Run dir:   {run_dir}")
    print()

    mc = config.get("model", {})

    env = make_env(env_config)
    eval_metrics = EvalMetricsWrapper(make_env(env_config))
    eval_env = Monitor(eval_metrics)

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=POLICY_KWARGS,
        learning_rate=mc.get("learning_rate", 5e-4),
        buffer_size=mc.get("buffer_size", 15_000),
        learning_starts=mc.get("learning_starts", 200),
        batch_size=mc.get("batch_size", 64),
        gamma=mc.get("gamma", 0.95),
        train_freq=mc.get("train_freq", 1),
        target_update_interval=mc.get("target_update_interval", 512),
        exploration_fraction=mc.get("exploration_fraction", 0.3),
        exploration_final_eps=mc.get("exploration_final_eps", 0.05),
        verbose=1,
    )

    eval_log_csv = os.path.join(run_dir, "training_eval_log.csv")

    plot_cb = TrainingPlotCallback(plot_path, episodes_csv, losses_csv, config)
    eval_log_cb = EvalLogCallback(eval_log_csv, eval_metrics)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=config.get("eval_episodes", 10),
        deterministic=True,
        verbose=1,
        callback_after_eval=eval_log_cb,
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
        return DQN.load(best_path), best_path + ".zip"
    return model, model_path + ".zip"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def run_episodes(model, env, num_episodes):
    """Run episodes and return per-episode results."""
    results = []
    for ep in range(num_episodes):
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


def evaluate(model, env_config, run_dir, num_episodes, model_path=None, eval_csv=None):
    if eval_csv is None:
        eval_csv = os.path.join(run_dir, "evaluation.csv")

    print()
    print(f"=== Evaluating Trained Agent ({num_episodes} episodes) ===")
    if model_path:
        print(f"Model: {model_path}")
    print()

    env = make_env(env_config)
    results = run_episodes(model, env, num_episodes)
    env.close()

    with open(eval_csv, "w", newline="") as f:
        if model_path:
            f.write(f"# model: {model_path}\n")
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
    print(f"  Arrived: {n_arrived}/{num_episodes} ({100*n_arrived/num_episodes:.0f}%)  Crashed: {n_crashed}/{num_episodes} ({100*n_crashed/num_episodes:.0f}%)  Avg reward: {avg_reward:.2f}")
    print(f"  Saved to {eval_csv}")


def demo(model, env_config, run_dir, num_episodes):
    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")

    print()
    print(f"=== Recording Demo Videos ({num_episodes} episodes) ===")
    print()

    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    env = make_env(env_config)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix="social_attn_dqn",
    )
    env.unwrapped.set_record_video_wrapper(env)

    run_episodes(model, env, num_episodes)
    env.close()

    print()
    print("Generated videos:")
    for f in sorted(os.listdir(video_folder)):
        size = os.path.getsize(os.path.join(video_folder, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Social Attention DQN for intersection env"
    )
    parser.add_argument(
        "--demo-only", action="store_true",
        help="Skip training, demo existing model"
    )
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help="Skip training, evaluate existing model and save timestamped CSV"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config with overrides"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a .zip model file (for --demo-only or --evaluation-only)"
    )
    args = parser.parse_args()

    if args.demo_only:
        if not args.model_path:
            parser.error("--demo-only requires --model-path")
        if not args.config:
            parser.error("--demo-only requires --config")

        model_file = args.model_path
        if not model_file.endswith(".zip"):
            model_file += ".zip"
        if not os.path.exists(model_file):
            print(f"Error: Model not found: {model_file}")
            return

        # Derive run_dir by walking up to the parent of "models/"
        p = os.path.dirname(os.path.abspath(model_file))
        run_dir = p  # fallback
        while p != os.path.dirname(p):
            if os.path.basename(p) == "models":
                run_dir = os.path.dirname(p)
                break
            p = os.path.dirname(p)

        config = load_config(args.config)
        if config.get("idm"):
            apply_idm_params(config["idm"])

        load_path = model_file[:-4]  # strip .zip for SB3
        print(f"Loading model from {model_file}")
        model = DQN.load(load_path)

        demo_episodes = config.get("demo_episodes", 3)
        demo(model, config["env"], run_dir, demo_episodes)
    elif args.evaluation_only:
        if not args.model_path:
            parser.error("--evaluation-only requires --model-path")
        if not args.config:
            parser.error("--evaluation-only requires --config")

        model_file = args.model_path
        if not model_file.endswith(".zip"):
            model_file += ".zip"
        if not os.path.exists(model_file):
            print(f"Error: Model not found: {model_file}")
            return

        # Derive run_dir by walking up to the parent of "models/"
        p = os.path.dirname(os.path.abspath(model_file))
        run_dir = p  # fallback
        while p != os.path.dirname(p):
            if os.path.basename(p) == "models":
                run_dir = os.path.dirname(p)
                break
            p = os.path.dirname(p)

        config = load_config(args.config)
        if config.get("idm"):
            apply_idm_params(config["idm"])

        load_path = model_file[:-4]  # strip .zip for SB3
        print(f"Loading model from {model_file}")
        model = DQN.load(load_path)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_csv = os.path.join(run_dir, f"evaluation_{timestamp}.csv")
        eval_episodes = config.get("eval_episodes", 20)
        evaluate(model, config["env"], run_dir, eval_episodes,
                 model_path=model_file, eval_csv=eval_csv)
    else:
        config = load_config(args.config)
        if config.get("idm"):
            apply_idm_params(config["idm"])
        run_dir = make_run_dir(config)
        model, best_model_path = train(config, run_dir)
        eval_episodes = config.get("eval_episodes", 20)
        evaluate(model, config["env"], run_dir, eval_episodes, model_path=best_model_path)
        if config.get("demo_on_train_end", True):
            demo_episodes = config.get("demo_episodes", 3)
            demo(model, config["env"], run_dir, demo_episodes)


if __name__ == "__main__":
    main()
