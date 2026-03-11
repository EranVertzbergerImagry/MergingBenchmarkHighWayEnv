"""Step callback that logs per-episode metrics to episodes.csv during rl-agents training.

Optionally updates a training_progress.html plot every ``plot_interval`` episodes.
"""
import csv

from src.plotting import generate_training_plot


class TrainingTracker:
    """Passed to ``Evaluation(step_callback_fn=tracker.step_callback)``.

    Accumulates per-step rewards and writes a row on episode end.
    """

    def __init__(self, episodes_csv, plot_path=None, description="",
                 plot_interval=50):
        self.episodes_csv = episodes_csv
        self.plot_path = plot_path
        self.description = description
        self.plot_interval = plot_interval
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_destination = "?"
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
            # Capture destination from vehicle route
            try:
                vehicle = env.unwrapped.vehicle
                route = vehicle.route
                self._ep_destination = route[-1][1] if route else "?"
            except (AttributeError, IndexError):
                self._ep_destination = "?"

        self._ep_reward += reward
        self._ep_steps += 1

        if terminated or truncated:
            crashed = info.get("crashed", False)
            arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
            if not arrived:
                arrived = info.get("is_success", False)
            self._write_row(
                episode + 1, self._ep_reward, self._ep_steps,
                crashed, arrived, self._ep_destination,
            )
            self._maybe_update_plot(episode + 1)

    def _write_row(self, episode, reward, length, crashed, arrived, destination):
        with open(self.episodes_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["episode", "reward", "length", "crashed", "arrived", "destination"])
                self._header_written = True
            writer.writerow([episode, f"{reward:.4f}", length, crashed, arrived, destination])

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
