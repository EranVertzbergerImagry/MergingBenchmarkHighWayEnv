"""Step callback that logs per-episode metrics to episodes.csv during rl-agents training.

Supports both single-agent and multi-agent environments. In multi-agent mode,
writes one row per agent per episode with agent_id and per-agent status.
Optionally updates a training_progress.html plot every ``plot_interval`` episodes.
"""
import csv

from src.plotting import generate_training_plot


class TrainingTracker:
    """Passed to ``Evaluation(step_callback_fn=tracker.step_callback)``.

    Accumulates per-step rewards and writes rows on episode end.
    """

    def __init__(self, episodes_csv, plot_path=None, description="",
                 plot_interval=50):
        self.episodes_csv = episodes_csv
        self.plot_path = plot_path
        self.description = description
        self.plot_interval = plot_interval
        self._ep_rewards = None  # scalar or list (per-agent)
        self._ep_steps = 0
        self._current_ep = -1
        self._header_written = False
        self._is_multi_agent = None
        self._last_plot_ep = 0

    def step_callback(self, episode, env, agent, transition, writer):
        obs, reward, terminated, truncated, info = transition

        # Detect multi-agent on first call (per-agent rewards in info)
        if self._is_multi_agent is None:
            self._is_multi_agent = "agent_rewards" in info

        if episode != self._current_ep:
            self._current_ep = episode
            if self._is_multi_agent:
                n = len(info.get("agent_rewards", []))
                self._ep_rewards = [0.0] * n
            else:
                self._ep_rewards = 0.0
            self._ep_steps = 0

        if self._is_multi_agent:
            agent_rewards = info.get("agent_rewards", [])
            for i, r in enumerate(agent_rewards):
                self._ep_rewards[i] += r
        else:
            self._ep_rewards += reward
        self._ep_steps += 1

        if terminated or truncated:
            if self._is_multi_agent:
                self._write_multi_agent_rows(episode + 1, info)
            else:
                self._write_single_agent_row(episode + 1, env, info)
            self._maybe_update_plot(episode + 1)

    def _write_single_agent_row(self, episode, env, info):
        try:
            vehicle = env.unwrapped.vehicle
            route = vehicle.route
            destination = route[-1][1] if route else "?"
        except (AttributeError, IndexError):
            destination = "?"

        crashed = info.get("crashed", False)
        arrived = info.get("is_success", False)

        with open(self.episodes_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["episode", "agent_id", "reward", "length",
                                 "crashed", "arrived", "destination", "entry"])
                self._header_written = True
            writer.writerow([
                episode, 0, f"{self._ep_rewards:.4f}", self._ep_steps,
                crashed, arrived, destination, "?",
            ])

    def _write_multi_agent_rows(self, episode, info):
        agents_info = info.get("agents", [])

        with open(self.episodes_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["episode", "agent_id", "reward", "length",
                                 "crashed", "arrived", "spawned", "destination", "entry"])
                self._header_written = True
            for ai in agents_info:
                i = ai["agent_id"]
                agent_reward = self._ep_rewards[i] if i < len(self._ep_rewards) else 0.0
                writer.writerow([
                    episode,
                    i,
                    f"{agent_reward:.4f}",
                    self._ep_steps,
                    ai.get("crashed", False),
                    ai.get("arrived", False),
                    ai.get("spawned", False),
                    ai.get("destination", "?"),
                    ai.get("entry", "?"),
                ])

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
