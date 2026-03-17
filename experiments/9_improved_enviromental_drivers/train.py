"""
Social Attention DQN — Experiment 9: Unified Intersection Training

Supports both IDM and frozen-model traffic, with single or multi-agent control.
Set "enviromental_driver_model" to "IDM" for IDM traffic or a checkpoint path
for frozen model traffic. Set "controlled_vehicles" > 1 for multi-agent.

Usage:
    python train.py
    python train.py --initial-model <warm_start.tar>
    python train.py --episodes 5000
    python train.py --evaluation-only --recover-from <ckpt.tar>
    python train.py --demo-only --recover-from <ckpt.tar>
    python train.py --visualize --recover-from <ckpt.tar>
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
# Multi-agent record() patch (from experiment 8)
# ---------------------------------------------------------------------------
def _patch_record_for_multi_agent(agent):
    """Monkey-patch agent.record() to use per-agent rewards from info."""
    original_record = agent.record.__func__

    def patched_record(self, state, action, reward, next_state, done, info):
        if not self.training:
            return
        if isinstance(state, tuple) and isinstance(action, tuple):
            agent_rewards = info.get("agent_rewards")
            if agent_rewards is None:
                agent_rewards = [reward] * len(state)
            for i, (s, a, ns) in enumerate(
                zip(state, action, next_state)
            ):
                if s.sum() == 0:
                    continue
                self.memory.push(s, a, agent_rewards[i], ns, done, info)
            batch = self.sample_minibatch()
            if batch:
                loss, _, _ = self.compute_bellman_residual(batch)
                self.step_optimizer(loss)
                self.update_target_network()
        else:
            original_record(self, state, action, reward, next_state, done, info)

    import types
    agent.record = types.MethodType(patched_record, agent)


# ---------------------------------------------------------------------------
# Multi-agent attention display (from experiment 8)
# ---------------------------------------------------------------------------
AGENT_COLORS = [
    (50, 200, 50),    # green  — agent 0
    (50, 100, 255),   # blue   — agent 1
    (255, 165, 0),    # orange — agent 2
    (200, 50, 200),   # purple — agent 3
]

MIN_ATTENTION = 0.01


def _compute_attention_for_state(agent, single_state):
    import torch
    state_t = torch.tensor([single_state], dtype=torch.float).to(agent.device)
    attention = agent.value_net.get_attention_matrix(state_t)
    attention = attention.squeeze(0).squeeze(1).detach().cpu().numpy()
    _, _, mask = agent.value_net.split_input(state_t)
    mask = mask.squeeze()
    return attention, mask


def _match_state_to_vehicles(single_state, mask, obs_type, road_vehicles, ego_vehicle):
    from rl_agents.utils import remap

    ego_heading = ego_vehicle.heading
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    ego_pos = ego_vehicle.position

    x_idx = obs_type.features.index("x")
    y_idx = obs_type.features.index("y")

    v_map = {}
    for v_index in range(single_state.shape[0]):
        if mask[v_index]:
            continue
        if v_index == 0:
            v_map[v_index] = ego_vehicle
            continue

        ex = remap(single_state[v_index, x_idx], [-1, 1], obs_type.features_range["x"])
        ey = remap(single_state[v_index, y_idx], [-1, 1], obs_type.features_range["y"])

        wx = cos_h * ex - sin_h * ey
        wy = sin_h * ex + cos_h * ey

        world_pos = np.array([wx + ego_pos[0], wy + ego_pos[1]])
        v_map[v_index] = min(road_vehicles, key=lambda v: np.linalg.norm(v.position - world_pos))
    return v_map


def unified_attention_display(agent, agent_surface, sim_surface):
    """Attention display that delegates to standard rl-agents graphics for single-agent
    and custom multi-agent overlay for multi-agent."""
    from rl_agents.agents.common.graphics import AgentGraphics

    state = getattr(agent, '_multi_agent_state', None)
    if state is None:
        state = agent.previous_state

    # Single-agent: use standard AgentGraphics which draws Q-value pane + attention
    if not isinstance(state, tuple):
        AgentGraphics.display(agent, agent_surface, sim_surface)
        return

    # Multi-agent: custom overlay
    _multi_agent_attention_overlay(agent, agent_surface, sim_surface, state)


def _multi_agent_attention_overlay(agent, agent_surface, sim_surface, state):
    import pygame
    pygame.draw.rect(agent_surface, (0, 0, 0),
                     (0, 0, agent_surface.get_width(), agent_surface.get_height()), 0)
    if state is None:
        return

    env = agent.env.unwrapped
    obs_type = env.observation_type

    agent_states = state if isinstance(state, tuple) else (state,)

    for agent_idx in range(len(agent_states)):
        single_state = agent_states[agent_idx]

        if single_state.sum() == 0:
            continue

        # For multi-agent with staggered spawning
        if hasattr(env, '_agent_spawned'):
            if not (agent_idx < len(env._agent_spawned)
                    and env._agent_spawned[agent_idx]
                    and env._agent_vehicle_idx[agent_idx] is not None):
                continue
            vidx = env._agent_vehicle_idx[agent_idx]
            ego_vehicle = env.controlled_vehicles[vidx]
        else:
            if not env.controlled_vehicles:
                continue
            ego_vehicle = env.controlled_vehicles[min(agent_idx, len(env.controlled_vehicles) - 1)]

        try:
            attention, mask = _compute_attention_for_state(agent, single_state)
            v_map = _match_state_to_vehicles(
                single_state, mask, obs_type,
                env.road.vehicles, ego_vehicle,
            )
        except (ValueError, IndexError, RuntimeError):
            continue

        base_color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

        for head in range(attention.shape[0]):
            attn_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
            for v_index, vehicle in v_map.items():
                att_val = attention[head, v_index]
                if att_val < MIN_ATTENTION:
                    continue
                width = att_val * 5
                alpha = int(min(att_val * 400, 200))
                color = (*base_color, alpha)

                ego_pix = sim_surface.vec2pix(ego_vehicle.position)
                if vehicle is ego_vehicle:
                    pygame.draw.circle(
                        attn_surface, color, ego_pix,
                        max(sim_surface.pix(width / 2), 1),
                    )
                else:
                    pygame.draw.line(
                        attn_surface, color, ego_pix,
                        sim_surface.vec2pix(vehicle.position),
                        max(sim_surface.pix(width), 1),
                    )
            sim_surface.blit(attn_surface, (0, 0))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_agent_config(config_dict):
    """Resolve base_config paths if loading from a file-based agent config."""
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
def is_multi_agent(env_config):
    """Check if the env config is for a multi-agent environment."""
    return env_config.get("controlled_vehicles", 1) > 1


def train(config, run_dir, initial_model=None, visualize=False):
    env_config = config["env"]
    agent_config = config["agent"]
    train_episodes = config["train_episodes"]

    episodes_csv = os.path.join(run_dir, "episodes.csv")
    plot_path = os.path.join(run_dir, "training_progress.html")

    n_agents = env_config.get("controlled_vehicles", 1)
    traffic_model = config.get("enviromental_driver_model", "IDM")
    traffic_label = "IDM" if traffic_model == "IDM" else "Frozen model"

    print("=== Training ===")
    print(f"Env id:          {env_config['id']}")
    print(f"Traffic model:   {traffic_label}")
    print(f"Agents:          {n_agents}")
    print(f"Episodes:        {train_episodes}")
    print(f"Run dir:         {run_dir}")
    if traffic_model != "IDM":
        print(f"Frozen model:    {env_config.get('enviromental_driver_model_path', 'N/A')}")
    print(f"Obs features:    {env_config['observation']['features']}")
    if initial_model:
        print(f"Warm-start from: {initial_model}")
    print()

    env = make_env(env_config)
    agent = load_agent(load_agent_config(agent_config), env)

    if initial_model:
        print(f"Loading initial model from {initial_model}")
        agent.load(initial_model)

    if is_multi_agent(env_config):
        _patch_record_for_multi_agent(agent)

    tracker = TrainingTracker(
        episodes_csv,
        plot_path=plot_path,
        description=config.get("description", ""),
    )

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

    return agent, env_config


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def run_episodes(agent, env, num_episodes):
    """Run episodes with agent in eval mode and return per-episode results."""
    multi = isinstance(env.unwrapped.observation_space, gym.spaces.Tuple)
    results = []

    for ep in range(num_episodes):
        obs, info = env.reset()

        done = False
        total_reward = 0.0
        agent_reward_sums = None
        steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            per_agent = info.get("agent_rewards")
            if per_agent is not None:
                if agent_reward_sums is None:
                    agent_reward_sums = [0.0] * len(per_agent)
                for i, r in enumerate(per_agent):
                    agent_reward_sums[i] += r
            steps += 1

        if multi:
            agents_info = info.get("agents", [])
            for ai in agents_info:
                i = ai["agent_id"]
                r = agent_reward_sums[i] if agent_reward_sums else total_reward
                results.append({
                    "episode": ep + 1,
                    "agent_id": i,
                    "steps": steps,
                    "reward": r,
                    "crashed": ai.get("crashed", False),
                    "arrived": ai.get("arrived", False),
                    "destination": ai.get("destination", "?"),
                    "entry": ai.get("entry", "?"),
                    "spawned": ai.get("spawned", False),
                })
            n_arrived = sum(1 for a in agents_info if a.get("arrived"))
            n_crashed = sum(1 for a in agents_info if a.get("crashed"))
            n_spawned = sum(1 for a in agents_info if a.get("spawned"))
            print(
                f"  Episode {ep + 1}: "
                f"arrived={n_arrived}/{n_spawned} crashed={n_crashed}/{n_spawned} "
                f"| Steps: {steps} | Avg Reward: {total_reward:.2f}"
            )
        else:
            try:
                vehicle = env.unwrapped.vehicle
                route = vehicle.route
                destination = route[-1][1] if route else "?"
            except (AttributeError, IndexError):
                destination = "?"

            crashed = info.get("crashed", False)
            arrived = info.get("rewards", {}).get("arrived_reward", 0) > 0
            if not arrived:
                arrived = info.get("is_success", False)
            results.append({
                "episode": ep + 1,
                "agent_id": 0,
                "steps": steps,
                "reward": total_reward,
                "crashed": crashed,
                "arrived": arrived,
                "destination": destination,
                "entry": "?",
                "spawned": True,
            })
            status = "ARRIVED" if arrived else ("CRASHED" if crashed else "TIMEOUT")
            print(f"  Episode {ep + 1}: {status} | dest={destination} | Steps: {steps} | Reward: {total_reward:.2f}")

    return results


def evaluate_agent(agent, env_config, run_dir, num_episodes, eval_csv=None):
    """Post-training evaluation with per-agent and per-destination breakdown."""
    if eval_csv is None:
        eval_csv = os.path.join(run_dir, "evaluation.csv")

    print()
    print(f"=== Evaluating Trained Agent ({num_episodes} episodes) ===")
    print()

    env = make_env(env_config)
    agent.eval()
    results = run_episodes(agent, env, num_episodes)
    env.close()

    fieldnames = ["episode", "agent_id", "steps", "reward", "crashed", "arrived",
                  "destination", "entry", "spawned"]
    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute stats (only for spawned agents)
    spawned_results = [r for r in results if r.get("spawned", True)]
    n_total = len(spawned_results)
    n_crashed = sum(r["crashed"] for r in spawned_results)
    n_arrived = sum(r["arrived"] for r in spawned_results)
    n_stall = n_total - n_crashed - n_arrived

    ep_rewards = defaultdict(list)
    for r in spawned_results:
        ep_rewards[r["episode"]].append(r["reward"])
    avg_reward = np.mean([np.mean(v) for v in ep_rewards.values()]) if ep_rewards else 0.0

    # Per-destination breakdown
    dest_stats = defaultdict(lambda: {"total": 0, "arrived": 0, "crashed": 0, "stall": 0})
    for r in spawned_results:
        d = r["destination"]
        dest_stats[d]["total"] += 1
        if r["arrived"]:
            dest_stats[d]["arrived"] += 1
        elif r["crashed"]:
            dest_stats[d]["crashed"] += 1
        else:
            dest_stats[d]["stall"] += 1

    # Per-entry breakdown
    entry_stats = defaultdict(lambda: {"total": 0, "arrived": 0, "crashed": 0, "stall": 0})
    for r in spawned_results:
        e = r["entry"]
        entry_stats[e]["total"] += 1
        if r["arrived"]:
            entry_stats[e]["arrived"] += 1
        elif r["crashed"]:
            entry_stats[e]["crashed"] += 1
        else:
            entry_stats[e]["stall"] += 1

    with open(eval_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow([
            "SUMMARY", f"{num_episodes} episodes",
            f"{n_total} agents spawned",
            f"avg_reward={avg_reward:.2f}",
            f"arrived={100 * n_arrived / max(n_total, 1):.0f}%",
            f"crashed={100 * n_crashed / max(n_total, 1):.0f}%",
            f"stall={100 * n_stall / max(n_total, 1):.0f}%",
        ])
        writer.writerow([])
        writer.writerow(["destination", "agents", "arrived%", "crashed%", "stall%"])
        for dest in sorted(dest_stats.keys()):
            s = dest_stats[dest]
            n = s["total"]
            writer.writerow([
                dest, n,
                f"{100 * s['arrived'] / n:.0f}%",
                f"{100 * s['crashed'] / n:.0f}%",
                f"{100 * s['stall'] / n:.0f}%",
            ])
        writer.writerow([])
        writer.writerow(["entry", "agents", "arrived%", "crashed%", "stall%"])
        for entry in sorted(entry_stats.keys()):
            s = entry_stats[entry]
            n = s["total"]
            writer.writerow([
                entry, n,
                f"{100 * s['arrived'] / n:.0f}%",
                f"{100 * s['crashed'] / n:.0f}%",
                f"{100 * s['stall'] / n:.0f}%",
            ])

    print()
    print(
        f"  Agents spawned: {n_total} across {num_episodes} episodes\n"
        f"  Arrived: {n_arrived}/{n_total} "
        f"({100 * n_arrived / max(n_total, 1):.0f}%)  "
        f"Crashed: {n_crashed}/{n_total} "
        f"({100 * n_crashed / max(n_total, 1):.0f}%)  "
        f"Stall: {n_stall}/{n_total} "
        f"({100 * n_stall / max(n_total, 1):.0f}%)  "
        f"Avg reward: {avg_reward:.2f}"
    )
    print()
    print(f"  {'Dest':<8} {'Agents':>7} {'Arrived':>8} {'Crashed':>8} {'Stall':>8}")
    print(f"  {'-'*41}")
    for dest in sorted(dest_stats.keys()):
        s = dest_stats[dest]
        n = s["total"]
        print(f"  {dest:<8} {n:>7} {100*s['arrived']/n:>7.0f}% {100*s['crashed']/n:>7.0f}% {100*s['stall']/n:>7.0f}%")
    if len(entry_stats) > 1:
        print()
        print(f"  {'Entry':<8} {'Agents':>7} {'Arrived':>8} {'Crashed':>8} {'Stall':>8}")
        print(f"  {'-'*41}")
        for entry in sorted(entry_stats.keys()):
            s = entry_stats[entry]
            n = s["total"]
            print(f"  {entry:<8} {n:>7} {100*s['arrived']/n:>7.0f}% {100*s['crashed']/n:>7.0f}% {100*s['stall']/n:>7.0f}%")
    print()
    print(f"  Saved to {eval_csv}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo(agent, env_config, num_episodes):
    """Record demo videos with attention overlay using RecordVideo wrapper."""
    video_folder = os.path.join(SCRIPT_DIR, "data", "videos")
    multi = is_multi_agent(env_config)

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
        name_prefix="unified",
    )
    try:
        env.unwrapped.set_record_video_wrapper(env)
    except AttributeError:
        pass

    agent.env = env
    agent.eval()

    from highway_env.envs.common.graphics import EnvViewer
    EnvViewer.agent_display = lambda a_surf, s_surf: unified_attention_display(agent, a_surf, s_surf)

    # Fixed camera for multi-agent
    class _FixedObserver:
        position = np.array([0, 0])

    for ep in range(num_episodes):
        obs, info = env.reset()

        if multi:
            try:
                env.unwrapped.viewer.observer_vehicle = _FixedObserver()
            except AttributeError:
                pass

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            agent._multi_agent_state = obs
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        agents_info = info.get("agents", [])
        if agents_info:
            n_arrived = sum(1 for a in agents_info if a.get("arrived"))
            n_crashed = sum(1 for a in agents_info if a.get("crashed"))
            n_spawned = sum(1 for a in agents_info if a.get("spawned"))
            print(
                f"  Episode {ep + 1}: arrived={n_arrived}/{n_spawned} "
                f"crashed={n_crashed}/{n_spawned} | Steps: {steps} | Avg Reward: {total_reward:.2f}"
            )
        else:
            crashed = info.get("crashed", False)
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
    multi = is_multi_agent(env_config)

    print()
    print(f"=== Visualizing Agent ({num_episodes} episodes) ===")
    print("  Close the pygame window or press Ctrl+C to stop.")
    print()

    env = make_env_for_rendering(env_config, render_mode="human")

    agent.env = env
    agent.eval()

    class _FixedObserver:
        position = np.array([0, 0])

    for ep in range(num_episodes):
        obs, info = env.reset()
        if ep == 0:
            try:
                env.unwrapped.viewer.set_agent_display(
                    lambda a_surf, s_surf: unified_attention_display(agent, a_surf, s_surf)
                )
            except AttributeError:
                print("  Warning: viewer does not support agent display overlay")
        if multi:
            try:
                env.unwrapped.viewer.observer_vehicle = _FixedObserver()
            except AttributeError:
                pass

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            agent._multi_agent_state = obs
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        agents_info = info.get("agents", [])
        if agents_info:
            n_arrived = sum(1 for a in agents_info if a.get("arrived"))
            n_crashed = sum(1 for a in agents_info if a.get("crashed"))
            print(f"  Episode {ep + 1}: arrived={n_arrived} crashed={n_crashed} | Steps: {steps} | Avg Reward: {total_reward:.2f}")
        else:
            crashed = info.get("crashed", False)
            arrived = info.get("is_success", False)
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
        description="Social Attention DQN — Experiment 9: Unified Intersection"
    )
    parser.add_argument(
        "--initial-model", type=str, default=None,
        help="Path to .tar checkpoint for warm-starting the ego agent",
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

    # Resolve traffic model: "IDM" means standard IDM, otherwise it's a frozen model path
    driver_model = config.get("enviromental_driver_model", "IDM")
    if driver_model and driver_model != "IDM":
        # Resolve frozen model path
        repo_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
        frozen_model_path = os.path.join(repo_root, driver_model)
        if not os.path.exists(frozen_model_path):
            frozen_model_path = driver_model
        if not os.path.exists(frozen_model_path):
            print(f"Error: Frozen model checkpoint not found: {frozen_model_path}")
            return
        frozen_model_path = os.path.abspath(frozen_model_path)
        config["env"]["enviromental_driver_model_path"] = frozen_model_path
        config["env"]["frozen_agent_config"] = config["agent"]
    # else: IDM mode — no frozen model path needed, env uses parent IDM behavior

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
            config, run_dir,
            initial_model=args.initial_model,
            visualize=args.visualize,
        )

        eval_episodes = config.get("eval_episodes", 100)
        evaluate_agent(agent, env_cfg, run_dir, eval_episodes)

        if config.get("demo_on_train_end", True):
            demo_episodes = config.get("demo_episodes", 3)
            demo(agent, env_cfg, demo_episodes)


if __name__ == "__main__":
    main()
