# Project: ExperimentingWithHighwayEnv

## Overview

This project explores reinforcement learning for autonomous driving at a 4-way intersection using the [highway-env](https://github.com/Farama-Foundation/HighwayEnv) simulator. The core question: **what level of driving competence can RL achieve at an unsignalized intersection?**

The project progresses through 8 experiments, from random baselines to multi-agent social attention policies. The best single-agent result (experiment 7) achieves **7% crash, 63% arrival** with ego-centric observations and random destinations. Current work (experiment 8) focuses on multi-agent training to improve robustness.

## Key Libraries

- **highway-env**: Gymnasium-based driving simulator with intersection environment, IDM traffic, and kinematic observations
- **rl-agents**: [Leurent's RL framework](https://github.com/eleurent/rl-agents) used in experiments 6-8. Provides DQN with EgoAttentionNetwork, built-in attention visualization, and training harness
- **stable-baselines3**: Used in experiments 2-5 (deprecated in favor of rl-agents)
- **Python 3.10**, virtualenv in `venv/`

## Branch

All work is on branch `modified_env_and_RL_algos_implementation`.

## Experiment Progression

| Exp | Description | Framework | Crash | Arrive | Key Innovation |
|-----|-------------|-----------|-------|--------|----------------|
| 1 | Random policy baseline | - | 55% | 45% | - |
| 2-4 | Vanilla DQN/PPO/SAC | SB3 | ~50% | ~50% | Grid search on rewards |
| 5 | DQN + Social Attention | SB3 | 21% | 72% | Transformer attention over vehicles |
| 6 | rl-agents reproduction | rl-agents | 12% | 80% | Paper's own library, 2-head attention |
| 7 | Ego-centric + random dest | rl-agents | 7% | 63% | Heading-invariant obs, all destinations |
| 8 | Multi-agent intersection | rl-agents | WIP | WIP | N controlled vehicles, shared policy |

## Architecture: EgoAttentionNetwork (Experiments 5-8)

Based on "Social Attention for Autonomous Decision-Making in Dense Traffic" (Leurent & Mercat, 2019).

```
Observation (15 vehicles x 9 features)
    |
    v
[Ego Embedding MLP 64-64] + [Others Embedding MLP 64-64]
    |                              |
    v                              v
        EgoAttention (2 heads)
        - Query: ego only
        - Key/Value: all vehicles
        - Softmax attention with presence mask
    |
    v
[Output MLP 64-64] -> Q-values (3 actions: SLOWER/IDLE/FASTER)
```

The 9 observation features (experiments 7-8): `[presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]`

Ego-centric transform (experiments 7-8): positions and velocities are rotated into the ego vehicle's heading frame, making the policy heading-invariant.

## Code Structure

### Experiments 1-5 (standalone scripts)
Each experiment has a single `train_*.py` script with inline model definitions, callbacks, and plotting.

### Experiments 6-8 (rl-agents framework)
```
experiments/N_name/
    train.py              # Main script: train/eval/demo/visualize
    default_config.json   # Unified config (env + agent + training params)
    setup.sh              # Install dependencies
    src/                  # Shared modules (experiments 7-8)
        __init__.py       # Imports + gym.register() for custom envs
        random_spawn_intersection.py   # Custom env: random entry/destination
        ego_centric_wrapper.py         # Observation wrapper: ego-frame rotation
        training_tracker.py            # Step callback: CSV logging + live plot
        plotting.py                    # Plotly HTML chart generation
        multi_agent_intersection.py    # (exp 8 only) Multi-agent env
    data/
        runs/             # Timestamped training runs with checkpoints
        videos/           # Demo recordings
```

### Key Config Parameters

```json
{
  "env": {
    "id": "intersection-multi-agent-v0",
    "controlled_vehicles": 4,
    "policy_frequency": 5,          // decisions per second (default: 1)
    "duration": 20,                  // episode length in seconds
    "initial_vehicle_count": 0,      // IDM vehicles at start
    "spawn_probability": 0.3,        // IDM spawn rate per step
    "spawn_window_fraction": 0.1,    // fraction of duration for agent spawning
    "observation": {
      "vehicles_count": 15,          // max vehicles in observation
      "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "cos_d", "sin_d"],
      "order": "shuffled"            // prevents positional memorization
    },
    "action": {
      "type": "DiscreteMetaAction",
      "target_speeds": [0, 4.5, 9]   // CRITICAL: must match IntersectionEnv defaults
    }
  },
  "agent": {
    "exploration": {
      "method": "EpsilonGreedy",
      "tau": 15000,                   // decay time constant (in steps, not episodes)
      "temperature": 1.0,             // initial epsilon
      "final_temperature": 0.05       // asymptotic epsilon
    },
    "optimizer": {
      "type": "ADAM",
      "lr": 5e-4                      // default learning rate
    }
  }
}
```

## Critical Technical Details

### target_speeds Bug
The `action.target_speeds` config MUST be `[0, 4.5, 9]` to match IntersectionEnv defaults. If omitted when providing explicit action config, MDPVehicle falls back to `[20, 25, 30]` m/s, causing vehicles to accelerate uncontrollably. This caused a major debugging session in experiment 8.

### Exploration Decay in Multi-Agent
The epsilon decay formula is: `ε(t) = ε_final + (ε_init - ε_final) * exp(-t / τ)`

Where `t` increments per `act()` call. In multi-agent mode, `act()` recurses N times (once per agent), so the exploration clock ticks N times faster. To maintain the same episode-based schedule, multiply tau by N.

Episodes to reach target epsilon:
```
E = (-τ / (L * N)) * ln((ε_target - ε_final) / (ε_init - ε_final))
```
Where L = duration * policy_frequency (steps per episode), N = number of agents.

### Multi-Agent Reward (Experiment 8)
- `step()` returns a scalar reward (average of spawned agents) for gymnasium compatibility
- Per-agent rewards are in `info["agent_rewards"]` as a tuple
- `agent.record()` is monkey-patched to read per-agent rewards from info and skip unspawned agents (zero observations)
- Without this patch, all agents get the same averaged reward in the replay buffer, breaking credit assignment

### Multi-Agent Attention Visualization
- `agent.previous_state` is clobbered by DQN's recursive `act()` — only holds last agent's state
- Fix: save full tuple as `agent._multi_agent_state` before calling `act()`
- Attention display reads `_multi_agent_state` and draws per-agent colored lines (green/blue/orange/purple)

### Training Progress Plot
- `training_progress.html` is generated every 50 episodes during training with auto-refresh (30s meta tag)
- Final plot after training removes auto-refresh
- Multi-agent plotting aggregates per-episode (average reward, max length)

## Running Experiments

```bash
# Setup
source venv/bin/activate

# Train (experiment 8 example)
python experiments/8_multi_agent/train.py

# Train with warm-start from experiment 7 checkpoint
python experiments/8_multi_agent/train.py --initial-model experiments/7_Social_attention_generalization/data/runs/2026-03-10_11-18-09_ego_attention_2h_random_dest_ego_centric_reference/checkpoint-final.tar

# Evaluate only
python experiments/8_multi_agent/train.py --evaluation-only --recover-from <checkpoint.tar>

# Record demo videos
python experiments/8_multi_agent/train.py --demo-only --recover-from <checkpoint.tar>

# Visualize with attention overlay (pygame window)
python experiments/8_multi_agent/train.py --visualize --recover-from <checkpoint.tar>
```

## Current Status (Experiment 8)

### Problem
The exp7 model occasionally crashes because IDM vehicles collide with the ego. The attention visualization shows the ego sometimes fails to attend to nearby vehicles.

### Hypothesis
Training with multiple controlled (ego) vehicles instead of IDM will produce a more robust policy, since ego vehicles learn to avoid each other rather than relying on IDM behavior.

### Findings So Far
- **8 agents + 10 IDM**: Stall behavior (reward -> 0). Too congested; agent learns "stop = don't crash"
- **1 agent (validation)**: Reproduces exp7 results, confirming no bugs
- **Policy frequency 5Hz**: Significantly improves zero-shot transfer (77% arrive vs 54% at 1Hz)
- **Fine-tuning with low epsilon**: Model degraded without learning rate reduction
- **Fine-tuning with lr=5e-5**: Currently being tested

### Next Steps
- Curriculum: 2 agents -> 4 -> 8, warm-starting from previous checkpoint
- Reduce IDM density when adding more controlled agents
- Adjust tau proportionally to N agents
