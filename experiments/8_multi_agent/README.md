# Experiment 8 — Multi-Agent Intersection

## Overview

Building on Experiment 7's ego-centric generalization, this experiment introduces **multi-agent training and evaluation** at an intersection. Multiple controlled vehicles navigate the intersection simultaneously, each using the same shared DQN policy independently. The goal is to produce a more robust policy by training against other learning agents rather than scripted IDM traffic.

## Motivation

The exp7 model (7% crash, 63% arrival) occasionally crashes because IDM vehicles collide with the ego. Attention visualization reveals the ego sometimes fails to attend to nearby vehicles or attends too late. Hypothesis: training with multiple ego agents forces the policy to handle more unpredictable interactions, since ego agents don't follow IDM rules.

## Code Structure

```
8_multi_agent/
    train.py                  # Main script: train/eval/demo/visualize
    default_config.json       # Unified config (env + agent + training)
    setup.sh
    src/
        __init__.py           # Imports + gym.register("intersection-multi-agent-v0")
        multi_agent_intersection.py   # MultiAgentIntersectionEnv
        random_spawn_intersection.py  # Base class (from exp7)
        ego_centric_wrapper.py        # Ego-frame observation wrapper
        training_tracker.py           # Per-agent CSV logging + live plots
        plotting.py                   # Plotly charts (aggregates multi-agent)
    data/
        runs/                 # Training runs with checkpoints
        videos/               # Demo recordings
```

## Key Design

### Staggered Random Spawning
- Each agent gets a random spawn time in `[0, duration * spawn_window_fraction)`
- Random entry points (0-3) and random destinations
- Collision-safe: checks lane clearance before spawning (`spawn_clearance` config)
- Unspawned agents get zero observations and are excluded from training

### Per-Agent Rewards
- `step()` returns scalar reward (average of spawned agents) for gymnasium compatibility
- Per-agent rewards stored in `info["agent_rewards"]` as a tuple
- `agent.record()` is monkey-patched (`_patch_record_for_multi_agent`) to:
  - Read per-agent rewards from `info["agent_rewards"]`
  - Skip unspawned agents (zero observations) — they don't enter the replay buffer
- Without this patch, all agents get the same averaged reward, breaking credit assignment

### Ego-Centric Observations (9 features)
Each agent's observation is independently rotated into its own heading frame via `EgoCentricWrapper`. The shared policy is heading-invariant.

### Attention Visualization
- Per-agent colored attention overlay (green/blue/orange/purple)
- `agent._multi_agent_state` saves the tuple before `act()` clobbers `agent.previous_state`
- Lines from ego to attended vehicles; width/opacity proportional to attention weight
- Circle around ego = self-attention weight
- Camera fixed at intersection center (`_FixedObserver`)

## Exploration Decay Warning

The epsilon formula `ε(t) = ε_final + (ε_init - ε_final) * exp(-t / τ)` ticks per `act()` call. In multi-agent, `act()` recurses N times per step, so exploration decays N times faster in wall-clock episodes. **Compensate by setting `tau = tau_single * N`.**

## Experiment Log

### exp0: 8 agents + 10 IDM (stall)
- Config: 8 controlled, 10 initial IDM, 0.6 spawn prob, tau=15000
- Result: Reward -> 0, 0.5% arrival. Stall behavior (agent stops to avoid crash)
- Cause: Too congested + exploration decayed 8x too fast
- Run: `data/runs/2026-03-10_21-28-15_multi_agent_8_controlled_ego_centric`

### Policy frequency experiments
- 1Hz model tested at 1Hz on 4 agents: 54% arrive, 18% crash
- 1Hz model tested at 5Hz on 4 agents: 77% arrive, 7% crash
- 5Hz trained model at 5Hz on 4 agents: 79% arrive, 9% crash
- **Conclusion**: 5Hz inference helps significantly; 5Hz training shows marginal improvement

### Fine-tuning experiments
- Fine-tune exp7 model with temp=0.05 (no exploration): model degraded
- Fine-tune with lr=5e-5 (reduced from 5e-4): in progress

### Current config (`default_config.json`)
- 4 controlled vehicles, 0 IDM, 0.3 spawn probability
- 5Hz policy frequency, duration=20s
- tau=1000000, temp=0.05, final_temp=0.05 (minimal exploration for fine-tuning)
- lr=5e-5 (reduced for fine-tuning stability)

## Usage

```bash
# Train from scratch
python experiments/8_multi_agent/train.py

# Train with warm-start from exp7 checkpoint
python experiments/8_multi_agent/train.py --initial-model <exp7_checkpoint.tar>

# Evaluate
python experiments/8_multi_agent/train.py --evaluation-only --recover-from <checkpoint.tar>

# Record demo videos with attention overlay
python experiments/8_multi_agent/train.py --demo-only --recover-from <checkpoint.tar>

# Live visualization (pygame window)
python experiments/8_multi_agent/train.py --visualize --recover-from <checkpoint.tar>

# Override config
python experiments/8_multi_agent/train.py --config override.json --episodes 5000
```

## Configurable Parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `controlled_vehicles` | env | Number of ego agents (1-8) |
| `initial_vehicle_count` | env | IDM vehicles at start |
| `spawn_probability` | env | IDM spawn rate per step |
| `spawn_window_fraction` | env | Fraction of duration for agent spawn window |
| `spawn_clearance` | env | Min distance for safe spawning |
| `policy_frequency` | env | Decisions per second (default 1) |
| `duration` | env | Episode length in seconds |
| `tau` | agent.exploration | Epsilon decay constant (steps). Scale by N agents |
| `temperature` | agent.exploration | Initial epsilon |
| `final_temperature` | agent.exploration | Asymptotic epsilon |
| `lr` | agent.optimizer | Learning rate (default 5e-4) |
| `--initial-model` | CLI | Warm-start from checkpoint |
