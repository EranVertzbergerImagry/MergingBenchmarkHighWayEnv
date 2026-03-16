# Experiment 9: Improved Environmental Drivers (Frozen Opponents)

## Motivation

Experiment 7 achieves 7% crash / 63% arrival, but some crashes are caused by IDM vehicles behaving unrealistically (e.g., sudden stops, ignoring intersection geometry). This creates:

1. **Evaluation noise**: Can never reach 0% crash because IDM collisions are unavoidable
2. **Training noise**: IDM-caused crashes penalize the ego for potentially correct decisions

Experiment 8 attempted to fix this by training all N agents simultaneously, but this led to stall behavior (reward → 0) due to congestion.

## Approach

Replace IDM traffic entirely with **frozen model-driven vehicles** using the trained exp7 checkpoint. Train a single ego agent against these frozen opponents.

### Architecture

```
Training Agent (single-agent DQN, same as exp7)
        |
        v
  EgoCentricWrapper        <-- transforms ego's obs only (single-agent)
        |
        v
  FrozenOpponentIntersectionEnv   <-- 1 ego + dynamic frozen MDPVehicles
        |                              spawns/removes frozen vehicles like IDM would
        |                              computes frozen obs + actions internally
        v
  Road simulation (highway-env)
```

The environment is **single-agent** from the outside — the training pipeline is identical to exp7. Frozen vehicles are managed internally: each step, the env builds ego-centric observations for each frozen vehicle, runs them through the frozen network, and applies greedy actions.

### Key Design Decisions

- **No IDM at all**: All traffic vehicles are MDPVehicles driven by the frozen model
- **Same lifecycle as IDM**: Frozen vehicles spawn/despawn with `initial_vehicle_count` and `spawn_probability`, just like IDM
- **Same observation transform**: Frozen vehicles use the same ego-centric rotation as the ego (via standalone `ego_centric_transform()` function)
- **Greedy actions**: Frozen vehicles always take argmax Q-value (no exploration)

## Usage

```bash
# Setup
source venv/bin/activate

# Train against frozen exp7 opponents (path configured in default_config.json)
python experiments/9_improved_enviromental_drivers/train.py

# Train with warm-start from exp7
python experiments/9_improved_enviromental_drivers/train.py \
    --initial-model <exp7_checkpoint.tar>

# Evaluate only
python experiments/9_improved_enviromental_drivers/train.py \
    --evaluation-only --recover-from <exp9_checkpoint.tar>

# Record demo videos
python experiments/9_improved_enviromental_drivers/train.py \
    --demo-only --recover-from <exp9_checkpoint.tar>

# Visualize with attention overlay
python experiments/9_improved_enviromental_drivers/train.py \
    --visualize --recover-from <exp9_checkpoint.tar>
```

## Config

Key differences from exp7 config:
- `enviromental_driver_model`: path to frozen model checkpoint (relative to repo root)
- `env.id`: `"intersection-frozen-opponents-v0"`
- `env.duration`: 20s (longer to allow frozen vehicles to navigate)
- `env.spawn_entry`: `null` (random spawn for ego)
- `env.policy_frequency`: 5 Hz (from exp8 findings)

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script with `--frozen-model` CLI arg |
| `default_config.json` | Environment + agent config |
| `src/frozen_opponent_env.py` | Core environment: manages frozen MDPVehicles |
| `src/ego_centric_transform.py` | Standalone ego-centric rotation function |
| `src/ego_centric_wrapper.py` | Gymnasium wrapper for ego observation (from exp7) |
| `src/random_spawn_intersection.py` | Base env class (from exp7) |
| `src/training_tracker.py` | CSV logging + plot updates (from exp7) |
| `src/plotting.py` | Plotly HTML chart generation (from exp7) |
