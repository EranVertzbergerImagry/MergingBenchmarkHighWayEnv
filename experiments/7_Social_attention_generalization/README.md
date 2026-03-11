# Experiment 7 — Social Attention Generalization

Generalization of Experiment 6. All observations are transformed into the **ego vehicle's reference frame** and the agent learns to reach **random destinations** from **random initial positions** at an intersection.

Experiment 6 reproduced the Social Attention paper results (crash 12%, arrive 80%) but trained on a single scenario only: south entry, left turn. The learned policy could not generalize to other entry points or turn directions. This experiment addresses that limitation.

## What Changed from Experiment 6

### 1. Code Architecture — Unified Config & Modular Source

**Unified configuration** — Experiment 6 used three separate config files (`env.json`, `ego_attention.json` with `base_config` chain to `baseline.json`, and `default_config.json` referencing the other two by path). These are now merged into a single `default_config.json` containing both environment and agent settings inline. The full config is saved to each run directory for reproducibility.

**Modular source layout** — The monolithic `train_rl_agents.py` was split into a slim `train.py` entry point that imports from a `src/` package:

```
7_Social_attention_generalization/
├── train.py                  # CLI entry point
├── default_config.json       # unified env + agent config
├── setup.sh                  # rl-agents install + patches
├── src/
│   ├── __init__.py           # package imports, triggers gym.register
│   ├── random_spawn_intersection.py   # custom env subclass
│   ├── ego_centric_wrapper.py         # observation wrapper
│   ├── training_tracker.py            # training episode logger
│   └── plotting.py                    # plotly training charts
└── data/                     # generated run directories
```

### 2. Environment — Random Destinations & Configurable Spawn

**Custom environment** (`RandomSpawnIntersectionEnv`, registered as `intersection-random-spawn-v0`) — subclasses `IntersectionEnv` with two new config keys:

- `spawn_entry`: integer 0–3 for a fixed entry (0 = south), or `null` for random. Training uses `0` (south only); inference uses `null` via the `--random-spawn` CLI flag.
- `destination`: set to `null` so each episode picks a random exit from the remaining three exits (excluding the ego's own entry).

This means the agent trains on all three turn types (right, straight, left) from a single entry point.

### 3. Observations — Ego-Centric Frame & Destination Features

**Two new features** — `cos_d` and `sin_d` (destination direction) added to the observation vector, expanding it from 7 to 9 features:

```
[presence, x, y, vx, vy, cos_h, sin_h, cos_d, sin_d]
```

These tell the agent which direction its destination lies, so it can learn different speed profiles for different turn types.

**Ego-centric wrapper** (`EgoCentricWrapper`) — applied on top of the environment, rotates all observations into the ego vehicle's heading frame:

- **Position** (x, y): translated to ego origin, then rotated by −ego_heading
- **Velocity** (vx, vy): rotated only (no subtraction — ego needs its own speed)
- **Heading** (cos_h, sin_h): rotated
- **Destination** (cos_d, sin_d): injected for ego only (other vehicles' intentions stay zero), then rotated

After transformation, the ego row always reads: `x=0, y=0, cos_h=1, sin_h=0`, with `vx` = forward speed and `cos_d/sin_d` pointing toward the destination in the ego frame. This makes the policy **heading-invariant** — a left turn looks the same regardless of which intersection arm the vehicle enters from.

### 4. Network — Embedding Input Size

The EgoAttention network's embedding layers (`embedding_layer` and `others_embedding_layer`) were updated from `in: 7` to `in: 9` to match the expanded observation features.

### 5. Logging — Per-Destination Breakdown

Both training (`episodes.csv`) and evaluation (`evaluation.csv`) now log the **destination** for each episode. The evaluation summary includes a per-destination breakdown:

```
  Dest   Episodes  Arrived  Crashed    Stall
  --------------------------------------
  o1           29      41%      48%      10%
  o2           38      53%      45%       3%
  o3           33      88%       6%       6%
```

This makes it easy to see whether the agent performs equally across turn types (right turns via o3 are inherently easier than left turns via o1).

## Usage

```bash
# Train (10K episodes, south entry, random destinations)
python train.py

# Train with fewer episodes
python train.py --episodes 500

# Evaluate a checkpoint
python train.py --evaluation-only --recover-from data/runs/<run>/checkpoint-final.tar

# Evaluate with random spawn points (all 4 entries)
python train.py --evaluation-only --recover-from <ckpt> --random-spawn

# Record demo videos
python train.py --demo-only --recover-from <ckpt>

# Demo with random spawn
python train.py --demo-only --recover-from <ckpt> --random-spawn
```
