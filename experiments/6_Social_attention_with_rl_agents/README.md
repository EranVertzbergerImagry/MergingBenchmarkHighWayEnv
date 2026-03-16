# Experiment 6: Social Attention with rl-agents

## Overview

Reproduces the Social Attention paper results using the authors' own [rl-agents](https://github.com/eleurent/rl-agents) library. This provides the EgoAttentionNetwork, built-in attention visualization, and DQN training harness used to produce the original paper results.

**Results:** 12% crash, 80% arrival (single scenario: south entry, left turn to o1)

This is the best single-scenario result and the baseline for all subsequent experiments.

## Key Observation

When crashes occur, in all examined cases the cause is **IDM vehicles crashing into the ego**, not the ego crashing into others. This motivated experiments 7-8.

## Architecture

- rl-agents `EgoAttentionNetwork` with 2 attention heads
- 7 observation features: `[presence, x, y, vx, vy, cos_h, sin_h]`
- Absolute (world-frame) coordinates
- Fixed destination: o1 (left turn from south entry)

## Files

- `train_rl_agents.py` — Main script (train/eval/demo/visualize)
- `default_config.json` — Points to modular config files
- `configs/env.json` — Environment config
- `configs/agents/ego_attention_2h.json` — Agent config (2-head attention)
- `configs/agents/baseline.json` — Base agent config
- `setup.sh` — Install rl-agents

## Transition from Exp 5

This experiment switched from stable-baselines3 to rl-agents. Key improvements:
- Native EgoAttentionNetwork (no custom reimplementation)
- Built-in attention visualization with decision pane
- Episode-based training (10K episodes vs 100K timesteps)
- Modular config files (env.json + agent.json)

The `TrainingTracker` class was introduced here as a step callback for `Evaluation`, logging per-episode metrics to `episodes.csv`. Live plot generation during training was added in a later session.

## Usage

```bash
source venv/bin/activate
python experiments/6_Social_attention_with_rl_agents/train_rl_agents.py
python experiments/6_Social_attention_with_rl_agents/train_rl_agents.py --test-only --recover-from <checkpoint.tar>
python experiments/6_Social_attention_with_rl_agents/train_rl_agents.py --demo-only --recover-from <checkpoint.tar>
python experiments/6_Social_attention_with_rl_agents/train_rl_agents.py --visualize --recover-from <checkpoint.tar>
```
