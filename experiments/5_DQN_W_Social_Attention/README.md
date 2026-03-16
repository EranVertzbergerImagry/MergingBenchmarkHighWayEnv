# Experiment 5: DQN with Social Attention (stable-baselines3)

## Overview

First implementation of the Social Attention architecture from "Social Attention for Autonomous Decision-Making in Dense Traffic" (Leurent & Mercat, 2019), using stable-baselines3's DQN.

**Results:** 21% crash, 72% arrival (single scenario: south entry, left turn to o1)

## Architecture

Custom `SocialAttentionExtractor` in `social_attention_model.py`:
- Embedding layers: MLP 64-64 for ego and others
- EgoAttention: 2 heads, feature_size=64
- Output MLP: 64-64 -> Q-values

## Key Differences from Experiments 6+

| Aspect | Exp 5 | Exp 6+ |
|--------|-------|--------|
| Framework | stable-baselines3 | rl-agents |
| Training unit | Timesteps (100K) | Episodes (10K) |
| Observation | 7 features, absolute | 7 features (exp6), 9 features ego-centric (exp7+) |
| Policy frequency | 5 Hz | 1 Hz (default) |
| Target speeds | `[0, 2, 5, 10, 15, 20]` | `[0, 4.5, 9]` |
| Plotting | Live callback (every 4 episodes) | Post-training (fixed in later sessions) |

## Files

- `train_social_attention_dqn.py` — Training script with `TrainingPlotCallback` for live HTML plots
- `social_attention_model.py` — `SocialAttentionExtractor` (SB3 feature extractor)
- `default_config.json` — Config (100K timesteps, 5Hz policy, fixed destination o1)
- `run_reward_grid.sh` — Grid search over reward weights

## Usage

```bash
python experiments/5_DQN_W_Social_Attention/train_social_attention_dqn.py
python experiments/5_DQN_W_Social_Attention/train_social_attention_dqn.py --demo-only --model-path <path>
```
