# Experiment 3: DQN with Social Attention

## Goal

Implement the Social Attention DQN from the paper
*"Social Attention for Autonomous Decision-Making in Dense Traffic"*
(Leurent & Mercat, 2019, [arXiv:1911.12250](https://arxiv.org/abs/1911.12250))
on the `intersection-v1` environment.

## Background

The official highway-env Colab notebook
(`scripts/intersection_social_dqn.ipynb`) uses the `eleurent/rl-agents`
repository which contains a full implementation of:

- **`EgoAttention`** — multi-head attention where only the ego vehicle
  generates queries; all vehicles produce keys and values.
- **`SelfAttention`** — standard symmetric multi-head self-attention
  (optional preprocessing step).
- **`EgoAttentionNetwork`** — the complete model: separate ego/others
  embeddings -> optional self-attention -> ego-attention -> output MLP.

### Do we need to implement the algorithm ourselves?

**Partially.** The neural network architecture (EgoAttention, SelfAttention,
EgoAttentionNetwork) is fully implemented in `rl-agents` at
`rl_agents/agents/common/models.py`. However:

| Aspect | Status in `rl-agents` | Our approach |
|--------|----------------------|--------------|
| Attention model (EgoAttentionNetwork) | Fully implemented | **Extract and port** into our repo |
| DQN training loop | Custom loop in `Evaluation` class | **Use stable-baselines3 DQN** instead (consistent with experiment 2) |
| Replay buffer, target net, exploration | Custom implementation | **Use SB3 built-ins** |
| Training plots | TensorBoard only | **Reuse our `TrainingPlotCallback`** |
| Attention visualization | Implemented in `DQNGraphics` | **Port the visualization** |
| JSON config system | `Configurable` base class | **Not needed** — use Python constants |

**Decision: Extract the model architecture from `rl-agents` and integrate it
as a custom SB3 feature extractor.** This avoids pulling in the full
unmaintained `rl-agents` dependency (last commit Jan 2024, CI only tests
Python 3.8) while keeping our training pipeline consistent across experiments.

## Architecture

```
Observation: (batch, 15, 7)
  |
  +--> split into ego (batch, 1, 7) and others (batch, 14, 7)
  |                                    + presence mask
  |
  +--> ego_embedding:    MLP(7 -> 64 -> 64)
  +--> others_embedding: MLP(7 -> 64 -> 64)
  |
  +--> [Optional] SelfAttention(feature_size=64, heads=2)
  |
  +--> EgoAttention(feature_size=64, heads=2)
  |      - Q from ego only, K/V from all entities
  |      - Masked softmax (absent vehicles get -inf)
  |      - Residual: (attended + ego) / 2
  |
  +--> output_layer: MLP(64 -> 64 -> 64 -> n_actions)
```

The Colab notebook uses `ego_attention_2h.json`: 2-head ego-attention,
no self-attention, [64,64] embedding layers.

## Key requirement: observation must NOT be flattened

The attention model expects a 2D input `(vehicles_count, features)`.
Experiment 2 uses `"flatten": True` for the MLP policy — here we must
set `"flatten": False` (the default) and handle the 2D observation
in a custom SB3 feature extractor.

## Implementation Plan

### Step 1: Port the attention model

Create `social_attention_model.py` with classes extracted from
`rl-agents/rl_agents/agents/common/models.py`:

- `EgoAttention(nn.Module)` — ego-only query attention mechanism
- `SelfAttention(nn.Module)` — optional symmetric self-attention
- `EgoAttentionNetwork(nn.Module)` — full pipeline

Minimal changes from original:
- Remove the `Configurable` / `BaseModule` base classes (not needed)
- Remove `model_factory` / JSON config wiring
- Keep the forward pass and `get_attention_matrix()` identical

### Step 2: Create custom SB3 feature extractor

Create a `SocialAttentionExtractor(BaseFeaturesExtractor)` that:
- Takes the 2D gym observation `(vehicles_count, features)`
- Runs it through `EgoAttentionNetwork`
- Outputs a flat feature vector for SB3's Q-network head

Use it via:
```python
policy_kwargs = dict(
    features_extractor_class=SocialAttentionExtractor,
    features_extractor_kwargs=dict(...),
    net_arch=[],  # no extra MLP layers — output_layer is inside the extractor
)
```

### Step 3: Create training script

Create `train_social_attention_dqn.py` mirroring experiment 2's structure:
- Same env config but with `"flatten": False` and `"order": "shuffled"`
- DQN hyperparameters matching the paper's config:
  - `gamma=0.95`, `batch_size=64`, `buffer_size=15000`
  - `target_update_interval=512`
  - `exploration_fraction` tuned to match `tau=15000` over training steps
  - `exploration_final_eps=0.05`
  - Double DQN: `True` (SB3 default is already Double DQN... **verify**)
- Reuse `TrainingPlotCallback` from experiment 2
- `--demo-only` flag for running a saved model
- Save model + plot + videos to `data/`

### Step 4: Add attention visualization

Port attention visualization from
`rl-agents/rl_agents/agents/deep_q_network/graphics.py`:
- Extract attention weights via `model.get_attention_matrix(obs)`
- Draw lines from ego to other vehicles with width proportional to
  attention weight
- Save visualization frames or overlay on recorded video

### Step 5: Compare with experiment 2

After training, compare:
- Episode reward curves (experiment 2 vs 3)
- Arrival rate
- Crash rate
- Qualitative behavior via demo videos

## Files to create

```
experiments/3_DQN_W_Social_Attention/
  PLAN.md                          <-- this file
  social_attention_model.py        <-- ported attention architecture
  train_social_attention_dqn.py    <-- training + demo script
  data/
    models/                        <-- saved checkpoints
    videos/                        <-- demo recordings
    training_progress.png          <-- live training plot
```

## References

- Paper: https://arxiv.org/abs/1911.12250
- Original model code: `rl-agents/rl_agents/agents/common/models.py`
- Original configs: `rl-agents/scripts/configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json`
- Colab notebook: `HighwayEnv/scripts/intersection_social_dqn.ipynb`
- rl-agents repo: https://github.com/eleurent/rl-agents
