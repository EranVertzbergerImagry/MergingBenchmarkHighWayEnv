"""
Social Attention model for autonomous driving.

Ported from eleurent/rl-agents (rl_agents/agents/common/models.py).
Paper: "Social Attention for Autonomous Decision-Making in Dense Traffic"
       Leurent & Mercat, 2019  (arXiv:1911.12250)

Contains:
    - EgoAttention:             ego-only query, multi-head attention
    - SelfAttention:            symmetric multi-head self-attention
    - EgoAttentionNetwork:      full pipeline (embed -> [self-attn] -> ego-attn -> output)
    - SocialAttentionExtractor: SB3-compatible BaseFeaturesExtractor wrapper
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---- Scaled dot-product attention ----------------------------------------

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention.

    :param query:   (batch, head, 1, features_per_head)       [ego only]
                 or (batch, head, entities, features_per_head) [self-attn]
    :param key:     (batch, head, entities, features_per_head)
    :param value:   (batch, head, entities, features_per_head)
    :param mask:    (batch, head, 1, entities) — True where absent
    :param dropout: optional nn.Dropout
    :return: (output, attention_weights)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


# ---- Building blocks -----------------------------------------------------

class MLP(nn.Module):
    """Simple configurable MLP (no reshape by default)."""

    def __init__(self, in_size, layers, out_size=None, activation=F.relu):
        super().__init__()
        self.activation = activation
        sizes = [in_size] + list(layers)
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.predict = nn.Linear(sizes[-1], out_size) if out_size else None

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.predict is not None:
            x = self.predict(x)
        return x


class EgoAttention(nn.Module):
    """
    Multi-head attention where only the ego entity generates queries,
    while all entities (ego + others) produce keys and values.
    """

    def __init__(self, feature_size=64, heads=2, dropout=0.0):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = feature_size // heads

        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.query_ego = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ego, others, mask=None):
        """
        :param ego:    (batch, 1, feature_size)
        :param others: (batch, n_others, feature_size)
        :param mask:   (batch, n_entities, 1) — True where absent
        :return: (result, attention_matrix)
        """
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat(
            (ego.view(batch_size, 1, self.feature_size), others), dim=1
        )

        # (batch, entities, heads, features_per_head)
        key_all = self.key_all(input_all).view(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        value_all = self.value_all(input_all).view(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        query_ego = self.query_ego(ego).view(
            batch_size, 1, self.heads, self.features_per_head
        )

        # -> (batch, heads, entities, features_per_head)
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.view(batch_size, 1, 1, n_entities).repeat(1, self.heads, 1, 1)

        value, attention_matrix = attention(
            query_ego, key_all, value_all, mask, self.dropout
        )

        result = (
            self.attention_combine(
                value.reshape(batch_size, self.feature_size)
            )
            + ego.squeeze(1)
        ) / 2
        return result, attention_matrix


class SelfAttention(nn.Module):
    """
    Standard symmetric multi-head self-attention.
    All entities generate queries, keys, and values.
    """

    def __init__(self, feature_size=64, heads=2, dropout=0.0):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = feature_size // heads

        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.query_all = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat(
            (ego.view(batch_size, 1, self.feature_size), others), dim=1
        )

        key_all = self.key_all(input_all).view(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        value_all = self.value_all(input_all).view(
            batch_size, n_entities, self.heads, self.features_per_head
        )
        query_all = self.query_all(input_all).view(
            batch_size, n_entities, self.heads, self.features_per_head
        )

        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.view(batch_size, 1, 1, n_entities).repeat(1, self.heads, 1, 1)

        value, attention_matrix = attention(
            query_all, key_all, value_all, mask, self.dropout
        )

        result = (
            self.attention_combine(
                value.reshape(batch_size, n_entities, self.feature_size)
            )
            + input_all
        ) / 2
        return result, attention_matrix


# ---- Full network ---------------------------------------------------------

class EgoAttentionNetwork(nn.Module):
    """
    Full Social Attention pipeline:
        1. Separate ego / others embeddings (MLPs)
        2. Optional SelfAttention between all entities
        3. EgoAttention (ego attends to all)
        4. Output MLP -> Q-values
    """

    def __init__(
        self,
        in_features=7,
        out_features=3,
        embedding_layers=(64, 64),
        others_embedding_layers=(64, 64),
        attention_feature_size=64,
        attention_heads=2,
        self_attention=False,
        self_attention_heads=2,
        output_layers=(64, 64),
        presence_feature_idx=0,
    ):
        super().__init__()
        self.presence_feature_idx = presence_feature_idx

        self.ego_embedding = MLP(in_features, embedding_layers)
        self.others_embedding = MLP(in_features, others_embedding_layers)

        self.self_attention_layer = None
        if self_attention:
            self.self_attention_layer = SelfAttention(
                attention_feature_size, self_attention_heads
            )

        self.attention_layer = EgoAttention(
            attention_feature_size, attention_heads
        )

        self.output_layer = MLP(
            attention_feature_size, output_layers, out_size=out_features
        )

    def split_input(self, x, mask=None):
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = (
                x[:, :, self.presence_feature_idx : self.presence_feature_idx + 1]
                < 0.5
            )
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego = self.ego_embedding(ego)
        others = self.others_embedding(others)
        if self.self_attention_layer is not None:
            self_att, _ = self.self_attention_layer(ego, others, mask)
            ego, others, mask = self.split_input(self_att, mask=mask)
        return self.attention_layer(ego, others, mask)

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        return self.output_layer(ego_embedded_att)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


# ---- SB3 feature extractor wrapper --------------------------------------

class SocialAttentionExtractor(BaseFeaturesExtractor):
    """
    Wraps EgoAttentionNetwork as an SB3 features extractor.

    The observation space is Box(vehicles_count, n_features).
    This extractor runs the full EgoAttentionNetwork (embed + attention +
    output MLP) and returns a flat feature vector that SB3's Q-head can use.

    Use with policy_kwargs:
        policy_kwargs = dict(
            features_extractor_class=SocialAttentionExtractor,
            features_extractor_kwargs=dict(
                embedding_layers=[64, 64],
                attention_heads=2,
                output_layers=[64, 64],
            ),
            net_arch=[],  # output MLP is already inside the extractor
        )
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        embedding_layers=(64, 64),
        others_embedding_layers=None,
        attention_feature_size=64,
        attention_heads=2,
        self_attention=False,
        self_attention_heads=2,
        output_layers=(64, 64),
        presence_feature_idx=0,
    ):
        # The extractor outputs a vector of size output_layers[-1]
        features_dim = output_layers[-1]
        super().__init__(observation_space, features_dim)

        if others_embedding_layers is None:
            others_embedding_layers = embedding_layers

        n_features = observation_space.shape[-1]

        self.net = EgoAttentionNetwork(
            in_features=n_features,
            out_features=features_dim,  # extractor output, not action count
            embedding_layers=embedding_layers,
            others_embedding_layers=others_embedding_layers,
            attention_feature_size=attention_feature_size,
            attention_heads=attention_heads,
            self_attention=self_attention,
            self_attention_heads=self_attention_heads,
            output_layers=output_layers,
            presence_feature_idx=presence_feature_idx,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, vehicles_count, n_features)
        return self.net(observations)
