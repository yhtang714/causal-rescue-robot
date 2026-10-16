"""The candidate-set transformer ranker with its effect heads."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class nn:
        class Module:
            def __init__(self):
                pass
        class Linear:
            pass
        class LayerNorm:
            pass
        class Dropout:
            pass
        class Parameter:
            pass
        class ModuleList:
            pass
        class Sequential:
            pass

    class torch:
        @staticmethod
        def tensor(*args, **kwargs):
            return None
        class Tensor:
            pass

    class F:
        pass

    print("[Warning] PyTorch not available")


DEFAULT_CAUSAL_WEIGHTS = {
    "A_occluder_share": 0.110217,
    "A_indirect_occluder": 0.08,
    "A_num_above": -0.056563,
    "A_exposure_proxy": -0.028955,
    "A_height_norm": -0.020267,
    "A_region_over_hammer": -0.015895,
    "A_dz": -0.007410,
    "A_d_xy": -0.006174,
    "A_num_below": 0.005240,
    "A_support_depth": 0.004714,
    "A_region_xy_near": 0.000248,
}


class CausalFeatureScaling(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Dict[str, float],
        learnable: bool = True,
        init_alpha: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_names = feature_names
        self.learnable = learnable

        causal_vector = []
        for fname in feature_names:
            w = causal_weights.get(fname, 0.0)
            causal_vector.append(w)
        causal_vector = torch.tensor(causal_vector, dtype=torch.float32)

        max_abs = causal_vector.abs().max()
        if max_abs > 1e-9:
            causal_vector = causal_vector / max_abs

        if learnable:
            self.causal_scale = nn.Parameter(causal_vector.clone())
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.register_buffer("causal_scale", causal_vector)
            self.register_buffer("alpha", torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = 1.0 + self.alpha * torch.tanh(self.causal_scale)

        scaling = scaling.view(1, 1, -1)

        return x * scaling

    def get_effective_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            scaling = 1.0 + self.alpha * torch.tanh(self.causal_scale)
            return {
                name: scaling[i].item()
                for i, name in enumerate(self.feature_names)
            }


class CausalAttentionBias(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Dict[str, float],
        bias_scale: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_names = feature_names
        self.bias_scale = bias_scale

        causal_vector = []
        for fname in feature_names:
            w = causal_weights.get(fname, 0.0)
            causal_vector.append(w)
        causal_vector = torch.tensor(causal_vector, dtype=torch.float32)

        if learnable:
            self.causal_weights = nn.Parameter(causal_vector.clone())
            self.scale = nn.Parameter(torch.tensor(bias_scale))
        else:
            self.register_buffer("causal_weights", causal_vector)
            self.register_buffer("scale", torch.tensor(bias_scale))

    def compute_bias(self, features: torch.Tensor) -> torch.Tensor:
        importance = torch.sum(features * self.causal_weights.view(1, 1, -1), dim=-1)

        importance = self.scale * importance

        return importance

    def forward(
        self,
        attention_scores: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        importance = self.compute_bias(features)

        importance = importance.unsqueeze(1).unsqueeze(2)

        biased_scores = attention_scores + importance

        return biased_scores


class CausalMultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_causal_heads: int,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Dict[str, float],
        dropout: float = 0.1,
    ):
        super().__init__()

        assert n_causal_heads <= n_heads
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_causal_heads = n_causal_heads
        self.n_normal_heads = n_heads - n_causal_heads
        self.head_dim = d_model // n_heads

        self.feature_names = feature_names

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        self.causal_bias = CausalAttentionBias(
            input_dim=input_dim,
            feature_names=feature_names,
            causal_weights=causal_weights,
            learnable=True,
        )

        self._cached_features = None

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if features is not None and self.n_causal_heads > 0:
            causal_importance = self.causal_bias.compute_bias(features)
            causal_bias = causal_importance.unsqueeze(1).unsqueeze(2)

            if self.n_causal_heads < self.n_heads:
                normal_scores = attn_scores[:, :self.n_normal_heads]
                causal_scores = attn_scores[:, self.n_normal_heads:] + causal_bias
                attn_scores = torch.cat([normal_scores, causal_scores], dim=1)
            else:
                attn_scores = attn_scores + causal_bias

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class CausalTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_causal_heads: int,
        d_ff: int,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Dict[str, float],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = CausalMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_causal_heads=n_causal_heads,
            input_dim=input_dim,
            feature_names=feature_names,
            causal_weights=causal_weights,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(x, features=features, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class CausalAttentionTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Optional[Dict[str, float]] = None,
        causal_mode: str = "all",
        d_model: int = 64,
        n_heads: int = 4,
        n_causal_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_candidates: int = 128,
        mediator_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_names = feature_names
        self.causal_mode = causal_mode

        if causal_weights is None:
            causal_weights = DEFAULT_CAUSAL_WEIGHTS
        self.causal_weights = causal_weights

        self.use_scaling = causal_mode in ["scaling", "all"]
        if self.use_scaling:
            self.causal_scaling = CausalFeatureScaling(
                input_dim=input_dim,
                feature_names=feature_names,
                causal_weights=causal_weights,
                learnable=True,
            )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_candidates, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

        self.use_causal_attn = causal_mode in ["bias", "heads", "all"]

        if self.use_causal_attn:
            self.layers = nn.ModuleList([
                CausalTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_causal_heads=n_causal_heads if causal_mode in ["heads", "all"] else 0,
                    d_ff=d_ff,
                    input_dim=input_dim,
                    feature_names=feature_names,
                    causal_weights=causal_weights,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.mediator_names = list(mediator_names or [])
        if self.mediator_names:
            self.mediator_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, len(self.mediator_names)),
            )

        self.config = {
            "input_dim": input_dim,
            "feature_names": feature_names,
            "causal_weights": causal_weights,
            "causal_mode": causal_mode,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_causal_heads": n_causal_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "max_candidates": max_candidates,
            "mediator_names": self.mediator_names,
        }

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_mediators: bool = False,
    ):
        original_features = features

        if self.use_scaling:
            features = self.causal_scaling(features)

        x = self.input_proj(features)
        x = x + self.pos_encoding[:, :x.size(1), :]

        if self.use_causal_attn:
            for layer in self.layers:
                x = layer(x, features=original_features, mask=mask)
        else:
            if mask is not None:
                key_padding_mask = (mask == 0)
            else:
                key_padding_mask = None
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        scores = self.score_head(x).squeeze(-1)
        med = self.mediator_head(x) if (return_mediators and self.mediator_names) else None

        if mask is not None:
            S = scores.size(1)
            M = mask.size(1)
            if S < M:
                pad = torch.full((scores.size(0), M - S), float("-inf"),
                                 device=scores.device, dtype=scores.dtype)
                scores = torch.cat([scores, pad], dim=1)
            elif S > M:
                scores = scores[:, :M]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if return_mediators:
            return scores, med
        return scores

    def get_causal_weights_info(self) -> Dict[str, Dict[str, float]]:
        info = {
            "original_weights": self.causal_weights,
        }

        if self.use_scaling:
            info["learned_scaling"] = self.causal_scaling.get_effective_weights()

        return info


def build_causal_attention_model(
    feature_cols: List[str],
    causal_weights: Optional[Dict[str, float]] = None,
    causal_mode: str = "all",
    d_model: int = 64,
    n_heads: int = 4,
    n_causal_heads: int = 2,
    n_layers: int = 2,
    d_ff: int = 128,
    dropout: float = 0.1,
    max_candidates: int = 128,
) -> CausalAttentionTransformer:
    if causal_weights is None:
        causal_weights = DEFAULT_CAUSAL_WEIGHTS

    model = CausalAttentionTransformer(
        input_dim=len(feature_cols),
        feature_names=feature_cols,
        causal_weights=causal_weights,
        causal_mode=causal_mode,
        d_model=d_model,
        n_heads=n_heads,
        n_causal_heads=n_causal_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_candidates=max_candidates,
    )

    return model


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        exit(1)

    print("=" * 60)
    print("Testing Causal Attention Transformer")
    print("=" * 60)

    batch_size = 4
    seq_len = 32
    input_dim = 10

    feature_names = [
        "A_occluder_share", "A_num_above", "A_exposure_proxy",
        "A_height_norm", "A_d_xy", "A_dz",
        "A_num_below", "A_support_depth", "A_region_over_hammer", "A_region_xy_near"
    ]

    features = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 25:] = 0

    for mode in ["scaling", "bias", "heads", "all"]:
        print(f"\n[Test] Mode: {mode}")

        model = CausalAttentionTransformer(
            input_dim=input_dim,
            feature_names=feature_names,
            causal_mode=mode,
            d_model=64,
            n_heads=4,
            n_causal_heads=2,
            n_layers=2,
        )

        scores = model(features, mask)
        print(f"  Output shape: {scores.shape}")
        print(f"  Score range: [{scores[mask.bool()].min():.3f}, {scores[mask.bool()].max():.3f}]")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
