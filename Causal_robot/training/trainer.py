"""Training loop of the ranker: listwise loss, effect heads, checkpoint selection."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Warning] PyTorch not available. Install with: pip install torch")

    class Dataset:
        pass

    class DataLoader:
        pass

    class nn:
        Module = object

    class _DummyTorch:
        def no_grad(self):
            def decorator(func):
                return func
            return decorator

        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

    torch = _DummyTorch()

    class F:
        @staticmethod
        def log_softmax(*args, **kwargs):
            pass
        @staticmethod
        def nll_loss(*args, **kwargs):
            pass
        @staticmethod
        def binary_cross_entropy(*args, **kwargs):
            pass
        @staticmethod
        def logsumexp(*args, **kwargs):
            pass

CAUSAL_ATTENTION_AVAILABLE = False
try:
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _project_root = _os.path.dirname(_here)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    _decision_dir = _os.path.join(_project_root, "decision")
    if _decision_dir not in _sys.path:
        _sys.path.insert(0, _decision_dir)
    try:
        from decision.causal_attention_transformer import (
            CausalAttentionTransformer,
            DEFAULT_CAUSAL_WEIGHTS as CAUSAL_ATT_WEIGHTS,
        )
        CAUSAL_ATTENTION_AVAILABLE = True
        print("[Info] 因果注意力模块已加载 (decision.causal_attention_transformer)")
    except ImportError:
        from causal_attention_transformer import (
            CausalAttentionTransformer,
            DEFAULT_CAUSAL_WEIGHTS as CAUSAL_ATT_WEIGHTS,
        )
        CAUSAL_ATTENTION_AVAILABLE = True
        print("[Info] 因果注意力模块已加载 (causal_attention_transformer)")
except ImportError as _e:
    CAUSAL_ATTENTION_AVAILABLE = False
    print(f"[Info] 因果注意力模块未找到，使用基础模型 ({_e})")

try:
    from utils_policy import (
        validate_candidate_data,
        generate_optimal_labels,
        compute_feature_stats,
        validate_features,
        truncate_candidates,
        diagnose_training_issues,
        FeatureScaler,
        DataValidationResult,
    )
except ImportError:
    warnings.warn("utils_policy.py not found. Some features may not work.")


DEFAULT_GROUP_COLS = ["episode_id", "scene_id", "step_index"]

DEFAULT_EXCLUDE_COLS = {
    "episode_id", "scene_id", "step_index", "A_cube_id", "cube_id", "candidate_id",
    "is_chosen", "is_optimal", "is_optimal_any", "delta_E", "delta_C", "rel_delta_C",
    "C_before", "C_after", "E_before", "E_after", "exposure_before", "exposure_after",
    "exposure_status",
    "causal_score_delta_E", "top_causal_factor_delta_E", "top_causal_contrib_delta_E",
    "policy_seed",
    "cf_delta_E", "cf_delta_burden", "cf_burden_before",
    "cf_occ_share_removed", "cf_unlock", "cf_value", "cf_value2", "exposure_before", "n_feasible",
    "cf_unlock_E", "cf_done", "cf_togo", "cf_value_togo", "cf_delta_layers", "cf_unlock_L", "burial_before", "cf_value3", "stuck_state",
    "n_actions", "action_cid", "action_dir", "action_type", "cf_delta_clear",
}

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


def _weights_from_summary(csv_path, feature_cols, outcome="delta_E"):
    import os
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"causal weights csv not found: {csv_path}")
    s = pd.read_csv(csv_path)
    if "outcome" in s.columns:
        s = s[s["outcome"].astype(str) == outcome]
    prefer = ["ATE_ipw", "ATE_reg", "ATE_naive_diff", "CATE_mean"]
    w = {}
    for _, r in s.iterrows():
        t = str(r.get("treatment", ""))
        if t not in feature_cols:
            continue
        for c in prefer:
            if c in s.columns and pd.notna(r.get(c)):
                try:
                    v = float(r[c])
                except Exception:
                    continue
                if np.isfinite(v):
                    w[t] = v
                    break
    return w


def _weights_from_data(df, feature_cols, target_col="delta_E", l2=1.0):
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].values.astype(float)
    if target_col not in df.columns:
        return {}
    y = df[target_col].values.astype(float)
    Xc = np.hstack([X, np.ones((len(X), 1))])
    A = Xc.T @ Xc + l2 * np.eye(Xc.shape[1])
    try:
        coef = np.linalg.solve(A, Xc.T @ y)
    except Exception:
        coef = np.linalg.lstsq(Xc, y, rcond=None)[0]
    return {f: float(coef[i]) for i, f in enumerate(cols)}


def resolve_causal_weights(df_train, feature_cols, causal_weights_csv,
                           use_default_weights, target_col):
    if causal_weights_csv:
        try:
            w = _weights_from_summary(causal_weights_csv, feature_cols)
        except Exception as e:
            print(f"    ⚠️ 读取 {causal_weights_csv} 失败: {e}，回退到数据估计")
            w = {}
        if w:
            print(f"    [因果权重] 来源 = 因果发现 {causal_weights_csv} (去混杂 ATE)，匹配 {len(w)} 个特征")
            return w, f"impact_summary:{os.path.basename(causal_weights_csv)}"
        print("    ⚠️ impact_summary 未匹配到任何特征，回退到训练数据 ridge 估计")
    if use_default_weights:
        print("    ⚠️⚠️ 显式使用写死的 DEFAULT_CAUSAL_WEIGHTS（含预估值），不建议用于论文结果！")
        return dict(DEFAULT_CAUSAL_WEIGHTS), "DEFAULT_hardcoded"
    w = _weights_from_data(df_train, feature_cols, target_col)
    print(f"    [因果权重] 来源 = 训练数据 ridge 回归 {target_col}（数据驱动，非写死），{len(w)} 个特征")
    return w, "data_ridge"


class CausalFeatureScaling(nn.Module):

    def __init__(self, input_dim: int, feature_names: List[str], causal_weights: Dict[str, float]):
        super().__init__()

        causal_vector = []
        for fname in feature_names:
            w = causal_weights.get(fname, 0.0)
            causal_vector.append(w)
        causal_vector = torch.tensor(causal_vector, dtype=torch.float32)

        max_abs = causal_vector.abs().max()
        if max_abs > 1e-9:
            causal_vector = causal_vector / max_abs

        self.causal_scale = nn.Parameter(causal_vector.clone())
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, 'enabled', True):
            return x
        scaling = 1.0 + self.alpha * torch.tanh(self.causal_scale)
        return x * scaling.view(1, 1, -1)


class OptimalPolicyTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feature_names: List[str],
        causal_weights: Dict[str, float],
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_candidates: int = 128,
        use_pos_encoding: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_names = feature_names
        self.use_pos_encoding = use_pos_encoding

        self.causal_scaling = CausalFeatureScaling(input_dim, feature_names, causal_weights)
        self.causal_scaling.enabled = bool(globals().get('_CAUSAL_ENABLED', True))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.zeros(1, max_candidates, d_model))
            nn.init.normal_(self.pos_encoding, std=0.02)
        else:
            self.pos_encoding = None

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

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.causal_scaling(features)

        x = self.input_proj(x)

        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding[:, :x.size(1), :]

        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        scores = self.score_head(x).squeeze(-1)

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

        return scores


class OptimalPolicyDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        feature_cols: List[str],
        target_col: str = "is_optimal",
        max_candidates: int = 128,
        top_k: int = 48,
        random_k: int = 16,
        is_train: bool = True,
        tie_break: str = "first",
        mediator_cols: Optional[List[str]] = None,
    ):
        self.feature_cols = feature_cols
        self.mediator_cols = list(mediator_cols or [])
        self.target_col = target_col
        self.max_candidates = max_candidates
        self.top_k = top_k
        self.random_k = random_k
        self.is_train = is_train
        self.tie_break = tie_break

        self.groups = []

        n_skipped_single = 0
        n_skipped_no_pos = 0

        for _, group_df in df.groupby(group_cols, sort=False):
            if len(group_df) < 2:
                n_skipped_single += 1
                continue

            features = group_df[feature_cols].values.astype(np.float32)
            targets = group_df[target_col].values.astype(np.float32)

            if targets.sum() == 0:
                n_skipped_no_pos += 1
                continue

            if "is_optimal_any" in group_df.columns:
                targets_any = group_df["is_optimal_any"].values.astype(np.float32)
            else:
                targets_any = targets.copy()

            if "delta_E" in group_df.columns:
                delta_e = group_df["delta_E"].values.astype(np.float32)
            else:
                delta_e = targets.copy()

            if self.mediator_cols:
                med = group_df[self.mediator_cols].values.astype(np.float32)
            else:
                med = np.zeros((len(group_df), 0), dtype=np.float32)
            self.groups.append({
                "features": features,
                "targets": targets,
                "targets_any": targets_any,
                "delta_e": delta_e,
                "mediators": med,
                "size": len(group_df),
            })

        print(f"[Dataset] {'Train' if is_train else 'Val'}: {len(self.groups)} groups")
        if n_skipped_single > 0:
            print(f"         Skipped {n_skipped_single} single-candidate groups")
        if n_skipped_no_pos > 0:
            print(f"         Skipped {n_skipped_no_pos} groups without positive samples")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        features = group["features"].copy()
        targets = group["targets"].copy()
        targets_any = group["targets_any"].copy()
        delta_e = group["delta_e"].copy()
        med = group["mediators"].copy()
        size = group["size"]

        if self.is_train:
            perm = np.random.permutation(size)
            features = features[perm]
            targets = targets[perm]
            targets_any = targets_any[perm]
            delta_e = delta_e[perm]
            med = med[perm]

        if size > self.max_candidates:
            sel = self._truncate_indices(targets, delta_e)
            features, targets, targets_any, delta_e, med = (features[sel], targets[sel],
                                                            targets_any[sel], delta_e[sel], med[sel])
            size = len(sel)

        optimal_idx = np.argmax(targets)

        pad_size = self.max_candidates - size
        if pad_size > 0:
            features = np.pad(features, ((0, pad_size), (0, 0)), mode="constant")
            targets = np.pad(targets, (0, pad_size), mode="constant")
            targets_any = np.pad(targets_any, (0, pad_size), mode="constant")
            delta_e = np.pad(delta_e, (0, pad_size), mode="constant")
            if med.shape[1]:
                med = np.pad(med, ((0, pad_size), (0, 0)), mode="constant")
            else:
                med = np.zeros((self.max_candidates, 0), dtype=np.float32)

        mask = np.concatenate([np.ones(size), np.zeros(pad_size)]).astype(np.float32)

        return {
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(targets),
            "targets_any": torch.from_numpy(targets_any),
            "delta_e": torch.from_numpy(delta_e.astype(np.float32)),
            "mediators": torch.from_numpy(np.ascontiguousarray(med, dtype=np.float32)),
            "mask": torch.from_numpy(mask),
            "optimal_idx": optimal_idx,
            "size": size,
        }

    def _truncate_indices(self, targets, delta_e):

        sorted_indices = np.argsort(-delta_e)

        actual_top_k = min(self.top_k, self.max_candidates)
        actual_random_k = self.max_candidates - actual_top_k

        top_indices = sorted_indices[:actual_top_k]
        remaining_indices = sorted_indices[actual_top_k:]

        if actual_random_k > 0 and len(remaining_indices) > 0:
            random_sample = np.random.choice(
                remaining_indices,
                size=min(actual_random_k, len(remaining_indices)),
                replace=False
            )
            selected_indices = np.concatenate([top_indices, random_sample])
        else:
            selected_indices = top_indices

        optimal_idx = np.argmax(targets)
        if optimal_idx not in selected_indices:
            replace_pos = np.random.randint(0, len(selected_indices))
            selected_indices[replace_pos] = optimal_idx

        np.random.shuffle(selected_indices)

        return selected_indices


def listwise_ce_loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size = scores.shape[0]
    device = scores.device

    targets_masked = targets.clone()
    targets_masked[mask == 0] = -float("inf")
    optimal_idx = targets_masked.argmax(dim=1)

    scores_masked = scores.clone()
    scores_masked[mask == 0] = -float("inf")
    log_probs = F.log_softmax(scores_masked, dim=1)

    loss = F.nll_loss(log_probs, optimal_idx, reduction="mean")

    return loss


def multi_positive_ce_loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size = scores.shape[0]
    device = scores.device

    total_loss = 0.0
    n_valid = 0

    for i in range(batch_size):
        valid = mask[i].bool()
        if valid.sum() < 2:
            continue

        s = scores[i][valid]
        t = targets[i][valid]

        pos_mask = t > 0.5
        if pos_mask.sum() == 0:
            continue

        log_probs = F.log_softmax(s, dim=0)

        loss = -torch.logsumexp(log_probs[pos_mask], dim=0)

        total_loss += loss
        n_valid += 1

    if n_valid > 0:
        return total_loss / n_valid
    return torch.tensor(0.0, device=device)


def bce_loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    probs = torch.sigmoid(scores[valid])
    loss = F.binary_cross_entropy(probs, targets[valid])
    return loss


def listwise_reg_loss(scores: torch.Tensor, delta_e: torch.Tensor,
                      mask: torch.Tensor, huber_delta: float = 0.1) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=scores.device)
    err = scores[valid] - delta_e[valid]
    abse = err.abs()
    quad = torch.clamp(abse, max=huber_delta)
    lin = abse - quad
    return (0.5 * quad * quad + huber_delta * lin).mean()


def softrank_loss(scores: torch.Tensor, delta_e: torch.Tensor,
                  mask: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    bs = scores.shape[0]
    total = scores.new_tensor(0.0)
    n = 0
    for i in range(bs):
        v = mask[i].bool()
        if int(v.sum()) < 2:
            continue
        s = scores[i][v]
        d = delta_e[i][v]
        target = F.softmax(d / temp, dim=0)
        logp = F.log_softmax(s, dim=0)
        total = total + F.kl_div(logp, target, reduction="sum")
        n += 1
    return total / max(n, 1)


def compute_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    targets_any: torch.Tensor,
    mask: torch.Tensor,
    tie_break: str = "first",
    delta_e: torch.Tensor = None,
) -> Dict[str, float]:
    batch_size = scores.shape[0]
    metrics = {
        "p@1": [],
        "p@1_any": [],
        "p@3": [],
        "p@5": [],
        "mrr": [],
        "avg_rank": [],
        "spearman": [],
        "regret": [],
    }

    for i in range(batch_size):
        valid = mask[i].bool()
        n_valid = valid.sum().item()
        if n_valid < 2:
            continue

        s = scores[i][valid]
        t = targets[i][valid]
        t_any = targets_any[i][valid]

        if t.sum() == 0:
            continue

        sorted_indices = s.argsort(descending=True)
        pred_top1 = sorted_indices[0].item()

        target_optimal = t.argmax().item()
        metrics["p@1"].append(1.0 if pred_top1 == target_optimal else 0.0)

        if t_any[pred_top1] > 0.5:
            metrics["p@1_any"].append(1.0)
        else:
            metrics["p@1_any"].append(0.0)

        top3 = sorted_indices[:min(3, n_valid)]
        hit3 = any(t[idx] > 0.5 for idx in top3)
        metrics["p@3"].append(1.0 if hit3 else 0.0)

        top5 = sorted_indices[:min(5, n_valid)]
        hit5 = any(t[idx] > 0.5 for idx in top5)
        metrics["p@5"].append(1.0 if hit5 else 0.0)

        pos_indices = (t > 0.5).nonzero(as_tuple=True)[0]
        for pi in pos_indices:
            rank = (sorted_indices == pi).nonzero(as_tuple=True)[0].item() + 1
            metrics["mrr"].append(1.0 / rank)
            metrics["avg_rank"].append(rank)

        if delta_e is not None:
            d = delta_e[i][valid]
            sr = s.argsort().argsort().float()
            dr = d.argsort().argsort().float()
            ss, ds = sr.std(unbiased=False), dr.std(unbiased=False)
            if ss > 0 and ds > 0:
                metrics["spearman"].append(
                    (((sr - sr.mean()) * (dr - dr.mean())).mean() / (ss * ds)).item())
            metrics["regret"].append((d.max() - d[pred_top1]).item())

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [self.get_lr()]


def mediator_loss(pred, true, mask):
    if pred is None or true is None or true.numel() == 0 or true.size(-1) == 0:
        return None
    m = mask.unsqueeze(-1)
    denom = m.sum() * true.size(-1)
    if float(denom) <= 0:
        return None
    return (((pred - true) ** 2) * m).sum() / denom


def train_epoch(model, dataloader, optimizer, device, loss_type="ce", tie_break="first", reg_temp=0.1, reg_huber=0.1,
                mediator_weight=0.0):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        targets_any = batch["targets_any"].to(device)
        delta_e = batch["delta_e"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        med_pred = None
        if mediator_weight > 0 and getattr(model, "mediator_names", None):
            scores, med_pred = model(features, mask, return_mediators=True)
        else:
            scores = model(features, mask)

        if loss_type == "reg":
            loss = listwise_reg_loss(scores, delta_e, mask, reg_huber)
        elif loss_type == "softrank":
            loss = softrank_loss(scores, delta_e, mask, reg_temp)
        elif loss_type == "bce":
            loss = bce_loss(scores, targets, mask)
        elif loss_type == "ce_any" or (loss_type == "ce" and tie_break == "any"):
            loss = multi_positive_ce_loss(scores, targets_any, mask)
        else:
            loss = listwise_ce_loss(scores, targets, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if med_pred is not None:
            ml = mediator_loss(med_pred, batch["mediators"].to(device), mask)
            if ml is not None and not (torch.isnan(ml) or torch.isinf(ml)):
                loss = loss + float(mediator_weight) * ml

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, loss_type="ce", tie_break="first", reg_temp=0.1, reg_huber=0.1):
    model.eval()
    total_loss = 0
    n_batches = 0

    all_scores, all_targets, all_targets_any, all_masks, all_delta = [], [], [], [], []

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        targets_any = batch["targets_any"].to(device)
        delta_e = batch["delta_e"].to(device)
        mask = batch["mask"].to(device)

        scores = model(features, mask)

        if loss_type == "reg":
            loss = listwise_reg_loss(scores, delta_e, mask, reg_huber)
        elif loss_type == "softrank":
            loss = softrank_loss(scores, delta_e, mask, reg_temp)
        elif loss_type == "bce":
            loss = bce_loss(scores, targets, mask)
        elif loss_type == "ce_any" or (loss_type == "ce" and tie_break == "any"):
            loss = multi_positive_ce_loss(scores, targets_any, mask)
        else:
            loss = listwise_ce_loss(scores, targets, mask)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            n_batches += 1

        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())
        all_targets_any.append(targets_any.cpu())
        all_masks.append(mask.cpu())
        all_delta.append(delta_e.cpu())

    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_targets_any = torch.cat(all_targets_any, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_delta = torch.cat(all_delta, dim=0)

    metrics = compute_metrics(all_scores, all_targets, all_targets_any, all_masks, tie_break, all_delta)
    metrics["loss"] = total_loss / max(n_batches, 1)

    return metrics


def train(
    csv_path: str,
    save_dir: str,
    target_col: str = "delta_E",
    group_cols: Optional[List[str]] = None,
    loss_type: str = "ce",
    reg_temp: float = 0.1,
    reg_huber: float = 0.1,
    tie_break: str = "first",
    use_pos_encoding: bool = False,
    causal_mode: str = "scaling",
    n_causal_heads: int = 2,
    causal_weights_csv: Optional[str] = None,
    use_default_weights: bool = False,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 5e-4,
    patience: int = 30,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    max_candidates: int = 128,
    device: str = "auto",
    seed: int = 42,
    mediator_cols: Optional[List[str]] = None,
    mediator_weight: float = 0.0,
):

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install torch")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS.copy()

    print("\n" + "=" * 80)
    print("         物理最优策略训练 - 因果注意力版")
    print("=" * 80)
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Loss: {loss_type}")
    print(f"  Tie break: {tie_break}")
    print(f"  Causal mode: {causal_mode}")
    if causal_mode in ["heads", "all"]:
        print(f"  Causal heads: {n_causal_heads}")
    print(f"  Position encoding: {use_pos_encoding}")
    print(f"  Model: d={d_model}, heads={n_heads}, layers={n_layers}")

    if use_pos_encoding:
        print("  ⚠️ 警告: 位置编码已启用。候选顺序在训练中会随机打乱，需要固定候选顺序才合理。")

    print(f"\n[1. 加载数据] {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    形状: {df.shape}")

    print(f"\n[2. 验证数据]")
    try:
        validation_result = validate_candidate_data(df, group_cols, target_col, verbose=True)
    except ValueError as e:
        print(f"\n❌ 数据验证失败: {e}")
        raise

    if validation_result.candidate_counts:
        avg_candidates = np.mean(validation_result.candidate_counts)
        random_baseline = 1.0 / avg_candidates
        print(f"\n[随机基线] P@1 ≈ {random_baseline:.4f} (1/{avg_candidates:.1f} candidates)")

    print(f"\n[3. 生成标签]")
    df, label_info = generate_optimal_labels(df, group_cols, target_col, tie_break=tie_break, verbose=True)

    print(f"\n[4. 特征处理]")
    feature_cols = [
        c for c in df.columns
        if c not in DEFAULT_EXCLUDE_COLS
        and not c.startswith("contrib_")
        and not c.startswith("intensity_")
        and not c.startswith("cf_")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    feature_cols, feature_info = validate_features(df, feature_cols, verbose=True)
    print(f"    有效特征数: {len(feature_cols)}")

    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"\n[5. 划分数据]")
    groups = df.groupby(group_cols)
    group_keys = list(groups.groups.keys())
    random.shuffle(group_keys)

    n_val = int(len(group_keys) * float(os.environ.get('CAT_VAL_FRAC', '0.1')))
    val_keys = set(group_keys[:n_val])

    val_mask = df.apply(lambda r: tuple(r[c] for c in group_cols) in val_keys, axis=1)
    df_train = df[~val_mask]
    df_val = df[val_mask]

    print(f"    Train: {len(group_keys) - n_val} groups")
    print(f"    Val: {n_val} groups")

    print(f"\n[6. 特征标准化]")
    scaler = compute_feature_stats(df_train, feature_cols, verbose=True)

    df_train = scaler.transform(df_train)
    df_val = scaler.transform(df_val)
    print(f"    ✓ 使用训练集统计量标准化")

    print(f"\n[7. 创建数据集]")
    mediator_cols = [c for c in (mediator_cols or []) if c in df_train.columns]
    if mediator_cols:
        print(f" mediator heads: {mediator_cols} (aux weight {mediator_weight})")
    train_dataset = OptimalPolicyDataset(
        df_train, group_cols, feature_cols, "is_optimal",
        max_candidates=max_candidates, is_train=True, tie_break=tie_break,
        mediator_cols=mediator_cols
    )
    val_dataset = OptimalPolicyDataset(
        df_val, group_cols, feature_cols, "is_optimal",
        max_candidates=max_candidates, is_train=False, tie_break=tie_break,
        mediator_cols=mediator_cols
    )

    if len(train_dataset) == 0:
        raise ValueError("训练集为空！请检查数据是否满足候选集学习要求。")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"\n[8. 创建模型]")
    print(f"    因果模式: {causal_mode}")

    active_causal_weights, causal_weights_source = resolve_causal_weights(
        df_train, feature_cols, causal_weights_csv, use_default_weights, target_col)

    if causal_mode in ["bias", "heads", "all"] and CAUSAL_ATTENTION_AVAILABLE:
        print(f"    → 使用 CausalAttentionTransformer")
        model = CausalAttentionTransformer(
            input_dim=len(feature_cols),
            feature_names=feature_cols,
            causal_weights=active_causal_weights,
            causal_mode=causal_mode,
            d_model=d_model,
            n_heads=n_heads,
            n_causal_heads=n_causal_heads,
            n_layers=n_layers,
            d_ff=d_model * 2,
            dropout=0.1,
            max_candidates=max_candidates,
            mediator_names=list(mediator_cols or []),
        ).to(device)
    elif causal_mode == "scaling":
        print(f"    → 使用 OptimalPolicyTransformer (因果特征缩放)")
        model = OptimalPolicyTransformer(
            input_dim=len(feature_cols),
            feature_names=feature_cols,
            causal_weights=active_causal_weights,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_candidates=max_candidates,
            use_pos_encoding=use_pos_encoding,
        ).to(device)
    else:
        if causal_mode != "none" and not CAUSAL_ATTENTION_AVAILABLE:
            print(f"    ⚠️ 因果注意力模块不可用，回退到 scaling 模式")
            causal_mode = "scaling"
        print(f"    → 使用 OptimalPolicyTransformer (标准)")
        model = OptimalPolicyTransformer(
            input_dim=len(feature_cols),
            feature_names=feature_cols,
            causal_weights=active_causal_weights if causal_mode != "none" else {},
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_candidates=max_candidates,
            use_pos_encoding=use_pos_encoding,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    参数量: {n_params:,}")

    matched_weights = [(fname, w) for fname, w in active_causal_weights.items() if fname in feature_cols and w != 0]
    if matched_weights:
        print(f"    匹配的因果权重: {len(matched_weights)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs, base_lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[9. 训练] epochs={epochs}, patience={patience}")
    print("-" * 100)
    header = f"{'Epoch':>6} | {'Train':>8} | {'Val':>8} | {'P@1':>7} | {'P@1-any':>8} | {'P@3':>7} | {'P@5':>7} | {'MRR':>7} | {'Spear':>6} | {'Regret':>7} | {'Rank':>6} | {'LR':>10} | {'Time':>6}"
    print(header)
    print("-" * 100)

    best_p1 = 0.0

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, loss_type, tie_break, reg_temp, reg_huber,
                                 mediator_weight=(mediator_weight if mediator_cols else 0.0))
        val_metrics = evaluate(model, val_loader, device, loss_type, tie_break, reg_temp, reg_huber)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
            "lr": lr_now
        })

        main_metric = val_metrics['p@1_any'] if tie_break == "any" else val_metrics['p@1']

        print(f"{epoch+1:6d} | {train_loss:8.4f} | {val_metrics['loss']:8.4f} | "
              f"{val_metrics['p@1']:7.4f} | {val_metrics['p@1_any']:8.4f} | "
              f"{val_metrics['p@3']:7.4f} | {val_metrics['p@5']:7.4f} | "
              f"{val_metrics['mrr']:7.4f} | {val_metrics.get('spearman',0.0):6.3f} | {val_metrics.get('regret',0.0):7.4f} | {val_metrics['avg_rank']:6.1f} | "
              f"{lr_now:10.6f} | {elapsed:5.1f}s")

        val_loss_now = float(val_metrics.get("loss", float("inf")))
        if main_metric > best_p1 or (main_metric == best_p1 and val_loss_now < best_val_loss):
            best_p1 = main_metric
            best_val_loss = val_loss_now
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "feature_cols": feature_cols,
                "scaler": scaler.to_dict(),
                "config": {
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "max_candidates": max_candidates,
                    "use_pos_encoding": use_pos_encoding,
                    "loss_type": loss_type,
                    "tie_break": tie_break,
                    "group_cols": group_cols,
                    "target_col": target_col,
                    "causal_mode": causal_mode,
                    "n_causal_heads": n_causal_heads,
                    "causal_weights": active_causal_weights,
                    "causal_weights_source": causal_weights_source,
                    "mediator_names": list(mediator_cols or []),
                    "mediator_weight": float(mediator_weight if mediator_cols else 0.0),
                },
                "label_info": label_info,
                "seed": seed,
            }
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))
            print(f"         ✓ Saved best (P@1={val_metrics['p@1']:.4f}, P@1-any={val_metrics['p@1_any']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] epoch {epoch+1}")
                break
        try:
            torch.save({**checkpoint, "epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics}, os.path.join(save_dir, "last.pt"))
        except Exception:
            pass

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history.csv"), index=False)

    print("-" * 100)
    print(f"\n[完成]")
    print(f"  Best {'P@1-any' if tie_break == 'any' else 'P@1'}: {best_p1:.4f}")
    print(f"  随机基线: {random_baseline:.4f}")
    print(f"  提升: {best_p1 / random_baseline:.1f}x")
    print(f"  模型保存: {save_dir}/best.pt")

    return best_p1


def main():
    parser = argparse.ArgumentParser(description="物理最优策略训练 - 修订版")

    parser.add_argument("--csv", type=str, required=True, help="数据集路径")
    parser.add_argument("--save_dir", type=str, default="models/optimal_policy")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "prepare_only", "diagnose"],
                        help="运行模式: train, prepare_only, diagnose")

    parser.add_argument("--target_col", type=str, default="delta_E",
                        help="目标列（用于生成标签）")
    parser.add_argument("--group_cols", type=str, default="episode_id,scene_id,step_index",
                        help="分组列（逗号分隔）")

    parser.add_argument("--tie_break", type=str, default="first",
                        choices=["first", "any"],
                        help="并列处理策略: first（默认）或 any")

    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "bce", "ce_any", "reg", "softrank"],
                        help="损失函数: ce（默认，组内softmax+CE）, bce（对照）, ce_any（多正类CE）")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--use_pos_encoding", action="store_true",
                        help="启用位置编码（默认禁用，因为候选是集合）")
    parser.add_argument("--reg_temp", type=float, default=0.1,
                        help="softrank 温度 (loss=softrank 时, softmax(tau/temp))")
    parser.add_argument("--reg_huber", type=float, default=0.1,
                        help="reg Huber 阈值 (loss=reg 时)")
    parser.add_argument("--no_causal", action="store_true", help="真正关闭因果缩放(消融基线)")
    parser.add_argument("--mediators", type=str, default="",
                        help="comma-separated label columns to predict as CAUSAL MEDIATORS "
                             "with an auxiliary head, e.g. "
                             "cf_delta_E,cf_delta_layers,cf_unlock_L,cf_delta_clear")
    parser.add_argument("--mediator_weight", type=float, default=0.3,
                        help="weight of the auxiliary mediator MSE in the loss (0 disables)")
    parser.add_argument("--causal_mode", type=str, default="scaling",
                        choices=["none", "scaling", "bias", "heads", "all"],
                        help="因果注意力模式: none=无, scaling=特征缩放(默认), "
                             "bias=注意力偏置, heads=因果头, all=全部")
    parser.add_argument("--n_causal_heads", type=int, default=2,
                        help="因果注意力头数（仅 causal_mode=heads/all 时有效）")
    parser.add_argument("--causal_weights_csv", type=str, default=None,
                        help="因果发现产出的 impact_summary.csv 路径；提供后用其去混杂 ATE 作为注入权重")
    parser.add_argument("--use_default_weights", action="store_true",
                        help="显式使用写死的 DEFAULT_CAUSAL_WEIGHTS(含预估值);默认不使用,改为数据驱动")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    global _CAUSAL_ENABLED; _CAUSAL_ENABLED = not args.no_causal
    print(f'    [消融] 因果缩放 enabled = {_CAUSAL_ENABLED}')

    group_cols = args.group_cols.split(",")

    if args.mode == "diagnose":
        df = pd.read_csv(args.csv)

        feature_cols = [
            c for c in df.columns
            if c not in DEFAULT_EXCLUDE_COLS
            and not c.startswith("contrib_")
            and not c.startswith("intensity_")
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        diagnose_training_issues(df, group_cols, args.target_col, feature_cols)

    elif args.mode == "prepare_only":
        df = pd.read_csv(args.csv)
        df, label_info = generate_optimal_labels(df, group_cols, args.target_col, args.tie_break)

        output_path = args.csv.replace(".csv", "_optimal.csv")
        df.to_csv(output_path, index=False)
        print(f"\n保存到: {output_path}")

    else:
        train(
            csv_path=args.csv,
            save_dir=args.save_dir,
            target_col=args.target_col,
            group_cols=group_cols,
            loss_type=args.loss,
            reg_temp=args.reg_temp,
            reg_huber=args.reg_huber,
            tie_break=args.tie_break,
            use_pos_encoding=args.use_pos_encoding,
            causal_mode=args.causal_mode,
            n_causal_heads=args.n_causal_heads,
            causal_weights_csv=args.causal_weights_csv,
            use_default_weights=args.use_default_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_candidates=args.max_candidates,
            device=args.device,
            seed=args.seed,
            mediator_cols=[c.strip() for c in args.mediators.split(",") if c.strip()],
            mediator_weight=args.mediator_weight,
        )


if __name__ == "__main__":
    main()
