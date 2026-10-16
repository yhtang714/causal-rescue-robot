"""Loads a ranker checkpoint and scores the candidate actions of a state."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from causal_core.feature_builder import (
    CANONICAL_FEATURE_NAMES,
    GraphContext,
    build_occlusion_chain,
    missing_features,
)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_CAUSAL_WEIGHTS = {
    "A_occluder_share": 0.110217,
    "A_indirect_occluder": 0.08,
    "A_num_above": -0.056563,
    "A_exposure_proxy": -0.028955,
    "A_height_norm": -0.020267,
    "A_dz": -0.007410,
    "A_d_xy": -0.006174,
    "A_num_below": 0.005240,
    "A_support_depth": 0.004714,
}

DEFAULT_FEATURE_COLS = [
    "A_cube_x", "A_cube_y", "A_cube_z",
    "A_dx", "A_dy", "A_dz", "A_d", "A_d_xy",
    "A_exposure_proxy", "A_occluder_share", "A_indirect_occluder",
    "A_height_norm", "A_num_neighbors", "A_num_below", "A_num_above",
    "A_is_top", "A_support_depth", "A_is_graspable",
]

_CHAIN_FEATURES = {"A_is_in_chain", "A_chain_position", "A_is_chain_top"}


NO_FEASIBLE_CANDIDATE = "NO_FEASIBLE_CANDIDATE"


@dataclass
class DecisionResult:
    chosen_block_id: int
    confidence: float
    all_scores: Dict[int, float]
    features_used: Dict[int, Dict[str, float]]
    reason: str = ""


def _build_graph_context(state: Dict[str, Any]) -> GraphContext:
    graph = state.get("graph", {})
    return GraphContext.build(
        node_features=graph.get("node_features", np.zeros((0, 4))),
        edge_index=graph.get("edge_index", np.zeros((2, 0), dtype=np.int64)),
        node_types=graph.get("node_types", None),
        hammer_node_idx=int(graph.get("hammer_node_idx", -1)),
        cube_id_to_node_idx=graph.get("cube_id_to_node_idx", {}),
    )


def _maybe_build_chain(state: Dict[str, Any], feature_cols, graspable_ids,
                        analyze_fn: Optional[Callable] = None,
                        target_id: Optional[int] = None) -> List[int]:
    if not (_CHAIN_FEATURES & set(feature_cols)):
        return []
    if "occlusion_chain" in state:
        chain = state["occlusion_chain"]
        return list(chain.get("chain", chain) if isinstance(chain, dict) else chain)
    if analyze_fn is None or target_id is None:
        raise RuntimeError(
            "This checkpoint's feature_cols require chain features "
            f"({sorted(_CHAIN_FEATURES & set(feature_cols))}) but no "
            "`analyze_fn`/`target_id` was supplied to compute the occlusion "
            "chain online. Pass them from the caller (see robot_demo.coordinator), "
            "or drop chain features from the checkpoint's feature_cols."
        )
    cube_ids = list(state.get("cube_ids", []))
    graspable = graspable_ids if graspable_ids is not None else state.get("graspable_blocks", [])
    return build_occlusion_chain(analyze_fn, target_id, cube_ids, graspable)


class HeuristicPolicy:

    def __init__(self, causal_weights: Dict[str, float] = None):
        self.weights = causal_weights or DEFAULT_CAUSAL_WEIGHTS
        self.name = "Heuristic"

    def select(
        self,
        state: Dict[str, Any],
        feasible_blocks: Optional[List[int]] = None,
        analyze_fn: Optional[Callable] = None,
        target_id: Optional[int] = None,
    ) -> DecisionResult:
        if feasible_blocks is None:
            graspable = state.get("graspable_blocks", [])
        else:
            graspable = feasible_blocks
        if not graspable:
            return DecisionResult(
                chosen_block_id=-1, confidence=0.0, all_scores={}, features_used={},
                reason=NO_FEASIBLE_CANDIDATE,
            )

        ctx = _build_graph_context(state)
        chain = _maybe_build_chain(state, list(self.weights.keys()), set(graspable),
                                    analyze_fn, target_id)
        occluder_share = state.get("occluder_share", {})
        indirect_occ = state.get("indirect_occluder_contribution", {})
        graspable_set = set(int(x) for x in state.get("graspable_blocks", []))

        scores = {}
        features_used = {}
        for cid in graspable:
            features = ctx.features_for(cid, occluder_share=occluder_share,
                                         indirect_occluder=indirect_occ,
                                         chain=chain, graspable_ids=graspable_set)
            features_used[cid] = features
            scores[cid] = sum(self.weights.get(fname, 0.0) * fval
                               for fname, fval in features.items())

        best_cid = max(scores, key=scores.get)
        best_score = scores[best_cid]
        score_values = list(scores.values())
        max_score = max(score_values)
        exp_scores = [np.exp(s - max_score) for s in score_values]
        sum_exp = sum(exp_scores)
        confidence = np.exp(best_score - max_score) / sum_exp

        return DecisionResult(
            chosen_block_id=best_cid, confidence=float(confidence),
            all_scores=scores, features_used=features_used,
        )

    def _extract_features(self, state: Dict[str, Any], cid: int) -> Dict[str, float]:
        ctx = _build_graph_context(state)
        occluder_share = state.get("occluder_share", {})
        indirect_occ = state.get("indirect_occluder_contribution", {})
        graspable_set = set(int(x) for x in state.get("graspable_blocks", []))
        return ctx.features_for(cid, occluder_share=occluder_share,
                                 indirect_occluder=indirect_occ,
                                 chain=[], graspable_ids=graspable_set)


class CausalPolicyAgent:

    def __init__(
        self,
        model_path: Optional[str] = None,
        causal_weights: Dict[str, float] = None,
        device: str = "cpu",
        require_checkpoint: bool = False,
    ):
        self.device = device
        self.causal_weights = causal_weights or DEFAULT_CAUSAL_WEIGHTS
        self.model = None
        self.scaler = None
        self.feature_cols = DEFAULT_FEATURE_COLS
        self.checkpoint_meta: Dict[str, Any] = {}
        self.order_sensitive: Optional[bool] = None
        self.require_checkpoint = bool(require_checkpoint)

        if model_path is None:
            if self.require_checkpoint:
                raise RuntimeError(
                    "CausalPolicyAgent: --policy cat requires a checkpoint but "
                    "no --model PATH was given. CAT mode fails closed: it will "
                    "never silently fall back to the heuristic. Pass "
                    "--model /absolute/path/to/checkpoint.pt, or use "
                    "--policy heuristic explicitly."
                )
            self.name = "CausalHeuristic"
            print("[CausalPolicyAgent] no checkpoint given, heuristic policy")
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "CausalPolicyAgent: a checkpoint was requested but PyTorch is "
                "not installed. Install torch, or use --policy heuristic."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CausalPolicyAgent: checkpoint not found at '{model_path}'"
            )
        self._load_model(model_path)
        self._check_feature_support()
        self._smoke_test()
        self.name = "CausalTransformer"
        print(f"[CausalPolicyAgent] checkpoint loaded: {model_path}")
        print(f"  provenance: {self.checkpoint_meta}")

    def _load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError(f"{model_path}: checkpoint has no 'model_state_dict' key.")

        self.feature_cols = list(checkpoint.get("feature_cols", DEFAULT_FEATURE_COLS))
        cfg = checkpoint.get("config", {})
        d_model = int(cfg.get("d_model", 128))
        n_heads = int(cfg.get("n_heads", 8))
        n_layers = int(cfg.get("n_layers", 4))
        max_candidates = int(cfg.get("max_candidates", 128))
        use_pos_encoding = bool(cfg.get("use_pos_encoding", False))
        dropout = float(cfg.get("dropout", 0.1))
        n_causal_heads = int(cfg.get("n_causal_heads", 2))

        sd = checkpoint["model_state_dict"]
        sd_keys = set(sd.keys())

        d_ff = int(cfg.get("d_ff", 0)) or None
        if d_ff is None:
            d_ff = self._infer_d_ff(sd, d_model)

        is_cat = (any("layers.0.ffn" in k for k in sd_keys)
                  and not any("transformer.layers" in k for k in sd_keys))

        if is_cat:
            causal_mode = cfg.get("causal_mode", "all")
            from .causal_attention_transformer import CausalAttentionTransformer
            self.model = CausalAttentionTransformer(
                input_dim=len(self.feature_cols),
                feature_names=self.feature_cols,
                causal_weights=self.causal_weights,
                causal_mode=causal_mode,
                d_model=d_model, n_heads=n_heads, n_causal_heads=n_causal_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
                max_candidates=max_candidates,
                mediator_names=(list(cfg.get("mediator_names", []) or [])
                                or [f"m{i}" for i in range(int(
                                    sd["mediator_head.3.weight"].shape[0]))]
                                if "mediator_head.3.weight" in sd else
                                list(cfg.get("mediator_names", []) or [])),
            )
            arch_name = f"CausalAttentionTransformer(mode={causal_mode})"
            self._has_learned_pos_encoding = True
        else:
            import sys as _sys, os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            _train_dir = _os.path.join(_os.path.dirname(_here), "training")
            if _train_dir not in _sys.path:
                _sys.path.insert(0, _train_dir)
            from trainer import OptimalPolicyTransformer
            self.model = OptimalPolicyTransformer(
                input_dim=len(self.feature_cols),
                feature_names=self.feature_cols,
                causal_weights=self.causal_weights,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                dropout=dropout, max_candidates=max_candidates,
                use_pos_encoding=use_pos_encoding,
            )
            arch_name = "OptimalPolicyTransformer (vanilla)"
            self._has_learned_pos_encoding = bool(use_pos_encoding)

        self.scaler = checkpoint.get("scaler", None)
        if self.scaler is not None:
            sc_cols = list(self.scaler.get("feature_cols", self.feature_cols))
            mean = np.asarray(self.scaler["mean"], dtype=np.float32)
            std = np.asarray(self.scaler["std"], dtype=np.float32)
            if len(sc_cols) != len(mean) or len(sc_cols) != len(std):
                raise RuntimeError(
                    f"{model_path}: scaler feature_cols ({len(sc_cols)}) does not "
                    f"match mean/std length ({len(mean)}/{len(std)})."
                )
            eps = float(self.scaler.get("eps", 1e-6))
            self.scaler = {
                "feature_cols": sc_cols, "mean": mean,
                "std": np.where(std < eps, 1.0, std).astype(np.float32),
                "eps": eps,
            }
            missing_in_model = [c for c in sc_cols if c not in self.feature_cols]
            if missing_in_model:
                raise RuntimeError(
                    f"{model_path}: scaler references features {missing_in_model} "
                    f"absent from the model's own feature_cols — checkpoint is "
                    f"internally inconsistent."
                )

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_meta = {
            "path": os.path.abspath(model_path),
            "arch": arch_name,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "n_causal_heads": n_causal_heads if is_cat else None,
            "d_ff": d_ff, "dropout": dropout, "max_candidates": max_candidates,
            "causal_mode": cfg.get("causal_mode"),
            "has_learned_pos_encoding": self._has_learned_pos_encoding,
            "use_pos_encoding_config": use_pos_encoding,
            "feature_cols": list(self.feature_cols),
            "n_features": len(self.feature_cols),
            "epoch": checkpoint.get("epoch"),
            "val_metrics": checkpoint.get("val_metrics"),
            "has_scaler": self.scaler is not None,
        }

    @staticmethod
    def _infer_d_ff(state_dict, d_model: int) -> int:
        for key, tensor in state_dict.items():
            if key.endswith(".weight") and "ffn" in key and tensor.dim() == 2:
                out_dim, in_dim = tensor.shape
                if in_dim == d_model and out_dim != d_model:
                    return int(out_dim)
        return d_model * 2

    def _check_feature_support(self):
        bad = missing_features(self.feature_cols)
        if bad:
            raise RuntimeError(
                f"Checkpoint requests feature(s) {bad} that "
                f"causal_core.feature_builder cannot compute online or offline. "
                f"Known features: {CANONICAL_FEATURE_NAMES}. Refusing to load "
                f"(no silent zero-fill)."
            )

    def _smoke_test(self, n_candidates: int = 4, n_trials: int = 6):
        rng = np.random.default_rng(0)
        x_np = rng.normal(size=(n_candidates, len(self.feature_cols))).astype(np.float32)
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = torch.ones(1, n_candidates, dtype=torch.float32).to(self.device)
            logits = self.model(x, mask)
        if not torch.isfinite(logits).all():
            raise RuntimeError("CausalPolicyAgent: smoke test produced non-finite logits.")

        base_order = list(range(n_candidates))
        base_scores = logits[0, :n_candidates].cpu().numpy().copy()
        base_top1 = int(np.argmax(base_scores))
        flips = 0
        for _ in range(n_trials):
            perm = rng.permutation(n_candidates)
            with torch.no_grad():
                xp = torch.tensor(x_np[perm], dtype=torch.float32).unsqueeze(0).to(self.device)
                logits_p = self.model(xp, mask)
            scores_p = logits_p[0, :n_candidates].cpu().numpy()
            restored = np.empty_like(scores_p)
            restored[perm] = scores_p
            top1_restored = int(np.argmax(restored))
            if top1_restored != base_top1 or not np.allclose(restored, base_scores, atol=1e-4):
                flips += 1
        self.order_sensitive = flips > 0
        if self.order_sensitive:
            print(f"[CausalPolicyAgent] checkpoint is candidate-order sensitive "
                  f"({flips}/{n_trials} permutations changed the scores); candidates are ordered by piece id")
        self.checkpoint_meta["order_sensitive"] = self.order_sensitive

    def select(
        self,
        state: Dict[str, Any],
        feasible_blocks: Optional[List[int]] = None,
        analyze_fn: Optional[Callable] = None,
        target_id: Optional[int] = None,
    ) -> DecisionResult:
        if feasible_blocks is None:
            graspable = state.get("graspable_blocks", [])
        else:
            graspable = feasible_blocks

        if not graspable:
            return DecisionResult(
                chosen_block_id=-1, confidence=0.0, all_scores={}, features_used={},
                reason=NO_FEASIBLE_CANDIDATE,
            )

        graspable = sorted(int(c) for c in graspable)

        ctx = _build_graph_context(state)
        chain = _maybe_build_chain(state, self.feature_cols, set(graspable),
                                    analyze_fn, target_id)
        occluder_share = state.get("occluder_share", {})
        indirect_occ = state.get("indirect_occluder_contribution", {})
        graspable_set = set(int(x) for x in state.get("graspable_blocks", graspable))

        access_all = state.get("access_features", {}) or {}
        features_list = []
        features_used = {}
        for cid in graspable:
            feat_dict = ctx.features_for(cid, occluder_share=occluder_share,
                                          indirect_occluder=indirect_occ,
                                          chain=chain, graspable_ids=graspable_set,
                                          access=access_all.get(int(cid)))
            features_used[cid] = feat_dict
            missing = [c for c in self.feature_cols if c not in feat_dict]
            if missing:
                raise RuntimeError(
                    f"CausalPolicyAgent: checkpoint requests feature(s) {missing} "
                    f"that the online feature builder cannot compute for cube "
                    f"{cid}. Refusing to zero-fill silently."
                )
            features_list.append([feat_dict[col] for col in self.feature_cols])

        if self.model is not None and TORCH_AVAILABLE:
            return self._model_inference(graspable, features_list, features_used)
        else:
            return self._heuristic_inference(graspable, features_used)

    def select_actions(
        self,
        state: Dict[str, Any],
        actions: List[tuple],
        access_by_action: Dict[int, Dict[str, float]],
        analyze_fn: Optional[Callable] = None,
        target_id: Optional[int] = None,
    ) -> DecisionResult:
        if not actions:
            return DecisionResult(chosen_block_id=-1, confidence=0.0, all_scores={},
                                  features_used={}, reason=NO_FEASIBLE_CANDIDATE)
        ctx = _build_graph_context(state)
        cids = sorted({int(a[1]) for a in actions})
        chain = _maybe_build_chain(state, self.feature_cols, set(cids), analyze_fn, target_id)
        occluder_share = state.get("occluder_share", {})
        indirect_occ = state.get("indirect_occluder_contribution", {})
        graspable_set = set(int(x) for x in state.get("graspable_blocks", cids))
        features_list, features_used = [], {}
        for i, act in enumerate(actions):
            cid = int(act[1])
            feat = ctx.features_for(cid, occluder_share=occluder_share, indirect_occluder=indirect_occ,
                                    chain=chain, graspable_ids=graspable_set,
                                    access=access_by_action.get(i, {}))
            missing = [c for c in self.feature_cols if c not in feat]
            if missing:
                raise RuntimeError(
                    f"CausalPolicyAgent: checkpoint requests feature(s) {missing} that the online "
                    f"feature builder cannot compute for action {act}. Refusing to zero-fill silently.")
            features_used[i] = feat
            features_list.append([feat[col] for col in self.feature_cols])
        keys = list(range(len(actions)))
        if self.model is not None and TORCH_AVAILABLE:
            return self._model_inference(keys, features_list, features_used)
        return self._heuristic_inference(keys, features_used)

    def _model_inference(
        self,
        graspable: List[int],
        features_list: List[List[float]],
        features_used: Dict[int, Dict[str, float]],
    ) -> DecisionResult:
        with torch.no_grad():
            x_np = np.asarray(features_list, dtype=np.float32)
            if self.scaler is not None:
                sc_cols = list(self.scaler["feature_cols"])
                sc_mean = self.scaler["mean"]
                sc_std = self.scaler["std"]
                if sc_cols == list(self.feature_cols):
                    x_np = (x_np - sc_mean) / sc_std
                else:
                    aligned_mean = np.zeros(len(self.feature_cols), dtype=np.float32)
                    aligned_std = np.ones(len(self.feature_cols), dtype=np.float32)
                    for i, col in enumerate(self.feature_cols):
                        if col in sc_cols:
                            j = sc_cols.index(col)
                            aligned_mean[i] = sc_mean[j]
                            aligned_std[i] = sc_std[j]
                    x_np = (x_np - aligned_mean) / aligned_std

            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = torch.ones(1, len(graspable), dtype=torch.float32).to(self.device)
            logits = self.model(x, mask)
            if not torch.isfinite(logits).all():
                raise RuntimeError("CausalPolicyAgent: model produced non-finite scores at inference.")
            probs = F.softmax(logits[0, :len(graspable)], dim=0)
            best_idx = torch.argmax(probs).item()
            confidence = probs[best_idx].item()

        all_scores = {cid: float(probs[i].item()) for i, cid in enumerate(graspable)}
        return DecisionResult(
            chosen_block_id=int(graspable[best_idx]), confidence=float(confidence),
            all_scores=all_scores, features_used=features_used,
        )

    def _heuristic_inference(
        self,
        graspable: List[int],
        features_used: Dict[int, Dict[str, float]],
    ) -> DecisionResult:
        scores = {}
        for cid in graspable:
            features = features_used.get(cid, {})
            scores[cid] = sum(self.causal_weights.get(fname, 0.0) * fval
                               for fname, fval in features.items())

        score_values = list(scores.values())
        max_score = max(score_values) if score_values else 0
        exp_scores = {cid: np.exp(s - max_score) for cid, s in scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = {cid: exp_s / sum_exp for cid, exp_s in exp_scores.items()}
        best_cid = max(probs, key=probs.get)

        return DecisionResult(
            chosen_block_id=best_cid, confidence=float(probs[best_cid]),
            all_scores=probs, features_used=features_used,
        )

    def _extract_features(self, state: Dict[str, Any], cid: int) -> Dict[str, float]:
        ctx = _build_graph_context(state)
        occluder_share = state.get("occluder_share", {})
        indirect_occ = state.get("indirect_occluder_contribution", {})
        graspable_set = set(int(x) for x in state.get("graspable_blocks", []))
        return ctx.features_for(cid, occluder_share=occluder_share,
                                 indirect_occluder=indirect_occ,
                                 chain=[], graspable_ids=graspable_set)
