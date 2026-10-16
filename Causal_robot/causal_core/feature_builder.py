"""Per-candidate feature vectors in the order the ranker expects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

CANONICAL_FEATURE_NAMES: List[str] = [
    "A_cube_x", "A_cube_y", "A_cube_z",
    "A_dx", "A_dy", "A_dz", "A_d", "A_d_xy",
    "A_exposure_proxy", "A_occluder_share", "A_indirect_occluder",
    "A_height_norm", "A_num_neighbors", "A_num_below", "A_num_above",
    "A_is_top", "A_support_depth", "A_is_graspable",
    "A_region_over_hammer", "A_region_xy_near",
    "A_is_in_chain", "A_chain_position", "A_is_chain_top",
    "num_cubes_alive",
    "G_N_above", "G_total_height_above", "G_min_height_above",
    "G_max_height_above", "G_mean_height_above",
    "G_max_support_depth_above", "G_num_top_above",
    "A_adj_occ_share", "A_over_occ_share", "A_frees_occ_clear", "A_frees_occ_gate",
    "A_top_margin",
    "A_lean_contacts", "A_lean_occ_share", "A_occ_dist",
    "A_layer_share", "A_layers_below",
    "A_is_push", "A_push_align", "A_push_free_ahead", "A_push_occ_ahead", "A_push_cover_cos",
    "A_push_frees_clear", "A_push_frees_gate", "A_push_self_gate", "A_push_self_cover",
    "A_push_column_free", "A_push_col_short",
]

LEAKAGE_FEATURE_NAMES: List[str] = ["chosen_from_chain"]


def missing_features(feature_cols: Iterable[str]) -> List[str]:
    known = set(CANONICAL_FEATURE_NAMES)
    return [c for c in feature_cols if c not in known]


def compute_support_depths(num_nodes: int, down_adj: Dict[int, List[int]]) -> np.ndarray:
    depths = np.full(num_nodes, -1, dtype=np.int32)
    visiting: Set[int] = set()

    def dfs(u: int) -> int:
        if depths[u] >= 0:
            return int(depths[u])
        visiting.add(u)
        max_child_depth = 0
        for v in down_adj.get(u, []):
            if v in visiting:
                continue
            d = dfs(int(v)) + 1
            if d > max_child_depth:
                max_child_depth = d
        visiting.remove(u)
        depths[u] = int(max_child_depth)
        return int(depths[u])

    for u in range(num_nodes):
        if depths[u] < 0:
            dfs(int(u))
    return depths


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


@dataclass
class GraphContext:

    positions: np.ndarray
    hammer_pos: np.ndarray
    cube_id_to_node_idx: Dict[int, int]
    num_above_all: np.ndarray
    num_below_all: np.ndarray
    support_depth_all: np.ndarray
    z_min: float
    z_max: float
    G: Dict[str, float] = field(default_factory=dict)
    num_cubes_alive: int = 0

    @classmethod
    def build(cls, node_features: np.ndarray, edge_index: np.ndarray,
              node_types: Optional[np.ndarray], hammer_node_idx: int,
              cube_id_to_node_idx: Dict[int, int],
              xy_radius_above: float = 0.12) -> "GraphContext":
        positions = np.asarray(node_features, dtype=np.float64)[:, :3]
        N = positions.shape[0]
        if not (0 <= hammer_node_idx < N):
            hammer_node_idx = N - 1
        hammer_pos = positions[hammer_node_idx]

        edge_index = np.asarray(edge_index, dtype=np.int64)
        num_above_all = np.zeros(N, dtype=np.int64)
        num_below_all = np.zeros(N, dtype=np.int64)
        down_adj: Dict[int, List[int]] = {}
        if edge_index.ndim == 2 and edge_index.shape[0] == 2 and edge_index.shape[1] > 0:
            src = edge_index[0].astype(int)
            dst = edge_index[1].astype(int)
            for s, d in zip(src, dst):
                if 0 <= s < N and 0 <= d < N:
                    num_above_all[s] += 1
                    num_below_all[d] += 1
                    down_adj.setdefault(int(d), []).append(int(s))
        support_depth_all = compute_support_depths(N, down_adj)

        if node_types is not None and len(node_types) == N:
            is_cube = np.asarray(node_types) == 0
        else:
            is_cube = np.ones(N, dtype=bool)
            is_cube[hammer_node_idx] = False

        rel = positions - hammer_pos[None, :]
        dz_all = rel[:, 2]
        dist_xy_all = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
        zs = positions[:, 2]
        z_min, z_max = float(np.min(zs)), float(np.max(zs))

        above_mask = is_cube & (dz_all > 0.0) & (dist_xy_all <= float(xy_radius_above))
        idx_above = np.where(above_mask)[0]
        if idx_above.size > 0:
            z_rel_above = zs[idx_above] - float(hammer_pos[2])
            G = {
                "G_N_above": float(idx_above.size),
                "G_total_height_above": float(np.sum(z_rel_above)),
                "G_min_height_above": float(np.min(z_rel_above)),
                "G_max_height_above": float(np.max(z_rel_above)),
                "G_mean_height_above": float(np.mean(z_rel_above)),
                "G_max_support_depth_above": float(np.max(support_depth_all[idx_above])),
                "G_num_top_above": float(np.sum(num_above_all[idx_above] == 0)),
            }
        else:
            G = {k: 0.0 for k in (
                "G_N_above", "G_total_height_above", "G_min_height_above",
                "G_max_height_above", "G_mean_height_above",
                "G_max_support_depth_above", "G_num_top_above")}

        return cls(positions=positions, hammer_pos=hammer_pos,
                    cube_id_to_node_idx={int(k): int(v) for k, v in cube_id_to_node_idx.items()},
                    num_above_all=num_above_all, num_below_all=num_below_all,
                    support_depth_all=support_depth_all,
                    z_min=z_min, z_max=z_max, G=G,
                    num_cubes_alive=len(cube_id_to_node_idx))

    def features_for(self, cid: int, *, occluder_share: Dict[int, float],
                      indirect_occluder: Dict[int, float],
                      chain: Optional[Sequence[int]] = None,
                      graspable_ids: Optional[Set[int]] = None,
                      access: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        cid = int(cid)
        if cid not in self.cube_id_to_node_idx:
            raise KeyError(f"cube {cid} not present in this GraphContext's graph")
        idx = self.cube_id_to_node_idx[cid]
        pos = self.positions[idx]

        dx = float(pos[0] - self.hammer_pos[0])
        dy = float(pos[1] - self.hammer_pos[1])
        dz = float(pos[2] - self.hammer_pos[2])
        d_xy = float(np.sqrt(dx ** 2 + dy ** 2))
        d = float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

        dz_range = max(1e-6, self.z_max - self.z_min)
        height_norm = float(np.clip((float(pos[2]) - self.z_min) / dz_range, 0.0, 1.0))

        raw_num_above = int(self.num_above_all[idx])
        raw_num_below = int(self.num_below_all[idx])
        support_depth = int(self.support_depth_all[idx])

        max_above = float(np.max(self.num_above_all)) + 1e-6
        num_above_norm = raw_num_above / max_above
        is_top = 1.0 if num_above_norm < 0.5 else 0.0
        x_logit = 2.0 * height_norm + 1.0 * is_top - 1.0 * num_above_norm
        exposure_proxy = _sigmoid(x_logit)

        region_over_hammer = 1.0 if (dz > 0.0 and d_xy < 0.08) else 0.0
        region_xy_near = 1.0 if d_xy < 0.12 else 0.0
        chain = list(chain) if chain else []
        is_in_chain = 1.0 if cid in chain else 0.0
        chain_position = float(chain.index(cid) + 1) if cid in chain else 0.0
        is_chain_top = 1.0 if (chain and cid == chain[-1]) else 0.0
        num_neighbors = raw_num_above + raw_num_below

        is_graspable = 1.0
        if graspable_ids is not None:
            is_graspable = 1.0 if cid in graspable_ids else 0.0

        occ_share = float(occluder_share.get(cid, 0.0))
        if not np.isfinite(occ_share):
            occ_share = 0.0
        ind_occ = float(indirect_occluder.get(cid, 0.0))
        if not np.isfinite(ind_occ):
            ind_occ = 0.0

        out = {
            "A_cube_x": float(pos[0]), "A_cube_y": float(pos[1]), "A_cube_z": float(pos[2]),
            "A_dx": dx, "A_dy": dy, "A_dz": dz, "A_d": d, "A_d_xy": d_xy,
            "A_exposure_proxy": exposure_proxy,
            "A_occluder_share": occ_share,
            "A_indirect_occluder": ind_occ,
            "A_height_norm": height_norm,
            "A_num_neighbors": float(num_neighbors),
            "A_num_below": float(raw_num_below),
            "A_num_above": float(raw_num_above),
            "A_is_top": is_top,
            "A_support_depth": float(support_depth),
            "A_is_graspable": is_graspable,
            "A_region_over_hammer": region_over_hammer,
            "A_region_xy_near": region_xy_near,
            "A_is_in_chain": is_in_chain,
            "A_chain_position": chain_position,
            "A_is_chain_top": is_chain_top,
            "num_cubes_alive": float(self.num_cubes_alive),
        }
        out.update(self.G)
        for name in ("A_adj_occ_share", "A_over_occ_share", "A_frees_occ_clear", "A_frees_occ_gate", "A_top_margin",
                     "A_lean_contacts", "A_lean_occ_share", "A_occ_dist", "A_layer_share", "A_layers_below",
                     "A_is_push", "A_push_align", "A_push_free_ahead", "A_push_occ_ahead", "A_push_cover_cos",
                     "A_push_frees_clear", "A_push_frees_gate", "A_push_self_gate", "A_push_self_cover",
                     "A_push_column_free", "A_push_col_short"):
            out[name] = float((access or {}).get(name, 0.0))
        return out


def build_occlusion_chain(analyze_fn: Callable[[List[int], int], Dict],
                           target_id: int, alive_cube_ids: Sequence[int],
                           graspable_ids: Sequence[int],
                           max_depth: int = 6) -> List[int]:
    graspable = set(int(x) for x in graspable_ids)
    chain: List[int] = []
    tgt = int(target_id)
    for _ in range(int(max_depth)):
        state = analyze_fn(list(alive_cube_ids), tgt)
        occ = state.get("occluder_share", {}) or {}
        if not occ:
            break
        primary = int(max(occ.items(), key=lambda kv: kv[1])[0])
        chain.append(primary)
        if primary in graspable:
            break
        tgt = primary
    return chain
