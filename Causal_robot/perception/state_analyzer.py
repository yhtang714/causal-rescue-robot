"""Exposure, burial depth and occluder shares from vertical rays over the target."""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p


def get_aabb(body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    aabb = p.getAABB(int(body_id))
    return np.asarray(aabb[0], dtype=np.float32), np.asarray(aabb[1], dtype=np.float32)


def target_footprint_mask(target_id: int, xys: List[Tuple[float, float]]) -> Optional[List[bool]]:
    try:
        shapes = p.getCollisionShapeData(int(target_id), -1)
        base_pos, base_orn = p.getBasePositionAndOrientation(int(target_id))
    except Exception:
        return None
    if not shapes or any(int(s[2]) != p.GEOM_BOX for s in shapes):
        return None
    polys = []
    for s in shapes:
        dims = np.asarray(s[3], dtype=float)
        he = dims / 2.0
        lpos, lorn = s[5], s[6]
        wpos, worn = p.multiplyTransforms(base_pos, base_orn, lpos, lorn)
        R = np.asarray(p.getMatrixFromQuaternion(worn), dtype=float).reshape(3, 3)
        corners = np.array([[sx * he[0], sy * he[1], sz * he[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pts = (R @ corners.T).T[:, :2] + np.asarray(wpos[:2])
        pts = sorted(set((float(a), float(b)) for a, b in pts))
        if len(pts) < 3:
            continue

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower, upper = [], []
        for q in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
                lower.pop()
            lower.append(q)
        for q in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
                upper.pop()
            upper.append(q)
        polys.append(lower[:-1] + upper[:-1])

    def inside(pt, poly):
        n = len(poly)
        for i in range(n):
            if cross(poly[i], poly[(i + 1) % n], pt) < -1e-9:
                return False
        return True
    return [any(inside(xy, poly) for poly in polys) for xy in xys]


def compute_vcf_exposure_and_occluders(
    target_id: int,
    cube_ids: List[int],
    grid_n: int = 16,
    top_margin: float = 1.0,

    coverage: float = 1.0,
    noise_std: float = 0.0,
    rng=None,
    footprint_only: bool = True,) -> Tuple[float, Dict[int, float], Dict[int, int]]:
    try:
        aabb_min, aabb_max = p.getAABB(int(target_id))
    except Exception:
        return 1.0, {}, {}

    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    z_top = float(aabb_max[2]) + float(top_margin)

    ray_from_list = []
    ray_to_list = []

    xs = np.linspace(float(x0), float(x1), int(grid_n))
    ys = np.linspace(float(y0), float(y1), int(grid_n))

    xys = [(float(xi), float(yi)) for xi in xs for yi in ys]
    mask = target_footprint_mask(target_id, xys) if footprint_only else None
    z_end = float(z0) - 1e-3 if mask is not None else float(z1)
    for k, (xi, yi) in enumerate(xys):
        if mask is not None and not mask[k]:
            continue
        ray_from_list.append([xi, yi, z_top])
        ray_to_list.append([xi, yi, z_end])

    if not ray_from_list:
        return 1.0, {}, {}

    try:
        results = p.rayTestBatch(ray_from_list, ray_to_list)
    except Exception:
        return 1.0, {}, {}

    if rng is None:
        import numpy as _np_po
        rng = _np_po.random.default_rng()
    if coverage < 1.0:
        _kept = [res for res in results if rng.random() <= coverage]
        results = _kept if _kept else list(results)[:1]
    if noise_std > 0.0:
        _noisy = []
        for res in list(results):
            if rng.random() < noise_std:
                _noisy.append((-1,) + tuple(res[1:]))
            else:
                _noisy.append(res)
        results = _noisy
    cube_set = set(int(c) for c in cube_ids)
    occluder_hits: Dict[int, int] = {}
    free_count = 0
    total = len(results)

    for res in results:
        hit_id = int(res[0])
        if hit_id < 0 or hit_id == int(target_id):
            free_count += 1
        elif hit_id in cube_set:
            occluder_hits[hit_id] = occluder_hits.get(hit_id, 0) + 1

    exposure = float(free_count) / float(max(1, total))

    occluder_share: Dict[int, float] = {}
    blocked = total - free_count
    if blocked > 0:
        for cid, cnt in occluder_hits.items():
            occluder_share[int(cid)] = float(cnt) / float(blocked)

    return float(exposure), occluder_share, occluder_hits


def compute_burial_depth(
    target_id: int,
    cube_ids: List[int],
    grid_n: int = 16,
    top_margin: float = 1.0,
    max_layers: int = 10,
) -> Dict[str, Any]:
    try:
        aabb_min, aabb_max = p.getAABB(int(target_id))
    except Exception:
        return {"depth_mean": 0.0, "depth_hist": {}, "layer_share": {}, "below_mean": {}, "n_rays": 0}
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    xs = np.linspace(float(x0), float(x1), int(grid_n))
    ys = np.linspace(float(y0), float(y1), int(grid_n))
    xys = [(float(xi), float(yi)) for xi in xs for yi in ys]
    mask = target_footprint_mask(target_id, xys)
    if mask is not None:
        xys = [xy for xy, m in zip(xys, mask) if m]
    if not xys:
        return {"depth_mean": 0.0, "depth_hist": {}, "layer_share": {}, "below_mean": {}, "n_rays": 0}
    cube_set = set(int(c) for c in cube_ids)
    z_top = float(z1) + float(top_margin)
    z_end = float(z0) - 1e-3
    n = len(xys)
    chains: List[List[int]] = [[] for _ in range(n)]
    active = list(range(n))
    starts = [z_top] * n
    for _layer in range(int(max_layers)):
        if not active:
            break
        rf = [[xys[i][0], xys[i][1], starts[i]] for i in active]
        rt = [[xys[i][0], xys[i][1], z_end] for i in active]
        try:
            res = p.rayTestBatch(rf, rt)
        except Exception:
            break
        nxt = []
        for i, r in zip(active, res):
            hid = int(r[0])
            if hid < 0 or hid == int(target_id):
                continue
            if hid in cube_set and (not chains[i] or chains[i][-1] != hid):
                chains[i].append(hid)
            hz = float(r[3][2])
            starts[i] = hz - 1.5e-3
            if starts[i] > z_end:
                nxt.append(i)
        active = nxt
    depths = [len(c) for c in chains]
    hist: Dict[int, int] = {}
    for d in depths:
        hist[d] = hist.get(d, 0) + 1
    share: Dict[int, float] = {}
    below_sum: Dict[int, float] = {}
    for c in chains:
        for k, cid in enumerate(c):
            share[cid] = share.get(cid, 0.0) + 1.0
            below_sum[cid] = below_sum.get(cid, 0.0) + float(len(c) - 1 - k)
    below_mean = {cid: below_sum[cid] / share[cid] for cid in share}
    share = {cid: v / float(n) for cid, v in share.items()}
    return {"depth_mean": float(sum(depths)) / float(n), "depth_hist": hist, "layer_share": share,
            "below_mean": below_mean, "n_rays": n}


def build_contact_graph(
    cube_ids: List[int],
    target_id: int,
    contact_z_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int], int]:
    all_ids = [int(x) for x in cube_ids] + [int(target_id)]
    N = len(all_ids)

    node_features = []
    node_types = []
    cube_id_to_node_idx: Dict[int, int] = {}

    for idx, bid in enumerate(all_ids):
        try:
            pos, _ = p.getBasePositionAndOrientation(int(bid))
        except Exception:
            pos = (0.0, 0.0, 0.0)
        is_target = 1 if int(bid) == int(target_id) else 0
        if is_target == 0:
            cube_id_to_node_idx[int(bid)] = int(idx)
        node_features.append([pos[0], pos[1], pos[2], float(is_target)])
        node_types.append(is_target)

    node_features = np.asarray(node_features, dtype=np.float32)
    node_types = np.asarray(node_types, dtype=np.int64)

    edges_src = []
    edges_dst = []
    for i in range(N):
        for j in range(i + 1, N):
            A = int(all_ids[i])
            B = int(all_ids[j])
            try:
                cps = p.getContactPoints(A, B)
            except Exception:
                cps = []
            if not cps:
                continue
            try:
                posA, _ = p.getBasePositionAndOrientation(A)
                posB, _ = p.getBasePositionAndOrientation(B)
            except Exception:
                continue
            zA = float(posA[2])
            zB = float(posB[2])
            if zA > zB + float(contact_z_eps):
                edges_src.append(j)
                edges_dst.append(i)
            elif zB > zA + float(contact_z_eps):
                edges_src.append(i)
                edges_dst.append(j)
            else:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])

    if edges_src:
        edge_index = np.stack([
            np.asarray(edges_src, dtype=np.int64),
            np.asarray(edges_dst, dtype=np.int64)
        ], axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    hammer_node_idx = N - 1
    return node_features, edge_index, node_types, cube_id_to_node_idx, hammer_node_idx


def compute_indirect_occluder_contribution(
    cube_ids: List[int],
    cube_id_to_node_idx: Dict[int, int],
    edge_index: np.ndarray,
    occluder_share: Dict[int, float],
    decay: float = 0.8,
) -> Dict[int, float]:
    if edge_index.shape[1] == 0:
        return {int(cid): float(occluder_share.get(cid, 0.0)) for cid in cube_ids}

    presses_on: Dict[int, List[int]] = {}
    n_edges = edge_index.shape[1]
    for e in range(n_edges):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])
        if dst not in presses_on:
            presses_on[dst] = []
        presses_on[dst].append(src)

    node_idx_to_cube_id: Dict[int, int] = {}
    for cid, nidx in cube_id_to_node_idx.items():
        node_idx_to_cube_id[nidx] = cid

    indirect_contribution: Dict[int, float] = {}

    for cid in cube_ids:
        if cid not in cube_id_to_node_idx:
            indirect_contribution[int(cid)] = float(occluder_share.get(cid, 0.0))
            continue

        start_node = cube_id_to_node_idx[cid]
        visited: Set[int] = set()
        queue = deque()

        if start_node in presses_on:
            for neighbor in presses_on[start_node]:
                queue.append((neighbor, 1))

        total_contribution = 0.0

        while queue:
            node_idx, depth = queue.popleft()
            if node_idx in visited:
                continue
            visited.add(node_idx)

            if node_idx in node_idx_to_cube_id:
                pressed_cid = node_idx_to_cube_id[node_idx]
                share = occluder_share.get(pressed_cid, 0.0)
                total_contribution += share * (decay ** (depth - 1))

            if node_idx in presses_on:
                for neighbor in presses_on[node_idx]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        direct_share = occluder_share.get(cid, 0.0)
        indirect_contribution[int(cid)] = float(direct_share + total_contribution)

    return indirect_contribution


def compute_block_exposures(
    cube_ids: List[int],
    node_features: np.ndarray,
    edge_index: np.ndarray,
    cube_id_to_node_idx: Dict[int, int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    if not cube_ids:
        return {}, {}, {}

    num_above = np.zeros((node_features.shape[0],), dtype=np.float32)
    if edge_index.shape[1] > 0:
        src = edge_index[0]
        for s in src:
            num_above[int(s)] += 1.0

    z_vals = node_features[:, 2].astype(np.float32)
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    dz = max(1e-6, z_max - z_min)

    num_above_norm = num_above / max(1.0, float(np.max(num_above) + 1e-6))

    exposures: Dict[int, float] = {}
    height_norm: Dict[int, float] = {}
    num_above_norm_by_cid: Dict[int, float] = {}

    for cid in cube_ids:
        if int(cid) not in cube_id_to_node_idx:
            continue
        idx = int(cube_id_to_node_idx[int(cid)])
        z = float(node_features[idx, 2])
        hn = float(np.clip((z - z_min) / dz, 0.0, 1.0))
        height_norm[int(cid)] = hn
        above_n = float(num_above_norm[idx])
        num_above_norm_by_cid[int(cid)] = above_n

        is_top = 1.0 if above_n < 0.5 else 0.0
        x = 2.0 * hn + 1.0 * is_top - 1.0 * above_n
        expo = 1.0 / (1.0 + math.exp(-x))
        exposures[int(cid)] = float(np.clip(expo, 0.0, 1.0))

    return exposures, height_norm, num_above_norm_by_cid


class StateAnalyzer:

    def __init__(
        self,
        block_graspable_threshold: float = 0.75,
        exposure_grid_res: int = 16,
        prism_height: float = 1.0,
    ):
        self.block_graspable_threshold = block_graspable_threshold
        self.exposure_grid_res = exposure_grid_res
        self.prism_height = prism_height

    def analyze(
        self,
        alive_cube_ids: List[int],
        target_id: int,

        partial_obs: bool = False,
        coverage: float = 0.4,
        noise_std: float = 0.1,
        obs_rng=None,) -> Dict[str, Any]:
        cube_ids = [int(cid) for cid in alive_cube_ids]

        node_features, edge_index, node_types, cube_id_to_node_idx, hammer_node_idx = \
            build_contact_graph(cube_ids, int(target_id))

        block_exposures, block_height_norm, block_num_above_norm = compute_block_exposures(
            cube_ids, node_features, edge_index, cube_id_to_node_idx
        )

        graspable_blocks = [
            cid for cid in cube_ids
            if float(block_exposures.get(int(cid), 0.0)) >= self.block_graspable_threshold
        ]

        exposure, occluder_share, occluder_hits = compute_vcf_exposure_and_occluders(
            target_id=int(target_id),
            cube_ids=cube_ids,
            grid_n=self.exposure_grid_res,
            top_margin=self.prism_height,
            coverage=(coverage if partial_obs else 1.0),
            noise_std=(noise_std if partial_obs else 0.0),
            rng=obs_rng,
        )

        indirect_occluder_contribution = compute_indirect_occluder_contribution(
            cube_ids, cube_id_to_node_idx, edge_index, occluder_share, decay=0.8
        )

        num_below: Dict[int, int] = {}
        if edge_index.shape[1] > 0:
            for e in range(edge_index.shape[1]):
                dst = int(edge_index[1, e])
                if dst in cube_id_to_node_idx.values():
                    for cid, nidx in cube_id_to_node_idx.items():
                        if nidx == dst:
                            num_below[cid] = num_below.get(cid, 0) + 1

        support_depth = self._compute_support_depths(
            cube_ids, cube_id_to_node_idx, edge_index
        )

        burial = compute_burial_depth(int(target_id), cube_ids, grid_n=self.exposure_grid_res,
                                      top_margin=self.prism_height)

        return {
            "human_exposure": float(exposure),
            "occluder_share": occluder_share,
            "occluder_hits": occluder_hits,
            "burial_depth_mean": burial["depth_mean"],
            "burial_depth_hist": burial["depth_hist"],
            "layer_share": burial["layer_share"],
            "layers_below": burial["below_mean"],
            "indirect_occluder_contribution": indirect_occluder_contribution,
            "block_exposures": block_exposures,
            "block_height_norm": block_height_norm,
            "block_num_above_norm": block_num_above_norm,
            "num_below": num_below,
            "support_depth": support_depth,
            "graspable_blocks": graspable_blocks,
            "cube_ids": cube_ids,
            "graph": {
                "node_features": node_features,
                "edge_index": edge_index,
                "node_types": node_types,
                "cube_id_to_node_idx": cube_id_to_node_idx,
                "hammer_node_idx": hammer_node_idx,
            },
        }

    def _compute_support_depths(
        self,
        cube_ids: List[int],
        cube_id_to_node_idx: Dict[int, int],
        edge_index: np.ndarray,
    ) -> Dict[int, int]:
        if edge_index.shape[1] == 0:
            return {cid: 0 for cid in cube_ids}

        down_adj: Dict[int, List[int]] = {}
        for e in range(edge_index.shape[1]):
            src = int(edge_index[0, e])
            dst = int(edge_index[1, e])
            if dst not in down_adj:
                down_adj[dst] = []
            down_adj[dst].append(src)

        num_nodes = len(cube_ids) + 1
        depths = {}

        def dfs(node: int, visited: Set[int]) -> int:
            if node in depths:
                return depths[node]
            if node in visited:
                return 0
            visited.add(node)

            max_depth = 0
            for child in down_adj.get(node, []):
                d = dfs(child, visited) + 1
                max_depth = max(max_depth, d)

            depths[node] = max_depth
            return max_depth

        result = {}
        for cid in cube_ids:
            if cid in cube_id_to_node_idx:
                nidx = cube_id_to_node_idx[cid]
                result[cid] = dfs(nidx, set())
            else:
                result[cid] = 0

        return result
