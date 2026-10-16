"""Access features, occluder bookkeeping and the interventional label of an action."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

ACCESS_FEATURE_NAMES: List[str] = [
    "A_adj_occ_share", "A_over_occ_share", "A_frees_occ_clear", "A_frees_occ_gate",
    "A_top_margin",
    "A_lean_contacts", "A_lean_occ_share", "A_occ_dist",
    "A_layer_share", "A_layers_below",
    "A_is_push", "A_push_align", "A_push_free_ahead", "A_push_occ_ahead", "A_push_cover_cos",
    "A_push_frees_clear", "A_push_frees_gate", "A_push_self_gate", "A_push_self_cover",
    "A_push_column_free", "A_push_col_short",
]

C4_REQUIRED_M = 0.035
ADJ_XY_M = 0.075
ADJ_DZ_M = 0.040
OVER_XY_M = 0.050


def occluders_of(state: Dict[str, Any]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in (state.get("occluder_share") or {}).items() if float(v) > 0.0}


def _best_axis_clearance(gate, cid: int) -> float:
    from causal_core.cube_pose import grasp_yaws
    lo, hi = p.getAABB(int(cid))
    pos = p.getBasePositionAndOrientation(int(cid))[0]
    sx, sy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    best = 0.0
    for y in grasp_yaws(int(cid)):
        best = max(best, float(gate._side_clear_band(int(cid), pos[0], pos[1], hi[2], y, sx, sy)))
    return best


def access_features(gate, candidates: Iterable[int], occluders: Dict[int, float],
                    alive: Optional[Iterable[int]] = None,
                    state: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    cands = [int(c) for c in candidates]
    if not cands:
        return out
    occ_pos = {}
    for o in occluders:
        try:
            occ_pos[int(o)] = np.array(p.getBasePositionAndOrientation(int(o))[0])
        except Exception:
            pass
    tops = {}
    for b in (alive if alive is not None else list(occluders) + cands):
        try:
            lo, hi = p.getAABB(int(b))
            tops[int(b)] = (np.array(p.getBasePositionAndOrientation(int(b))[0][:2]), float(hi[2]))
        except Exception:
            pass
    for cid in cands:
        f = {n: 0.0 for n in ACCESS_FEATURE_NAMES}
        try:
            cpos = np.array(p.getBasePositionAndOrientation(cid)[0])
        except Exception:
            out[cid] = f
            continue
        if state is not None:
            f["A_layer_share"] = float((state.get("layer_share") or {}).get(cid, 0.0))
            f["A_layers_below"] = float((state.get("layers_below") or {}).get(cid, 0.0))
        try:
            my_top = float(p.getAABB(cid)[1][2])
            leaners = set()
            for cp in p.getContactPoints(bodyA=int(cid)):
                other = int(cp[2])
                if other == cid or other not in tops:
                    continue
                if float(p.getBasePositionAndOrientation(other)[0][2]) > my_top - 0.01:
                    leaners.add(other)
            f["A_lean_contacts"] = float(len(leaners))
            f["A_lean_occ_share"] = float(sum(occluders.get(o, 0.0) for o in leaners))
            if cid in occluders:
                f["A_occ_dist"] = 0.0
            else:
                ds = [float(np.linalg.norm(occ_pos[o][:2] - cpos[:2])) for o in occ_pos]
                f["A_occ_dist"] = float(min(ds)) if ds else 0.3
        except Exception:
            pass
        if cid in tops:
            my_top = tops[cid][1]
            nb = [t for b, (xy, t) in tops.items() if b != cid and float(np.linalg.norm(xy - cpos[:2])) < 0.07]
            f["A_top_margin"] = float(my_top - max(nb)) if nb else 0.10
        probe = []
        for o, share in occluders.items():
            if o == cid or o not in occ_pos:
                continue
            d_xy = float(np.linalg.norm(occ_pos[o][:2] - cpos[:2]))
            dz = float(cpos[2] - occ_pos[o][2])
            if d_xy < ADJ_XY_M and abs(dz) < ADJ_DZ_M:
                f["A_adj_occ_share"] += share
            if d_xy < OVER_XY_M and dz > 0.02:
                f["A_over_occ_share"] += share
            if d_xy < 0.12:
                probe.append(o)
        if probe:
            before = {o: _best_axis_clearance(gate, o) for o in probe}
            pos, orn = p.getBasePositionAndOrientation(cid)
            try:
                p.resetBasePositionAndOrientation(cid, [5.0, 5.0, 2.0], orn)
                for o in probe:
                    after = _best_axis_clearance(gate, o)
                    gain = min(after, C4_REQUIRED_M) - min(before[o], C4_REQUIRED_M)
                    if gain > 0.0:
                        f["A_frees_occ_clear"] += occluders[o] * gain / C4_REQUIRED_M
                    if before[o] < C4_REQUIRED_M <= after:
                        f["A_frees_occ_gate"] += occluders[o]
            finally:
                p.resetBasePositionAndOrientation(cid, pos, orn)
        out[cid] = f
    return out


LAYER_BETA = float(os.environ.get("CAUSAL_LAYER_BETA", 0.5))
CLEAR_BETA = float(os.environ.get("CAUSAL_CLEAR_BETA", 0.25))
UNLOCK_BETA = float(os.environ.get("CAUSAL_UNLOCK_BETA", 0.5))
OPTION_BETA = float(os.environ.get("CAUSAL_OPTION_BETA", 0.15))
DEADEND_BETA = float(os.environ.get("CAUSAL_DEADEND_BETA", 0.50))
OPTION_REF = float(os.environ.get("CAUSAL_OPTION_REF", 8.0))


def cover_share_of(state: Dict[str, Any]) -> Dict[int, float]:
    share = state.get("layer_share", {}) or {}
    if not share:
        share = {int(k): float(v) for k, v in (state.get("occluder_share") or {}).items()}
    return {int(k): float(v) for k, v in share.items() if float(v) > 0.0}


def cover_clearance(gate, ref_share: Dict[int, float], exclude: Iterable[int] = ()) -> float:
    ex = {int(c) for c in exclude}
    total = 0.0
    for cid, w in ref_share.items():
        if int(cid) in ex:
            continue
        try:
            span = _best_axis_clearance(gate, int(cid))
        except Exception:
            continue
        total += float(w) * min(float(span), C4_REQUIRED_M) / C4_REQUIRED_M
    return float(total)


def occluders_of(state: Dict[str, Any]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in (state.get("occluder_share") or {}).items() if float(v) > 0.0}


def _best_axis_clearance(gate, cid: int) -> float:
    from causal_core.cube_pose import grasp_yaws
    lo, hi = p.getAABB(int(cid))
    pos = p.getBasePositionAndOrientation(int(cid))[0]
    sx, sy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    best = 0.0
    for y in grasp_yaws(int(cid)):
        best = max(best, float(gate._side_clear_band(int(cid), pos[0], pos[1], hi[2], y, sx, sy)))
    return best


def access_features(gate, candidates: Iterable[int], occluders: Dict[int, float],
                    alive: Optional[Iterable[int]] = None,
                    state: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    cands = [int(c) for c in candidates]
    if not cands:
        return out
    occ_pos = {}
    for o in occluders:
        try:
            occ_pos[int(o)] = np.array(p.getBasePositionAndOrientation(int(o))[0])
        except Exception:
            pass
    tops = {}
    for b in (alive if alive is not None else list(occluders) + cands):
        try:
            lo, hi = p.getAABB(int(b))
            tops[int(b)] = (np.array(p.getBasePositionAndOrientation(int(b))[0][:2]), float(hi[2]))
        except Exception:
            pass
    for cid in cands:
        f = {n: 0.0 for n in ACCESS_FEATURE_NAMES}
        try:
            cpos = np.array(p.getBasePositionAndOrientation(cid)[0])
        except Exception:
            out[cid] = f
            continue
        if state is not None:
            f["A_layer_share"] = float((state.get("layer_share") or {}).get(cid, 0.0))
            f["A_layers_below"] = float((state.get("layers_below") or {}).get(cid, 0.0))
        try:
            my_top = float(p.getAABB(cid)[1][2])
            leaners = set()
            for cp in p.getContactPoints(bodyA=int(cid)):
                other = int(cp[2])
                if other == cid or other not in tops:
                    continue
                if float(p.getBasePositionAndOrientation(other)[0][2]) > my_top - 0.01:
                    leaners.add(other)
            f["A_lean_contacts"] = float(len(leaners))
            f["A_lean_occ_share"] = float(sum(occluders.get(o, 0.0) for o in leaners))
            if cid in occluders:
                f["A_occ_dist"] = 0.0
            else:
                ds = [float(np.linalg.norm(occ_pos[o][:2] - cpos[:2])) for o in occ_pos]
                f["A_occ_dist"] = float(min(ds)) if ds else 0.3
        except Exception:
            pass
        if cid in tops:
            my_top = tops[cid][1]
            nb = [t for b, (xy, t) in tops.items() if b != cid and float(np.linalg.norm(xy - cpos[:2])) < 0.07]
            f["A_top_margin"] = float(my_top - max(nb)) if nb else 0.10
        probe = []
        for o, share in occluders.items():
            if o == cid or o not in occ_pos:
                continue
            d_xy = float(np.linalg.norm(occ_pos[o][:2] - cpos[:2]))
            dz = float(cpos[2] - occ_pos[o][2])
            if d_xy < ADJ_XY_M and abs(dz) < ADJ_DZ_M:
                f["A_adj_occ_share"] += share
            if d_xy < OVER_XY_M and dz > 0.02:
                f["A_over_occ_share"] += share
            if d_xy < 0.12:
                probe.append(o)
        if probe:
            before = {o: _best_axis_clearance(gate, o) for o in probe}
            pos, orn = p.getBasePositionAndOrientation(cid)
            try:
                p.resetBasePositionAndOrientation(cid, [5.0, 5.0, 2.0], orn)
                for o in probe:
                    after = _best_axis_clearance(gate, o)
                    gain = min(after, C4_REQUIRED_M) - min(before[o], C4_REQUIRED_M)
                    if gain > 0.0:
                        f["A_frees_occ_clear"] += occluders[o] * gain / C4_REQUIRED_M
                    if before[o] < C4_REQUIRED_M <= after:
                        f["A_frees_occ_gate"] += occluders[o]
            finally:
                p.resetBasePositionAndOrientation(cid, pos, orn)
        out[cid] = f
    return out


def value_label(exposure_before: float, exposure_after: float, removed: int,
                occluders_before: Dict[int, float], feasible_before: Sequence[int],
                occluders_after: Dict[int, float], feasible_after: Sequence[int],
                exposed_thresh: float = 0.95, done_bonus: float = 0.5,
                state_before: Optional[Dict[str, Any]] = None,
                state_after: Optional[Dict[str, Any]] = None,
                clear_before: Optional[float] = None,
                clear_after: Optional[float] = None,
                n_options_before: Optional[int] = None,
                n_options_after: Optional[int] = None) -> Dict[str, float]:
    fb, fa = set(int(c) for c in feasible_before), set(int(c) for c in feasible_after)
    dE = float(exposure_after - exposure_before)
    if state_before is not None and state_after is not None:
        d_layers = float(state_before.get("burial_depth_mean", 0.0)) - float(state_after.get("burial_depth_mean", 0.0))
        share_after = state_after.get("layer_share", {}) or {}
        unlock_L = float(sum(float(v) for o, v in share_after.items() if int(o) in fa and int(o) not in fb and int(o) != removed))
    else:
        d_layers = 0.0
        unlock_L = float(sum(share for o, share in occluders_after.items() if o in fa and o not in fb and o != removed)) \
            * max(0.0, 1.0 - float(exposure_after))
    done = float(done_bonus) if (float(exposure_after) >= float(exposed_thresh) > float(exposure_before)) else 0.0
    d_opt, deadend = 0.0, 0.0
    if n_options_before is not None and n_options_after is not None:
        nb, na = float(n_options_before), float(n_options_after)
        d_opt = (min(na, OPTION_REF) - min(nb, OPTION_REF)) / OPTION_REF
        if na <= 0.0 and done <= 0.0:
            deadend = -float(DEADEND_BETA)
    d_clear = 0.0
    if clear_before is not None and clear_after is not None:
        d_clear = float(clear_after) - float(clear_before)
    return {
        "cf_delta_E": dE,
        "cf_occ_share_removed": float(occluders_before.get(int(removed), 0.0)),
        "cf_unlock": float(sum(share for o, share in occluders_after.items() if o in fa and o not in fb and o != removed)),
        "cf_delta_layers": d_layers,
        "cf_unlock_L": unlock_L,
        "cf_delta_clear": d_clear,
        "cf_done": done,
        "cf_delta_options": d_opt,
        "cf_deadend": deadend,
        "cf_value": (dE + LAYER_BETA * d_layers + UNLOCK_BETA * unlock_L + CLEAR_BETA * d_clear
                     + OPTION_BETA * d_opt + deadend + done),
    }
