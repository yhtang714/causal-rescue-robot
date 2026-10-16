"""The action space: picks and blade pushes, the pile-model stroke, the action features."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import pybullet as p

from causal_core.access import occluders_of

PUSH_DIRS: List[Tuple[float, float]] = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
PUSH_DIST = 0.07
PUSH_INCREMENTS = 14


Action = Tuple


def action_str(a: Action) -> str:
    if a[0] == "none":
        return "none(stuck)"
    if a[0] == "pick":
        return f"pick({a[1]})"
    d = PUSH_DIRS[a[2]]
    return f"push({a[1]},{'+x' if d == (1.0, 0.0) else '-x' if d == (-1.0, 0.0) else '+y' if d == (0.0, 1.0) else '-y'})"


def push_candidates(coord, state: Dict[str, Any]) -> List[int]:
    alive = [int(c) for c in state.get("cube_ids", []) if int(c) not in coord._placed]
    blacklisted = {b for b, n in coord._failure_counts.items() if n >= coord.max_failures_per_block}
    cands = [c for c in alive if c not in blacklisted]
    recs = coord.candidate_set.evaluate_all(cands)
    ok = [int(r.cube_id) for r in recs if r.passes(("C1", "C2", "C3"))]
    return sorted(c for c in ok if coord._inside_pick_tote(c))


def n_options(coord, state: Dict[str, Any]) -> int:
    feas, diag = coord._feasible_set(state)
    if isinstance(diag, dict) and "n_options" in diag:
        return int(diag["n_options"])
    return len(feas) + len(push_candidates(coord, state))


BLADE_HALF_ALONG = 0.020
BLADE_HALF_ACROSS = 0.013
BLADE_HALF_TALL = 0.035
PUSH_STANDOFF = 0.022
PUSH_DEPTH = 0.015
PUSH_EFFECTIVE = 0.025
PUSH_MAX_FORCE = float(os.environ.get("CAUSAL_PUSH_FORCE", 60.0))
BLADE_MASS = 0.3
BLADE_STEPS_PER_INCREMENT = 6

PUSH_COMPLIANT = os.environ.get("PUSH_COMPLIANT", "1") != "0"
PUSH_DESCENT_FORCE = float(os.environ.get("CAUSAL_PUSH_DESCENT_FORCE", 30.0))
BLADE_PAD = dict(half=(0.013, 0.0135, 0.021), dz=0.012)
BLADE_FOLLOWER = dict(half=(0.005, 0.0275, 0.0265), dz=0.0245)
PUSH_DESCENT_STEP = 0.010
PUSH_DESCENT_CLEAR = 0.020
LAST_PUSH_INFO: Dict[str, float] = {}


def apply_push_kinematic(sim, cid: int, dir_index: int, step_fn, dist: float = PUSH_DIST,
                         increments: int = PUSH_INCREMENTS) -> None:
    d = np.array(PUSH_DIRS[int(dir_index) % len(PUSH_DIRS)], dtype=float)
    step = float(dist) / max(1, int(increments))
    for _ in range(int(increments)):
        pos, orn = p.getBasePositionAndOrientation(int(cid))
        p.resetBasePositionAndOrientation(int(cid), [pos[0] + d[0] * step, pos[1] + d[1] * step, pos[2]], orn)
        p.resetBaseVelocity(int(cid), [0, 0, 0], [0, 0, 0])
        step_fn(3)


def blade_start(cid: int, dir_index: int):
    from causal_core.cube_pose import extent_along
    d = np.array(PUSH_DIRS[int(dir_index) % len(PUSH_DIRS)], dtype=float)
    yaw = float(np.arctan2(d[1], d[0]))
    lo, hi = p.getAABB(int(cid))
    pos0 = np.array(p.getBasePositionAndOrientation(int(cid))[0])
    half = 0.5 * float(extent_along(int(cid), yaw))
    back = half + PUSH_STANDOFF + BLADE_HALF_ALONG
    return (np.array([pos0[0] - d[0] * back, pos0[1] - d[1] * back]), float(hi[2]) - PUSH_DEPTH, yaw, d)


def column_top(cid: int, dir_index: int) -> float:
    (sx, sy), z_bottom, yaw, _d = blade_start(cid, dir_index)
    top = -1.0
    c, s_ = np.cos(yaw), np.sin(yaw)
    rf, rt = [], []
    ha, hx = BLADE_PAD["half"][0], BLADE_FOLLOWER["half"][1]
    for ox in (-ha, 0.0, ha):
        for oy in (-hx, -0.5 * hx, 0.0, 0.5 * hx, hx):
            x, y = sx + ox * c - oy * s_, sy + ox * s_ + oy * c
            rf.append([float(x), float(y), 0.9]); rt.append([float(x), float(y), -0.05])
    for r in p.rayTestBatch(rf, rt):
        if int(r[0]) >= 0:
            top = max(top, float(r[3][2]))
    return float(top)


def apply_push(sim, cid: int, dir_index: int, step_fn, dist: float = PUSH_DIST,
               increments: int = PUSH_INCREMENTS, max_force: Optional[float] = None,
               compliant: Optional[bool] = None) -> float:
    fmax = float(PUSH_MAX_FORCE if max_force is None else max_force)
    compliant = PUSH_COMPLIANT if compliant is None else bool(compliant)
    cid = int(cid)
    pos0 = np.array(p.getBasePositionAndOrientation(cid)[0])
    (sx, sy), z_bottom, yaw, d = blade_start(cid, dir_index)
    dv = np.array([d[0], d[1], 0.0])
    z_goal = z_bottom
    pad_below_tip = BLADE_PAD["half"][2] - BLADE_PAD["dz"]
    if compliant:
        top = column_top(cid, dir_index)
        start = np.array([sx, sy, max(z_goal, top + PUSH_DESCENT_CLEAR + pad_below_tip)])
    else:
        start = np.array([sx, sy, z_goal])
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    col = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
        halfExtents=[list(BLADE_PAD["half"]), list(BLADE_FOLLOWER["half"])],
        collisionFramePositions=[[0, 0, BLADE_PAD["dz"]], [0, 0, BLADE_FOLLOWER["dz"]]])
    blade = p.createMultiBody(baseMass=BLADE_MASS, baseCollisionShapeIndex=col, basePosition=start.tolist(),
                              baseOrientation=orn)
    p.changeDynamics(blade, -1, lateralFriction=0.6, linearDamping=0.9, angularDamping=0.9)
    cons = p.createConstraint(blade, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], start.tolist(),
                              childFrameOrientation=orn)
    short = 0.0
    try:
        p.changeConstraint(cons, jointChildPivot=start.tolist(), jointChildFrameOrientation=orn, maxForce=fmax)
        step_fn(4)
        base = start.copy()
        if compliant and start[2] > z_goal + 1e-6:
            q = start.copy()
            while q[2] > z_goal + 1e-6:
                q[2] = max(z_goal, q[2] - PUSH_DESCENT_STEP)
                p.changeConstraint(cons, jointChildPivot=q.tolist(), jointChildFrameOrientation=orn,
                                   maxForce=float(PUSH_DESCENT_FORCE))
                step_fn(BLADE_STEPS_PER_INCREMENT)
            step_fn(6)
            base = np.array(p.getBasePositionAndOrientation(blade)[0])
            short = float(base[2] - z_goal)
            base[2] = float(base[2])
        step = float(dist) / max(1, int(increments))
        for i in range(1, int(increments) + 1):
            q = base + dv * step * i
            p.changeConstraint(cons, jointChildPivot=q.tolist(), jointChildFrameOrientation=orn, maxForce=fmax)
            step_fn(BLADE_STEPS_PER_INCREMENT)
        step_fn(6)
        end = np.array(p.getBasePositionAndOrientation(blade)[0])
    finally:
        p.removeConstraint(cons)
        p.removeBody(blade)
    pos1 = np.array(p.getBasePositionAndOrientation(cid)[0])
    disp = float(np.dot((pos1 - pos0)[:2], d))
    LAST_PUSH_INFO.clear()
    LAST_PUSH_INFO.update(dict(cid=cid, dir=int(dir_index), compliant=bool(compliant), short_m=round(short, 4),
                               travelled_m=round(float(np.dot((end - base)[:2], d)), 4), disp_m=round(disp, 4)))
    return disp


def d3(d):
    return np.array([d[0], d[1], 0.0], dtype=float)


def apply_action(coord, sim, action: Action, step_fn, k: int = 0, settle: int = 0) -> None:
    if action[0] == "pick":
        cid = int(action[1])
        pos, orn = p.getBasePositionAndOrientation(cid)
        p.resetBasePositionAndOrientation(cid, [5.0, 5.0, 0.5 + 0.1 * (k % 40)], orn)
        coord._placed.add(cid)
    elif action[0] == "push":
        apply_push(sim, int(action[1]), int(action[2]), step_fn)
    else:
        raise ValueError(f"unknown action {action!r}")
    if settle:
        step_fn(int(settle))


def undo_bookkeeping(coord, action: Action) -> None:
    if action[0] == "pick":
        coord._placed.discard(int(action[1]))


PUSH_FEATURE_NAMES = ["A_is_push", "A_push_align", "A_push_free_ahead", "A_push_occ_ahead",
                      "A_push_cover_cos",
                      "A_push_frees_clear", "A_push_frees_gate", "A_push_self_gate", "A_push_self_cover",
                      "A_push_column_free", "A_push_col_short"]


def _free_run_ahead(cid: int, d, limit: float = PUSH_DIST) -> float:
    lo, hi = p.getAABB(int(cid))
    c = 0.5 * (np.array(lo) + np.array(hi))
    half = 0.5 * (np.array(hi) - np.array(lo))
    dv = np.array([d[0], d[1], 0.0], dtype=float)
    face = float(abs(dv[0]) * half[0] + abs(dv[1]) * half[1])
    perp = np.array([-dv[1], dv[0], 0.0])
    best = limit
    for s_perp in (-0.6, 0.0, 0.6):
        for fz in (0.25, 0.5, 0.75):
            o = c + perp * (s_perp * float(abs(perp[0]) * half[0] + abs(perp[1]) * half[1])) \
                + dv * (face + 1e-3)
            o[2] = float(lo[2]) + fz * float(hi[2] - lo[2])
            hit = p.rayTest(o.tolist(), (o + dv * limit).tolist())[0]
            if int(hit[0]) >= 0:
                best = min(best, float(hit[2]) * limit)
    return float(max(0.0, min(limit, best)))


def push_structural(gate, state: Dict[str, Any], cid: int, dir_index: int,
                    dist: float = PUSH_EFFECTIVE) -> Dict[str, float]:
    from causal_core.access import C4_REQUIRED_M, _best_axis_clearance
    out = {"A_push_frees_clear": 0.0, "A_push_frees_gate": 0.0,
           "A_push_self_gate": 0.0, "A_push_self_cover": 0.0}
    share = state.get("layer_share", {}) or {}
    if not share:
        share = {int(k): float(v) for k, v in (state.get("occluder_share") or {}).items()}
    out["A_push_self_cover"] = float(share.get(int(cid), 0.0))
    probe = [int(o) for o, w in share.items() if float(w) > 0.0 and int(o) != int(cid)]
    try:
        before = {o: _best_axis_clearance(gate, o) for o in probe}
        self_before = _best_axis_clearance(gate, int(cid))
    except Exception:
        return out
    d = PUSH_DIRS[int(dir_index) % len(PUSH_DIRS)]
    pos, orn = p.getBasePositionAndOrientation(int(cid))
    try:
        p.resetBasePositionAndOrientation(int(cid),
                                          [pos[0] + d[0] * dist, pos[1] + d[1] * dist, pos[2]], orn)
        for o in probe:
            after = _best_axis_clearance(gate, o)
            delta = min(after, C4_REQUIRED_M) - min(before[o], C4_REQUIRED_M)
            out["A_push_frees_clear"] += float(share.get(int(o), 0.0)) * delta / C4_REQUIRED_M
            if before[o] < C4_REQUIRED_M <= after:
                out["A_push_frees_gate"] += float(share.get(int(o), 0.0))
            elif after < C4_REQUIRED_M <= before[o]:
                out["A_push_frees_gate"] -= float(share.get(int(o), 0.0))
        self_after = _best_axis_clearance(gate, int(cid))
        out["A_push_self_gate"] = float(self_before < C4_REQUIRED_M <= self_after)
    finally:
        p.resetBasePositionAndOrientation(int(cid), pos, orn)
    return out


def push_features(coord, state: Dict[str, Any], cid: int, dir_index: int,
                  target_id: Optional[int] = None, blade_clear: Optional[bool] = None) -> Dict[str, float]:
    d = PUSH_DIRS[int(dir_index) % len(PUSH_DIRS)]
    out = {n: 0.0 for n in PUSH_FEATURE_NAMES}
    out["A_is_push"] = 1.0
    try:
        out["A_push_column_free"] = 1.0 if (blade_clear is None or bool(blade_clear)) else 0.0
        _xy, z_bottom, _yaw, _d = blade_start(int(cid), int(dir_index))
        out["A_push_col_short"] = float(np.clip((column_top(int(cid), int(dir_index)) - z_bottom) / 0.05, 0.0, 1.0))
    except Exception:
        pass
    try:
        cpos = np.array(p.getBasePositionAndOrientation(int(cid))[0])
        tid = int(target_id) if target_id is not None else int(getattr(coord, "_target_id", -1))
        tpos = np.array(p.getBasePositionAndOrientation(tid)[0]) if tid >= 0 else cpos
        away = cpos[:2] - tpos[:2]
        n = float(np.linalg.norm(away))
        out["A_push_align"] = float(np.dot(np.array(d), away / n)) if n > 1e-6 else 0.0
        out["A_push_free_ahead"] = _free_run_ahead(int(cid), d)
        occ = occluders_of(state)
        share = state.get("layer_share", {}) or {}
        ahead = 0.0
        for o in set(list(occ) + [int(k) for k in share]):
            if int(o) == int(cid):
                continue
            try:
                q = np.array(p.getBasePositionAndOrientation(int(o))[0])
            except Exception:
                continue
            v = q[:2] - cpos[:2]
            dist = float(np.linalg.norm(v))
            if dist < 1e-6 or dist > PUSH_DIST + 0.05:
                continue
            if float(np.dot(np.array(d), v / dist)) > 0.7:
                ahead += float(share.get(int(o), occ.get(int(o), 0.0)))
        out["A_push_occ_ahead"] = ahead
        cw = np.zeros(2)
        wsum = 0.0
        for o, w in (share or occ).items():
            if int(o) == int(cid) or float(w) <= 0.0:
                continue
            try:
                q = np.array(p.getBasePositionAndOrientation(int(o))[0])
            except Exception:
                continue
            v = q[:2] - cpos[:2]
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                cw += float(w) * v / nv
                wsum += float(w)
        if wsum > 1e-9:
            cw /= wsum
            ncw = float(np.linalg.norm(cw))
            out["A_push_cover_cos"] = float(np.dot(np.array(d), cw / ncw)) if ncw > 1e-6 else 0.0
    except Exception:
        pass
    try:
        out.update(push_structural(coord.candidate_set._gate, state, int(cid), int(dir_index)))
    except Exception:
        pass
    return out


def action_access(coord, state: Dict[str, Any], action: Action,
                  base_access: Dict[int, Dict[str, float]], target_id: Optional[int] = None,
                  blade_clear: Optional[bool] = None) -> Dict[str, float]:
    cid = int(action[1])
    f = dict(base_access.get(cid, {}))
    for n in PUSH_FEATURE_NAMES:
        f.setdefault(n, 0.0)
    if action[0] == "push":
        f.update(push_features(coord, state, cid, int(action[2]), target_id=target_id, blade_clear=blade_clear))
    return f
