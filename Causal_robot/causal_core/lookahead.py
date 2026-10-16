"""Counterfactual search over action sequences in the pile model and the relevance score of a piece."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pybullet as p

from causal_core.access import access_features, occluders_of, value_label


def relevance_scores(gate, state: Dict[str, Any], cands: Sequence[int], alive: Iterable[int]) -> Dict[int, float]:
    occ = occluders_of(state)
    acc = access_features(gate, cands, occ, alive=list(alive), state=state)
    out = {}
    for c in cands:
        a = acc.get(int(c), {})
        out[int(c)] = (2.0 * occ.get(int(c), 0.0) + 2.0 * a.get("A_layer_share", 0.0)
                       + a.get("A_over_occ_share", 0.0) + a.get("A_adj_occ_share", 0.0)
                       + a.get("A_frees_occ_clear", 0.0) + a.get("A_frees_occ_gate", 0.0))
    return out


def cf_search(coord, target_id: int, all_cubes: Sequence[int], state: Dict[str, Any],
              cands: Sequence[int], depth: int, width: int, settle_steps: int,
              step_fn: Callable[[int], None], gamma: float = 0.7,
              first_level_all: bool = True, k0: int = 0,
              settle_deeper: Optional[int] = None,
              feasible_before: Optional[Sequence[int]] = None) -> Dict[int, float]:
    placed_backup = set(coord._placed)
    e0 = float(state["human_exposure"])
    occ0 = occluders_of(state)
    feas0 = [int(c) for c in cands]
    fb = list(feasible_before) if feasible_before is not None else list(cands)
    alive = [c for c in all_cubes if c not in coord._placed]
    if not first_level_all and width > 0 and len(feas0) > width:
        rel = relevance_scores(coord.candidate_set._gate, state, feas0, alive)
        feas0 = sorted(feas0, key=lambda c: -rel[c])[:width]
    values: Dict[int, float] = {}
    sid = p.saveState()
    try:
        for k, cid in enumerate(feas0):
            try:
                pos, orn = p.getBasePositionAndOrientation(int(cid))
                p.resetBasePositionAndOrientation(int(cid), [5.0, 5.0, 0.5 + 0.1 * ((k0 + k) % 40)], orn)
                coord._placed.add(int(cid))
                step_fn(int(settle_steps))
                st1 = coord._observe(target_id, all_cubes)
                f1, _ = coord._feasible_set(st1)
                v = value_label(e0, float(st1["human_exposure"]), int(cid), occ0, fb, occluders_of(st1), f1,
                                exposed_thresh=coord.exposed_thresh, state_before=state, state_after=st1)["cf_value"]
                if depth > 1 and f1 and float(st1["human_exposure"]) < coord.exposed_thresh:
                    sub = cf_search(coord, target_id, all_cubes, st1, sorted(int(c) for c in f1), depth - 1, width,
                                    settle_deeper or settle_steps, step_fn, gamma, first_level_all=False, k0=k0 + 10,
                                    settle_deeper=settle_deeper)
                    v += gamma * max(sub.values(), default=0.0)
                values[int(cid)] = float(v)
            finally:
                coord._placed.clear()
                coord._placed.update(placed_backup)
                p.restoreState(sid)
    finally:
        p.removeState(sid)
        coord._placed.clear()
        coord._placed.update(placed_backup)
    return values
