"""The closed loop: observe, legal actions, rank, check the top K in the pile model, execute on the arm, verify, settle."""
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pybullet as p

from perception import StateAnalyzer
from causal_core.candidate_set import CorrectedCandidateSet
from decision.causal_policy_agent import CausalPolicyAgent, DecisionResult, NO_FEASIBLE_CANDIDATE
from production_robot_controller import ProductionRobotController, GraspRequest, GraspStatus

SETTLE_STEPS = 400

DEFAULT_EXPOSED_THRESH = 0.95

FLATNESS_DOT_THRESH = 0.70


def _is_flat_enough(cube_id: int, thresh: float = FLATNESS_DOT_THRESH) -> bool:
    try:
        _, orn = p.getBasePositionAndOrientation(cube_id)
        rot = p.getMatrixFromQuaternion(orn)
        return max(abs(rot[2]), abs(rot[5]), abs(rot[8])) >= thresh
    except Exception:
        return False


@dataclass
class StepLog:
    step: int
    state: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorResult:
    seed: int
    policy: str
    checkpoint_meta: Optional[Dict[str, Any]]
    exposed_thresh: float
    initial_exposure: float
    final_exposure: float
    steps_completed: int
    successful_placements: int
    failed_attempts: int
    blacklisted: List[int]
    rescue_success: bool
    termination_reason: str
    step_log: List[Dict[str, Any]]
    execution_time: float


class RescueCoordinator:
    def __init__(self, sim, controller: Optional[ProductionRobotController] = None,
                 policy: str = "heuristic", model_path: Optional[str] = None,
                 exposed_thresh: float = DEFAULT_EXPOSED_THRESH,
                 max_failures_per_block: int = 1,
                 max_steps: int = 20, enable_logging: bool = True,
                 place_layout: str = "opposite", nudge_mm: float = 0.0,
                 lookahead: int = 0, lookahead_settle: int = 120, lookahead_gamma: float = 0.7,
                 lookahead_width: int = 3, lookahead_deep_width: int = 5, lookahead_deep_budget: int = 3,
                 actions: str = "pick", push_top: int = 4, verify_k: int = 0):
        self.sim = sim
        self.verify_k = int(verify_k)
        self._push_history = {}
        self._picks_since_push = {}
        self.controller = controller or ProductionRobotController(sim=sim, enable_logging=enable_logging)
        self.analyzer = StateAnalyzer()
        self.candidate_set = CorrectedCandidateSet(sim)
        self.lookahead = int(lookahead)
        self.lookahead_settle = int(lookahead_settle)
        self.lookahead_gamma = float(lookahead_gamma)
        self.lookahead_width = int(lookahead_width)
        self.lookahead_deep_width = int(lookahead_deep_width)
        self.lookahead_deep_budget = int(lookahead_deep_budget)
        self._deep_used = 0
        self.actions_mode = str(actions)
        self.push_top = int(push_top)
        self.nudge_m = max(0.0, float(nudge_mm) / 1000.0)
        self.controller.neighbour_nudge_allow = self.nudge_m
        if self.nudge_m > 0.0:
            self.candidate_set.feas_cfg.safety_margin = max(
                0.0, self.candidate_set.feas_cfg.safety_margin - self.nudge_m)
            self.candidate_set.feas_cfg.finger_thickness = max(
                0.0, self.candidate_set.feas_cfg.finger_thickness
                - max(0.0, self.nudge_m - 0.005))
        self.exposed_thresh = float(exposed_thresh)
        self.max_failures_per_block = max_failures_per_block
        self.max_steps = max_steps
        self.enable_logging = enable_logging
        self.place_layout = place_layout

        self.policy_name = policy
        if policy == "cat":
            self.policy_agent = CausalPolicyAgent(model_path=model_path, require_checkpoint=True)
        else:
            self.policy_agent = CausalPolicyAgent()

        self._failure_counts: Dict[int, int] = {}
        self._placed: set = set()
        self.step_log: List[StepLog] = []

    def _log(self, msg: str):
        if self.enable_logging:
            print(f"[coordinator] {msg}")

    _HUD_TEXT_ID = None
    _TINT_FEASIBLE = (1.0, 1.0, 0.2, 1.0)
    _TINT_SELECTED = (0.15, 1.0, 0.15, 1.0)
    _TINT_FAILED = (1.0, 0.15, 0.15, 1.0)

    def _is_gui(self) -> bool:
        try:
            return p.getConnectionInfo()["connectionMethod"] == p.GUI
        except Exception:
            return False

    def _update_gui(self, step: int, exposure: float, feasible: List[int],
                    selected: Optional[int], scores: Dict[int, float], status: str):
        if not self._is_gui():
            return
        try:
            ckpt_path = None
            if self.policy_name == "cat":
                ckpt_path = (self.policy_agent.checkpoint_meta or {}).get("path")
            ckpt_label = os.path.basename(ckpt_path) if ckpt_path else "none (heuristic)"
            top_score = scores.get(selected) if selected is not None else None
            lines = [
                f"step {step}  policy={self.policy_name}  checkpoint={ckpt_label}",
                f"exposure {exposure:.3f} / threshold {self.exposed_thresh:.2f}",
                f"selected cube={selected}  score={top_score if top_score is None else round(top_score, 3)}",
                f"controller state=EXECUTING  status={status}",
            ]
            text = "\n".join(lines)
            self._HUD_TEXT_ID = p.addUserDebugText(
                text, [0.0, 0.0, 0.55], textColorRGB=[1, 1, 1], textSize=1.3,
                replaceItemUniqueId=self._HUD_TEXT_ID if self._HUD_TEXT_ID is not None else -1,
            )

            for cid in getattr(self, "_last_tinted", set()):
                orig = self._original_colors.get(cid)
                if orig is not None:
                    try:
                        p.changeVisualShape(cid, -1, rgbaColor=orig)
                    except Exception:
                        pass
            if not hasattr(self, "_original_colors"):
                self._original_colors: Dict[int, tuple] = {}
            newly_tinted = set()
            for cid in feasible:
                if cid not in self._original_colors:
                    try:
                        vis = p.getVisualShapeData(cid)
                        self._original_colors[cid] = vis[0][7] if vis else (0.6, 0.6, 0.6, 1.0)
                    except Exception:
                        self._original_colors[cid] = (0.6, 0.6, 0.6, 1.0)
                try:
                    p.changeVisualShape(cid, -1, rgbaColor=self._TINT_FEASIBLE)
                    newly_tinted.add(cid)
                except Exception:
                    pass
            if selected is not None:
                try:
                    p.changeVisualShape(selected, -1, rgbaColor=self._TINT_SELECTED)
                    newly_tinted.add(selected)
                except Exception:
                    pass
            self._last_tinted = newly_tinted
        except Exception as e:
            self._log(f"  (GUI overlay update failed, continuing without it: {e})")

    def _mark_failed_gui(self, cube_id: int):
        if not self._is_gui():
            return
        try:
            if not hasattr(self, "_original_colors"):
                self._original_colors: Dict[int, tuple] = {}
            if cube_id not in self._original_colors:
                vis = p.getVisualShapeData(cube_id)
                self._original_colors[cube_id] = vis[0][7] if vis else (0.6, 0.6, 0.6, 1.0)
            p.changeVisualShape(cube_id, -1, rgbaColor=self._TINT_FAILED)
        except Exception:
            pass

    def _observe(self, target_id: int, all_cube_ids: List[int]) -> Dict[str, Any]:
        alive = [c for c in all_cube_ids if c not in self._placed]
        state = self.analyzer.analyze(alive, target_id)
        return state

    def _feasible_set(self, state: Dict[str, Any]):
        alive_all = [int(c) for c in state.get("cube_ids", state["graspable_blocks"])]
        alive_graspable = [c for c in alive_all if c not in self._placed]
        blacklisted = {b for b, n in self._failure_counts.items() if n >= self.max_failures_per_block}
        candidates_after_blacklist = [c for c in alive_graspable if c not in blacklisted]

        all_recs = self.candidate_set.evaluate_all(candidates_after_blacklist)
        gate_names = ("C1", "C2", "C3", "C4")
        gate_failures = {int(r.cube_id): r.failed(gate_names) for r in all_recs}
        gated = [int(r.cube_id) for r in all_recs if r.passes(gate_names)]

        in_tote = [c for c in gated if self._inside_pick_tote(c)]
        outside = sorted(set(gated) - set(in_tote))
        if outside:
            self._log(f"  excluded {len(outside)} cube(s) lying outside the pick tote "
                      f"(not part of the pile over the target): {outside}")

        flat = [c for c in in_tote if _is_flat_enough(c)]
        non_flat = sorted(set(in_tote) - set(flat))
        if non_flat:
            self._log(f"  excluded {len(non_flat)} cube(s) tilted beyond 45deg (resting on an "
                      f"edge/corner; outside the measured grasp envelope): {non_flat}")

        clear, descent_blocked = [], {}
        for c in flat:
            ok, info = self.controller.descent_clear_for_any_yaw(c)
            if ok:
                clear.append(c)
            else:
                descent_blocked[c] = info
        if descent_blocked:
            self._log(f"  excluded {len(descent_blocked)} cube(s) whose descent would hit a neighbour "
                      f"(C6): {sorted(descent_blocked)}")

        ordered = sorted(clear)
        push_capable = sorted(int(r.cube_id) for r in all_recs
                              if r.passes(("C1", "C2", "C3")) and self._inside_pick_tote(int(r.cube_id)))
        diagnostics = {
            "push_capable": push_capable,
            "n_options": len(ordered) + len([c for c in push_capable if c not in set(ordered)]),
            "blacklisted_excluded": sorted(blacklisted),
            "gate_failures": {cid: fails for cid, fails in gate_failures.items() if fails},
            "outside_pick_tote_excluded": outside,
            "flatness_excluded": non_flat,
            "descent_blocked_excluded": {int(k): v for k, v in descent_blocked.items()},
            "ordering": "sorted by cube id",
        }
        return ordered, diagnostics

    INEFFECTIVE_PUSH_M = 0.012
    VERIFY_DIRECT_MIN = 0.02
    VERIFY_STUCK_MAX = 8
    VERIFY_TWO_STEP_MAX = 4

    def _push_memory_veto(self, cid: int, dir_index: int):
        hist = self._push_history.get(int(cid))
        if not hist:
            return None
        last_step, last_dir, last_disp = hist[-1]
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if int(dir_index) == opposite.get(int(last_dir), -1) and self._picks_since_push.get(int(cid), 0) < 2:
            return f"would reverse the previous stroke on {cid}"
        same = [h for h in hist if int(h[1]) == int(dir_index)]
        if same and float(same[-1][2]) < self.INEFFECTIVE_PUSH_M and self._picks_since_push.get(int(cid), 0) < 1:
            return f"last {['+x','-x','+y','-y'][int(dir_index)]} stroke on {cid} moved it {same[-1][2]*1000:.0f}mm and the pile is unchanged"
        return None

    def _verify_reselect_actions(self, decision, state, actions, target_id, all_cube_ids, step):
        from causal_core.access import cover_clearance, cover_share_of, occluders_of, value_label
        from causal_core.actions import action_str, apply_action, undo_bookkeeping
        sc = decision.all_scores or {}
        order = sorted(range(len(actions)), key=lambda i: -float(sc.get(i, 0.0)))
        top = order[:max(1, int(self.verify_k))]
        e0 = float(state["human_exposure"])
        occ = occluders_of(state)
        gate = self.candidate_set._gate
        ref_share = cover_share_of(state)
        feas0 = sorted(int(a[1]) for a in actions if a[0] == "pick")
        n_opt_b = len(actions)
        step_fn = lambda n: [p.stepSimulation() for _ in range(int(n))]
        placed_backup = set(self._placed)
        hist_backup = {k: list(v) for k, v in self._push_history.items()}
        results = {}
        for i in top:
            a = actions[i]
            clear_b = cover_clearance(gate, ref_share, exclude=[a[1]])
            sid = p.saveState()
            try:
                apply_action(self, self.sim, a, step_fn, k=i)
                step_fn(self.lookahead_settle)
                st2 = self._observe(target_id, all_cube_ids)
                f2, _ = self._feasible_set(st2)
                n_acts2 = self._count_actions_fast(st2, sorted(int(c) for c in f2), all_cube_ids)
                rem = int(a[1]) if a[0] == "pick" else -1
                lab = value_label(e0, float(st2["human_exposure"]), rem, occ, feas0, occluders_of(st2), f2,
                                  exposed_thresh=self.exposed_thresh, state_before=state, state_after=st2,
                                  clear_before=clear_b, clear_after=cover_clearance(gate, ref_share, exclude=[a[1]]),
                                  n_options_before=n_opt_b, n_options_after=n_acts2)
                trap = (lab["cf_delta_E"] < -0.005 or lab["cf_delta_layers"] < -0.05
                        or (n_acts2 == 0 and lab["cf_done"] <= 0.0))
                results[i] = dict(action=action_str(a), rank_score=float(sc.get(i, 0.0)),
                                  cf_value=float(lab["cf_value"]), dE=float(lab["cf_delta_E"]),
                                  d_layers=float(lab["cf_delta_layers"]), d_options=float(lab["cf_delta_options"]),
                                  n_options_after=n_acts2, trap=bool(trap), done=bool(lab["cf_done"] > 0))
            finally:
                undo_bookkeeping(self, a)
                p.restoreState(sid)
                p.removeState(sid)
                self._placed = set(placed_backup)
                self._push_history = {k: list(v) for k, v in hist_backup.items()}
        ok = [i for i in top if not results[i]["trap"]]
        from causal_core.access import LAYER_BETA as _LB
        def direct(i):
            r = results[i]
            return r["dE"] + _LB * r["d_layers"] + (0.5 if r.get("done") else 0.0)
        pool = ok if ok else list(top)
        two_step = {}
        if max(direct(i) for i in pool) < self.VERIFY_DIRECT_MIN and len(actions) > len(top):
            extra = [i for i in order if i not in top][:max(0, self.VERIFY_STUCK_MAX - len(top))]
            for i in extra:
                a = actions[i]
                clear_b = cover_clearance(gate, ref_share, exclude=[a[1]])
                sid = p.saveState()
                try:
                    apply_action(self, self.sim, a, step_fn, k=i)
                    step_fn(self.lookahead_settle)
                    st2 = self._observe(target_id, all_cube_ids)
                    f2, _ = self._feasible_set(st2)
                    n_acts2 = self._count_actions_fast(st2, sorted(int(c) for c in f2), all_cube_ids)
                    rem = int(a[1]) if a[0] == "pick" else -1
                    lab = value_label(e0, float(st2["human_exposure"]), rem, occ, feas0, occluders_of(st2), f2,
                                      exposed_thresh=self.exposed_thresh, state_before=state, state_after=st2,
                                      clear_before=clear_b, clear_after=cover_clearance(gate, ref_share, exclude=[a[1]]),
                                      n_options_before=n_opt_b, n_options_after=n_acts2)
                    trap = (lab["cf_delta_E"] < -0.005 or lab["cf_delta_layers"] < -0.05
                            or (n_acts2 == 0 and lab["cf_done"] <= 0.0))
                    results[i] = dict(action=action_str(a), rank_score=float(sc.get(i, 0.0)),
                                      cf_value=float(lab["cf_value"]), dE=float(lab["cf_delta_E"]),
                                      d_layers=float(lab["cf_delta_layers"]), d_options=float(lab["cf_delta_options"]),
                                      n_options_after=n_acts2, trap=bool(trap), done=bool(lab["cf_done"] > 0), widened=True)
                finally:
                    undo_bookkeeping(self, a)
                    p.restoreState(sid)
                    p.removeState(sid)
                    self._placed = set(placed_backup)
                    self._push_history = {k: list(v) for k, v in hist_backup.items()}
            top = list(top) + extra
            ok = [i for i in top if not results[i]["trap"]]
            pool = ok if ok else list(top)
        if max(direct(i) for i in pool) >= self.VERIFY_DIRECT_MIN:
            best = max(pool, key=lambda i: (direct(i), results[i]["cf_value"]))
        else:
            pool = sorted(pool, key=lambda i: -results[i]["cf_value"])[:self.VERIFY_TWO_STEP_MAX]
            for i in pool:
                a = actions[i]
                sid = p.saveState()
                try:
                    apply_action(self, self.sim, a, step_fn, k=i)
                    step_fn(self.lookahead_settle)
                    st2 = self._observe(target_id, all_cube_ids)
                    e1 = float(st2["human_exposure"])
                    f2, _ = self._feasible_set(st2)
                    acts2, access2, _ = self._build_action_set(st2, sorted(int(c) for c in f2), target_id, all_cube_ids)
                    best2, best2_str = 0.0, None
                    if acts2:
                        dec2 = self.policy_agent.select_actions(st2, acts2, access2)
                        sc2 = dec2.all_scores or {}
                        for j in sorted(range(len(acts2)), key=lambda j: -float(sc2.get(j, 0.0)))[:2]:
                            sid2 = p.saveState()
                            try:
                                apply_action(self, self.sim, acts2[j], step_fn, k=20 + j)
                                step_fn(self.lookahead_settle)
                                st3 = self._observe(target_id, all_cube_ids)
                                d2 = (float(st3["human_exposure"]) - e1
                                      + _LB * (float(st2.get("burial_depth_mean", 0.0)) - float(st3.get("burial_depth_mean", 0.0)))
                                      + (0.5 if float(st3["human_exposure"]) >= self.exposed_thresh > e1 else 0.0))
                                if d2 > best2:
                                    best2, best2_str = d2, action_str(acts2[j])
                            finally:
                                undo_bookkeeping(self, acts2[j])
                                p.restoreState(sid2)
                                p.removeState(sid2)
                    two_step[i] = dict(direct2=float(best2), via=best2_str,
                                       value=float(direct(i) + self.lookahead_gamma * best2))
                finally:
                    undo_bookkeeping(self, a)
                    p.restoreState(sid)
                    p.removeState(sid)
                    self._placed = set(placed_backup)
                    self._push_history = {k: list(v) for k, v in hist_backup.items()}
            best = max(pool, key=lambda i: (two_step[i]["value"], results[i]["cf_value"]))
            for i in pool:
                results[i]["two_step"] = two_step[i]
        info = dict(k=len(top), verified=results, chosen=int(best), ranker_top=int(top[0]),
                    overrode_ranker=bool(best != top[0]), all_traps=not ok, stuck=bool(two_step))
        for i in top:
            r = results[i]
            ts = r.get("two_step")
            self._log(f"  verify {r['action']:>12s}: rank {r['rank_score']:.3f}  cf {r['cf_value']:+.3f}  "
                      f"dE {r['dE']:+.3f}  dLayers {r['d_layers']:+.2f}  options-> {r['n_options_after']}"
                      f"{'  TRAP' if r['trap'] else ''}"
                      f"{('  then ' + str(ts['via']) + ' -> %+.3f' % ts['direct2']) if ts else ''}"
                      f"{'  <- chosen' if i == best else ''}")
        return int(best), info

    def _build_action_set(self, state, feasible, target_id, all_cube_ids):
        from causal_core.access import access_features, occluders_of
        from causal_core.actions import PUSH_DIRS, action_access, action_str, push_candidates
        from causal_core.lookahead import relevance_scores
        from production_robot_controller import PushRequest
        actions = [("pick", int(c)) for c in feasible]
        clear_flags = {}
        try:
            pc = push_candidates(self, state)
            pickable = {int(c) for c in feasible}
            pc = [c for c in pc if int(c) not in pickable]
            if pc and self.push_top > 0:
                alive = [c for c in all_cube_ids if c not in self._placed]
                rel = relevance_scores(self.candidate_set._gate, state, pc, alive)
                for cid in sorted(pc, key=lambda c: -rel[int(c)])[:self.push_top]:
                    for d in range(len(PUSH_DIRS)):
                        veto = self._push_memory_veto(int(cid), d)
                        if veto:
                            self._log(f"  push({cid},{d}) withheld: {veto}")
                            clear_flags[(int(cid), d)] = False
                            continue
                        ok, c7info = self.controller.push_descent_clear(PushRequest(target_id=int(cid), dir_index=d))
                        clear_flags[(int(cid), d)] = bool(c7info.get("column_free", ok)) if isinstance(c7info, dict) else bool(ok)
                        if ok:
                            actions.append(("push", int(cid), d))
        except Exception as e:
            self._log(f"  (push candidates unavailable: {e})")
        occ = occluders_of(state)
        base = access_features(self.candidate_set._gate, sorted({int(a[1]) for a in actions}), occ,
                               alive=[c for c in all_cube_ids if c not in self._placed], state=state)
        by_action = {i: action_access(self, state, a, base, target_id=target_id,
                                      blade_clear=clear_flags.get((int(a[1]), a[2] if a[0] == "push" else -1), True))
                     for i, a in enumerate(actions)}
        n_push = sum(1 for a in actions if a[0] == "push")
        self._log(f"  action set: {len(actions)-n_push} picks + {n_push} pushes")
        return actions, by_action, action_str

    def _count_actions_fast(self, state, feasible, all_cube_ids) -> int:
        from causal_core.actions import PUSH_DIRS, column_top, push_candidates
        from causal_core.lookahead import relevance_scores
        from production_robot_controller import PushRequest
        n = len(feasible)
        try:
            pickable = {int(c) for c in feasible}
            pc = [c for c in push_candidates(self, state) if int(c) not in pickable]
            if not pc or self.push_top <= 0:
                return n
            alive = [c for c in all_cube_ids if c not in self._placed]
            rel = relevance_scores(self.candidate_set._gate, state, pc, alive)
            lo, hi = self.controller._tray_interior_bounds(int(self.sim.get_ids()["toteA"]))
            m = self.controller._PUSH_WALL_MARGIN
            for cid in sorted(pc, key=lambda c: -rel[int(c)])[:self.push_top]:
                for d in range(len(PUSH_DIRS)):
                    if self._push_memory_veto(int(cid), d):
                        continue
                    pose = self.controller.push_start_pose(PushRequest(target_id=int(cid), dir_index=d))
                    if pose is None:
                        continue
                    sx, sy, sz = pose["start"]
                    if not (lo[0] + m <= sx <= hi[0] - m and lo[1] + m <= sy <= hi[1] - m):
                        continue
                    if column_top(int(cid), d) - float(sz) > self.controller._PUSH_SKID_MAX:
                        continue
                    n += 1
        except Exception as e:
            self._log(f"  (fast action count unavailable: {e})")
        return n

    def _lookahead_reselect(self, decision, state, feasible, target_id, all_cube_ids):
        from causal_core.lookahead import cf_search, relevance_scores
        ranked = sorted(feasible, key=lambda c: -float(decision.all_scores.get(c, 0.0)))
        top = ranked[:self.lookahead]
        step_fn = lambda n: [p.stepSimulation() for _ in range(int(n))]
        scores = cf_search(self, target_id, all_cube_ids, state, list(top), depth=2,
                           width=self.lookahead_width, settle_steps=self.lookahead_settle, step_fn=step_fn,
                           gamma=self.lookahead_gamma, feasible_before=feasible)
        best = max(top, key=lambda c: (scores.get(int(c), 0.0), float(decision.all_scores.get(c, 0.0))))
        stuck = False
        deep_scores = {}
        if max(scores.values(), default=0.0) <= 1e-6:
            stuck = True
            if self.lookahead_deep_width > 0 and self._deep_used < self.lookahead_deep_budget:
                self._deep_used += 1
                alive = [c for c in all_cube_ids if c not in self._placed]
                rel = relevance_scores(self.candidate_set._gate, state, feasible, alive)
                deep = sorted(feasible, key=lambda c: -rel[int(c)])[:self.lookahead_deep_width]
                deep_scores = cf_search(self, target_id, all_cube_ids, state, list(deep), depth=3, width=3,
                                        settle_steps=self.lookahead_settle, step_fn=step_fn,
                                        gamma=self.lookahead_gamma, feasible_before=feasible)
                if max(deep_scores.values(), default=0.0) > 1e-6:
                    best = max(deep, key=lambda c: (deep_scores.get(int(c), 0.0), float(decision.all_scores.get(c, 0.0))))
            if max(deep_scores.values(), default=0.0) <= 1e-6:
                acc = state.get("access_features", {}) or {}

                def _unlock(c):
                    a = acc.get(int(c), {})
                    return (a.get("A_frees_occ_clear", 0.0) + a.get("A_frees_occ_gate", 0.0)
                            + 0.5 * a.get("A_adj_occ_share", 0.0) + 0.25 * a.get("A_lean_occ_share", 0.0))
                cand = max(feasible, key=lambda c: (_unlock(c), float(decision.all_scores.get(c, 0.0))))
                if _unlock(cand) > 0.0:
                    best = cand
        info = {"top": [int(c) for c in top], "cf_scores": {int(k): round(v, 4) for k, v in scores.items()},
                "deep_scores": {int(k): round(v, 4) for k, v in deep_scores.items()},
                "ranker_choice": int(decision.chosen_block_id), "chosen": int(best), "stuck": stuck}
        if int(best) != int(decision.chosen_block_id):
            self._log(f"  lookahead overrode the ranker: {decision.chosen_block_id} -> {best} "
                      f"({info['cf_scores']}{' deep ' + str(info['deep_scores']) if deep_scores else ''})")
            from decision.causal_policy_agent import DecisionResult
            decision = DecisionResult(chosen_block_id=int(best), confidence=float(decision.all_scores.get(best, 0.0)),
                                      all_scores=decision.all_scores, features_used=decision.features_used,
                                      reason=(decision.reason or "") + " +lookahead")
        return decision, info

    @staticmethod
    def _snapshot_positions(cube_ids) -> Dict[int, tuple]:
        out = {}
        for c in cube_ids:
            try:
                out[int(c)] = tuple(p.getBasePositionAndOrientation(int(c))[0])
            except Exception:
                pass
        return out

    @staticmethod
    def _pile_disturbance(before: Dict[int, tuple]) -> Dict[str, Any]:
        moved1, moved3, dmax = [], [], 0.0
        per = {}
        for c, pos0 in before.items():
            try:
                pos1 = p.getBasePositionAndOrientation(int(c))[0]
            except Exception:
                continue
            d = float(math.sqrt(sum((pos1[i] - pos0[i]) ** 2 for i in range(3))))
            per[int(c)] = round(d, 4)
            dmax = max(dmax, d)
            if d > 0.01:
                moved1.append(int(c))
            if d > 0.03:
                moved3.append(int(c))
        return {"moved_over_1cm": sorted(moved1), "moved_over_3cm": sorted(moved3),
                "displacement_over_3cm_m": {c: per[c] for c in sorted(moved3)},
                "max_displacement_m": round(dmax, 4), "n_tracked": len(per)}

    def _inside_pick_tote(self, cid: int) -> bool:
        try:
            tote_id = int(self.sim.get_ids()["toteA"])
            bounds = self.controller._tray_interior_bounds(tote_id)
            pos = p.getBasePositionAndOrientation(cid)[0]
        except Exception as e:
            self._log(f"  C0 gate could not evaluate cube {cid} ({e}); excluding it (fail closed)")
            return False
        if bounds is None:
            self._log("  C0 gate: pick tote interior bounds unavailable; excluding all (fail closed)")
            return False
        lo, hi = bounds
        return bool(lo[0] <= pos[0] <= hi[0] and lo[1] <= pos[1] <= hi[1])

    def run(self, target_id: int, seed: int = 0) -> CoordinatorResult:
        t0 = time.time()
        ids = self.sim.get_ids()
        all_cube_ids = [int(c) for c in ids.get("cubes", [])]

        state = self._observe(target_id, all_cube_ids)
        initial_exposure = float(state["human_exposure"])
        self._log(f"initial exposure={initial_exposure:.3f}, cubes={len(all_cube_ids)}")

        step = 0
        final_exposure = initial_exposure
        termination_reason = "max_steps_reached"
        successful = 0
        failed = 0

        while step < self.max_steps:
            state = self._observe(target_id, all_cube_ids)
            current_exposure = float(state["human_exposure"])
            self.step_log.append(StepLog(step, "OBSERVE", {"exposure": current_exposure}))

            if current_exposure >= self.exposed_thresh:
                termination_reason = "exposure_threshold_reached"
                final_exposure = current_exposure
                break

            feasible, feasible_diag = self._feasible_set(state)
            self.step_log.append(StepLog(step, "FEASIBLE_SET",
                                         {"candidates": feasible, **feasible_diag}))
            if not feasible and self.actions_mode != "pick_push":
                termination_reason = "no_feasible_candidate"
                final_exposure = current_exposure
                self._log("no feasible candidate remains (gates C1-C4) -- stopping, not widening")
                break

            try:
                from causal_core.access import access_features, occluders_of
                state["access_features"] = access_features(self.candidate_set._gate, feasible,
                                                           occluders_of(state),
                                                           alive=[c for c in all_cube_ids if c not in self._placed],
                                                           state=state)
            except Exception as e:
                self._log(f"  (access features unavailable: {e})")
                state["access_features"] = {}
            action_list, action_str_of = None, None
            if self.actions_mode == "pick_push":
                action_list, access_by_action, action_str_of = self._build_action_set(state, feasible,
                                                                                       target_id, all_cube_ids)
                if not action_list:
                    termination_reason = "no_feasible_candidate"
                    final_exposure = current_exposure
                    self._log("no pick and no push remains -- stopping")
                    break
                decision = self.policy_agent.select_actions(state, action_list, access_by_action)
            else:
                decision = self.policy_agent.select(state, feasible)
            lookahead_info = None
            if (self.actions_mode == "pick_push" and self.verify_k > 0
                    and decision.chosen_block_id >= 0 and len(action_list) > 1):
                idx, lookahead_info = self._verify_reselect_actions(decision, state, action_list,
                                                                    target_id, all_cube_ids, step)
                if idx != int(decision.chosen_block_id):
                    self._log(f"  verification overrides the ranker: {action_str_of(action_list[int(decision.chosen_block_id)])}"
                              f" -> {action_str_of(action_list[idx])}")
                decision = DecisionResult(chosen_block_id=int(idx), confidence=decision.confidence,
                                          all_scores=decision.all_scores, features_used=decision.features_used,
                                          reason=decision.reason)
            if (self.actions_mode == "pick" and self.lookahead > 0
                    and decision.chosen_block_id >= 0 and len(feasible) > 1):
                decision, lookahead_info = self._lookahead_reselect(decision, state, feasible,
                                                                    target_id, all_cube_ids)
            self.step_log.append(StepLog(step, "RANK", {
                "lookahead": lookahead_info,
                "chosen": decision.chosen_block_id,
                "actions": [action_str_of(a) for a in action_list] if action_list else None,
                "all_scores": {int(k): float(v) for k, v in decision.all_scores.items()},
                "features_used": {int(k): dict(v) for k, v in decision.features_used.items()},
                "reason": decision.reason,
            }))
            if decision.chosen_block_id < 0 or decision.reason == NO_FEASIBLE_CANDIDATE:
                termination_reason = "no_feasible_candidate"
                final_exposure = current_exposure
                break
            if action_list is not None:
                action = action_list[int(decision.chosen_block_id)]
                chosen = int(action[1])
            else:
                action = ("pick", int(decision.chosen_block_id))
                chosen = int(decision.chosen_block_id)

            self._update_gui(step=step, exposure=current_exposure, feasible=feasible,
                             selected=chosen, scores=decision.all_scores, status="executing")

            self._log(f"step {step}: exposure={current_exposure:.3f}, "
                      f"{len(feasible)} feasible, chosen={chosen}")
            req = GraspRequest(target_id=chosen, timeout=30.0, place_layout=self.place_layout)
            is_push = (action[0] == "push")
            others_before = self._snapshot_positions([c for c in all_cube_ids
                                                      if c not in self._placed and c != chosen]
                                                     + [int(target_id)])
            if is_push:
                from production_robot_controller import PushRequest
                result = self.controller.execute_push_request(
                    PushRequest(target_id=chosen, dir_index=int(action[2]), timeout=30.0))
            else:
                result = self.controller.execute_grasp_request(req)
            ok = (result.status == GraspStatus.SUCCESS)
            disturbance = self._pile_disturbance(others_before)
            if disturbance["moved_over_1cm"]:
                self._log(f"  pile disturbance: {len(disturbance['moved_over_1cm'])} other cube(s) "
                          f"moved >1cm (max {disturbance['max_displacement_m']*1000:.0f}mm): "
                          f"{disturbance['moved_over_1cm']}")
            self.step_log.append(StepLog(step, "EXECUTE", {
                "target": chosen, "status": str(result.status),
                "action": (action_str_of(action) if action_str_of else f"pick({chosen})"),
                "failure_code": str(result.failure_code),
                "pile_disturbance": disturbance,
                "target_displacement_m": getattr(result, "target_displacement_m", None),
                "bilateral_contact_success": getattr(result, "bilateral_contact_success", None),
                "friction_micro_lift_success": getattr(result, "friction_micro_lift_success", None),
                "constraint_assisted_transport_success": getattr(result, "constraint_assisted_transport_success", None),
                "transitions": [
                    {"state": t.state.value, "ok": t.ok,
                     "failure_code": str(t.failure_code) if t.failure_code else None,
                     "message": t.message, "data": t.data}
                    for t in result.transitions
                ],
            }))

            if ok and is_push:
                successful += 1
                disp = float(getattr(result, "target_displacement_m", 0.0) or 0.0)
                self._push_history.setdefault(chosen, []).append((step, int(action[2]), disp))
                self._picks_since_push[chosen] = 0
                self.controller.ready()
            elif ok:
                successful += 1
                self._placed.add(chosen)
                for k_ in list(self._picks_since_push):
                    self._picks_since_push[k_] = self._picks_since_push.get(k_, 0) + 1
                for other in list(self._failure_counts):
                    if other != chosen and self._failure_counts[other] > 0:
                        self._failure_counts[other] -= 1
                self.controller.ready()
            elif is_push:
                failed += 1
                self._push_history.setdefault(chosen, []).append((step, int(action[2]), 0.0))
                self._picks_since_push[chosen] = 0
                self._log(f"  push failed ({result.failure_code}); that stroke is withheld until the pile changes")
                self.controller.ready()
            else:
                failed += 1
                self._failure_counts[chosen] = self._failure_counts.get(chosen, 0) + 1
                self._log(f"  grasp failed ({result.failure_code}); "
                          f"failure_count={self._failure_counts[chosen]}")
                self._mark_failed_gui(chosen)
                self.controller.ready()

            self._update_gui(step=step, exposure=current_exposure, feasible=feasible,
                             selected=chosen, scores=decision.all_scores,
                             status=f"{result.status} ({result.failure_code})")

            verify_state = self._observe(target_id, all_cube_ids)
            exposure_immediately_after = float(verify_state["human_exposure"])
            self.step_log.append(StepLog(step, "VERIFY", {
                "target": chosen, "action_success": ok,
                "exposure_before": current_exposure,
                "exposure_immediately_after": exposure_immediately_after,
                "exposure_delta": exposure_immediately_after - current_exposure,
                "retry_or_blacklist_action": (
                    "blacklisted" if (not ok and self._failure_counts.get(chosen, 0)
                                      >= self.max_failures_per_block)
                    else ("failure_recorded_not_yet_blacklisted" if not ok else "none")),
            }))

            for _ in range(SETTLE_STEPS):
                p.stepSimulation()
            self.step_log.append(StepLog(step, "SETTLE", {"steps": SETTLE_STEPS}))

            step += 1

        self.controller.home()

        final_state = self._observe(target_id, all_cube_ids)
        final_exposure = float(final_state["human_exposure"])
        rescue_success = final_exposure >= self.exposed_thresh

        return CoordinatorResult(
            seed=seed, policy=self.policy_name,
            checkpoint_meta=getattr(self.policy_agent, "checkpoint_meta", None),
            exposed_thresh=self.exposed_thresh,
            initial_exposure=initial_exposure, final_exposure=final_exposure,
            steps_completed=step, successful_placements=successful,
            failed_attempts=failed,
            blacklisted=sorted(b for b, n in self._failure_counts.items()
                               if n >= self.max_failures_per_block),
            rescue_success=rescue_success, termination_reason=termination_reason,
            step_log=[{"step": s.step, "state": s.state, **s.data} for s in self.step_log],
            execution_time=time.time() - t0,
        )
