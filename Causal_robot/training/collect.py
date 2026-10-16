"""Collects decision states with interventional labels: every legal action of every visited state is applied in the pile model."""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p

ONLINE_FEATURES = [
    "A_cube_x", "A_cube_y", "A_cube_z",
    "A_dx", "A_dy", "A_dz", "A_d", "A_d_xy",
    "A_exposure_proxy", "A_occluder_share", "A_indirect_occluder",
    "A_height_norm", "A_num_neighbors", "A_num_below", "A_num_above",
    "A_is_top", "A_support_depth", "A_is_graspable",
    "A_region_over_hammer", "A_region_xy_near",
]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out

LABELS = ["cf_delta_E", "cf_occ_share_removed", "cf_unlock", "cf_delta_layers", "cf_unlock_L",
          "cf_delta_clear", "cf_delta_options", "cf_deadend", "cf_done", "cf_value", "cf_value2",
          "cf2_expanded", "cf2_delta_E", "cf2_delta_layers", "cf2_unlock_L", "cf2_delta_clear",
          "cf2_delta_options", "cf2_deadend", "cf2_done", "cf2_is_push",
          "stuck_state"]
META = ["episode_id", "scene_id", "step_index", "candidate_id", "is_chosen", "exposure_before",
        "burial_before", "n_feasible", "n_actions", "action_type", "action_cid", "action_dir"]


def build_actions(coord, state, args, controller):
    from causal_core.actions import PUSH_DIRS, push_candidates
    from causal_core.lookahead import relevance_scores
    from production_robot_controller import PushRequest
    feas, _diag = coord._feasible_set(state)
    n_opt = int(_diag.get("n_options", len(feas))) if isinstance(_diag, dict) else len(feas)
    feas = sorted(int(c) for c in feas)
    actions = [("pick", int(c)) for c in feas]
    pc = push_candidates(coord, state)
    pc = [c for c in pc if int(c) not in set(feas)]
    if pc and args.push_top > 0:
        alive = [int(c) for c in state.get("cube_ids", []) if int(c) not in coord._placed]
        rel = relevance_scores(coord.candidate_set._gate, state, pc, alive)
        clear_flags = {}
        for cid in sorted(pc, key=lambda c: -rel[int(c)])[:args.push_top]:
            for d in range(len(PUSH_DIRS)):
                if coord._push_memory_veto(int(cid), d):
                    clear_flags[(int(cid), d)] = False
                    continue
                ok, c7info = controller.push_descent_clear(PushRequest(target_id=int(cid), dir_index=d))
                clear_flags[(int(cid), d)] = bool(c7info.get("column_free", ok)) if isinstance(c7info, dict) else bool(ok)
                if ok:
                    actions.append(("push", int(cid), d))
        return feas, actions, clear_flags, n_opt
    return feas, actions, {}, n_opt


def run_episode(seed, args, writer, log):
    from robot_demo.scenes import build_rescue_scene
    from robot_demo.coordinator import RescueCoordinator
    from causal_core.access import (ACCESS_FEATURE_NAMES, access_features, cover_clearance,
                                    cover_share_of, occluders_of, value_label)
    from causal_core.actions import action_access, action_str, apply_action, undo_bookkeeping
    from decision.causal_policy_agent import _build_graph_context
    rng = np.random.default_rng(seed)
    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass
    try:
        scene = build_rescue_scene(seed=seed, gui=False, place_layout="opposite", pile=args.pile)
    except RuntimeError as e:
        log(f"seed {seed}: scene rejected ({str(e)[:80]}) -- skipped")
        return 0
    sim = scene.sim
    coord = RescueCoordinator(sim=sim, policy="heuristic", enable_logging=False, place_layout="opposite")
    coord.controller.ready()
    controller = coord.controller
    gate = coord.candidate_set._gate
    target = int(scene.target_id)
    all_cubes = [int(c) for c in sim.get_ids()["cubes"]]
    step_fn = lambda n: sim.step_simulation(int(n))
    n_rows = 0
    t0 = time.time()
    for step in range(args.max_steps):
        state = coord._observe(target, all_cubes)
        exposure = float(state["human_exposure"])
        if exposure >= coord.exposed_thresh:
            log(f"seed {seed}: exposure {exposure:.3f} >= threshold at step {step} -- done")
            break
        feas, actions, clear_flags, _n_reach = build_actions(coord, state, args, coord.controller)
        n_opt_b = len(actions)
        if not actions:
            log(f"seed {seed}: no action at step {step} -- done")
            break
        occ = occluders_of(state)
        ref_share = cover_share_of(state)
        alive = [c for c in all_cubes if c not in coord._placed]
        base_access = access_features(gate, sorted({int(a[1]) for a in actions}), occ, alive=alive, state=state)
        access_by_action = {i: action_access(coord, state, a, base_access, target_id=target,
                                             blade_clear=clear_flags.get((int(a[1]), a[2] if a[0] == "push" else -1),
                                                                         True))
                            for i, a in enumerate(actions)}
        ctx = _build_graph_context(state)
        graspable_set = set(int(x) for x in state.get("graspable_blocks", feas))
        labels = {}
        for k, act in enumerate(actions):
            clear_b = cover_clearance(gate, ref_share, exclude=[act[1]])
            sid = p.saveState()
            try:
                apply_action(coord, sim, act, step_fn, k=k)
                sim.step_simulation(int(args.cf_settle))
                st2 = coord._observe(target, all_cubes)
                f2, _d2 = coord._feasible_set(st2)
                f2 = sorted(int(c) for c in f2)
                n_opt_a = coord._count_actions_fast(st2, f2, all_cubes)
                removed = int(act[1]) if act[0] == "pick" else -1
                labels[k] = value_label(exposure, float(st2["human_exposure"]), removed, occ, feas,
                                        occluders_of(st2), f2, exposed_thresh=coord.exposed_thresh,
                                        state_before=state, state_after=st2,
                                        clear_before=clear_b, clear_after=cover_clearance(gate, ref_share, exclude=[act[1]]),
                                        n_options_before=n_opt_b,
                                        n_options_after=n_opt_a)
            finally:
                undo_bookkeeping(coord, act)
                p.restoreState(sid)
                p.removeState(sid)
        for k in labels:
            labels[k]["cf_value2"] = labels[k]["cf_value"]
            labels[k].update(cf2_expanded=0, cf2_delta_E=0.0, cf2_delta_layers=0.0, cf2_unlock_L=0.0,
                             cf2_delta_clear=0.0, cf2_delta_options=0.0, cf2_deadend=0.0, cf2_done=0.0, cf2_is_push=0)
        order = sorted(range(len(actions)), key=lambda k: -labels[k]["cf_value"])
        _picks = [k for k in order if actions[k][0] == "pick"]
        _pushes = [k for k in order if actions[k][0] == "push"]
        expand, _i, _j = [], 0, 0
        while len(expand) < args.lookahead_top and (_i < len(_picks) or _j < len(_pushes)):
            if _i < len(_picks):
                expand.append(_picks[_i]); _i += 1
            if len(expand) < args.lookahead_top and _j < len(_pushes):
                expand.append(_pushes[_j]); _j += 1
        for k in expand:
            sid = p.saveState()
            try:
                apply_action(coord, sim, actions[k], step_fn, k=k)
                sim.step_simulation(int(args.cf_settle))
                st2 = coord._observe(target, all_cubes)
                e2 = float(st2["human_exposure"])
                feas2, act2, _cf, _ = build_actions(coord, st2, args, coord.controller)
                n_opt2 = len(act2)
                occ2 = occluders_of(st2)
                ref2 = cover_share_of(st2)
                _p2 = [x for x in act2 if x[0] == "pick"]
                _u2 = [x for x in act2 if x[0] == "push"]
                shortlist, _a, _b = [], 0, 0
                while len(shortlist) < args.cf2_width and (_a < len(_p2) or _b < len(_u2)):
                    if _a < len(_p2):
                        shortlist.append(_p2[_a]); _a += 1
                    if len(shortlist) < args.cf2_width and _b < len(_u2):
                        shortlist.append(_u2[_b]); _b += 1
                best2 = 0.0
                best2_lab, best2_act = None, None
                for k2, a2 in enumerate(shortlist):
                    cb2 = cover_clearance(gate, ref2, exclude=[a2[1]])
                    sid2 = p.saveState()
                    try:
                        apply_action(coord, sim, a2, step_fn, k=k2 + 20)
                        sim.step_simulation(int(args.cf_settle2))
                        st3 = coord._observe(target, all_cubes)
                        f3, _d3 = coord._feasible_set(st3)
                        f3 = sorted(int(c) for c in f3)
                        n_opt3 = coord._count_actions_fast(st3, f3, all_cubes)
                        rem2 = int(a2[1]) if a2[0] == "pick" else -1
                        v2 = value_label(e2, float(st3["human_exposure"]), rem2, occ2, feas2,
                                         occluders_of(st3), f3, exposed_thresh=coord.exposed_thresh,
                                         state_before=st2, state_after=st3,
                                         clear_before=cb2,
                                         clear_after=cover_clearance(gate, ref2, exclude=[a2[1]]),
                                         n_options_before=n_opt2,
                                         n_options_after=n_opt3)
                        if v2["cf_value"] > best2:
                            best2, best2_lab, best2_act = v2["cf_value"], v2, a2
                    finally:
                        undo_bookkeeping(coord, a2)
                        p.restoreState(sid2)
                        p.removeState(sid2)
                labels[k]["cf_value2"] = labels[k]["cf_value"] + args.gamma * best2
                labels[k]["cf2_expanded"] = 1
                if best2_lab is not None:
                    labels[k].update(cf2_delta_E=best2_lab["cf_delta_E"], cf2_delta_layers=best2_lab["cf_delta_layers"],
                                     cf2_unlock_L=best2_lab["cf_unlock_L"], cf2_delta_clear=best2_lab["cf_delta_clear"],
                                     cf2_delta_options=best2_lab["cf_delta_options"], cf2_deadend=best2_lab["cf_deadend"],
                                     cf2_done=best2_lab["cf_done"], cf2_is_push=int(best2_act[0] == "push"))
            finally:
                undo_bookkeeping(coord, actions[k])
                p.restoreState(sid)
                p.removeState(sid)
        stuck = int(max(labels[k]["cf_value2"] for k in labels) <= 1e-6)
        if rng.random() < args.p_best:
            best = max(labels[k]["cf_value2"] for k in labels)
            pool = [k for k in labels if labels[k]["cf_value2"] >= best - 1e-9]
        else:
            pool = list(range(len(actions)))
        chosen = int(rng.choice(pool))
        for i, act in enumerate(actions):
            feats = ctx.features_for(int(act[1]), occluder_share=state.get("occluder_share", {}),
                                     indirect_occluder=state.get("indirect_occluder_contribution", {}),
                                     chain=[], graspable_ids=graspable_set, access=access_by_action[i])
            row = {"episode_id": seed + args.episode_offset, "scene_id": seed, "step_index": step,
                   "candidate_id": i, "is_chosen": int(i == chosen), "exposure_before": exposure,
                   "burial_before": float(state.get("burial_depth_mean", 0.0)),
                   "n_feasible": len(feas), "n_actions": len(actions),
                   "action_type": act[0], "action_cid": int(act[1]),
                   "action_dir": int(act[2]) if act[0] == "push" else -1}
            for n in ONLINE_FEATURES + ACCESS_FEATURE_NAMES:
                row[n] = float(feats[n])
            row.update({k2: v2 for k2, v2 in labels[i].items()})
            row["stuck_state"] = stuck
            writer.writerow(row)
            n_rows += 1
        log(f"seed {seed} step {step:2d}: E={exposure:.3f} picks={len(feas)} actions={len(actions)} "
            f"best={max(labels[k]['cf_value2'] for k in labels):+.3f} chosen={action_str(actions[chosen])} "
            f"({time.time()-t0:.0f}s)")
        act = actions[chosen]
        if act[0] == "push":
            from causal_core.actions import apply_push
            disp = apply_push(sim, int(act[1]), int(act[2]), step_fn)
            sim.step_simulation(int(args.settle))
            coord._push_history.setdefault(int(act[1]), []).append((step, int(act[2]), abs(float(disp))))
            coord._picks_since_push[int(act[1])] = 0
        else:
            apply_action(coord, sim, act, step_fn, k=step, settle=int(args.settle))
            for k_ in list(coord._picks_since_push):
                coord._picks_since_push[k_] = coord._picks_since_push.get(k_, 0) + 1
    return n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--pile", default="aval_het")
    ap.add_argument("--out", default="datasets/labels.csv")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--settle", type=int, default=400)
    ap.add_argument("--cf-settle", type=int, default=200)
    ap.add_argument("--cf-settle2", type=int, default=160)
    ap.add_argument("--p-best", type=float, default=0.7)
    ap.add_argument("--push-top", type=int, default=4)
    ap.add_argument("--lookahead-top", type=int, default=2)
    ap.add_argument("--cf2-width", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    from causal_core.access import ACCESS_FEATURE_NAMES
    fields = META + ONLINE_FEATURES + ACCESS_FEATURE_NAMES + LABELS
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    done_seeds = set()
    if args.skip_done and not new_file:
        with open(args.out) as f0:
            for r in csv.DictReader(f0):
                done_seeds.add(int(r["scene_id"]))
    total = 0
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        def log(msg):
            print(f"[v6] {msg}", flush=True)
        for seed in parse_seeds(args.seeds):
            if seed in done_seeds:
                log(f"seed {seed}: already in {args.out} -- skipped")
                continue
            n = run_episode(seed, args, writer, log)
            f.flush()
            total += n
            log(f"seed {seed}: {n} action rows (total {total})")
    print(f"[v6] DONE: {total} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
