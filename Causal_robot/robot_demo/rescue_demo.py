"""Command line for an arm run without video and for the grasp-only stages."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pybullet as p


def _reports_dir():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(d, exist_ok=True)
    return d


def _start_recording(path):
    try:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path)
        return log_id
    except Exception as ex:
        print(f"[rescue_demo] WARNING: could not start video recording ({ex!r}); "
              f"continuing without a recording. This PyBullet build/GUI backend "
              f"may not support STATE_LOGGING_VIDEO_MP4 headlessly.")
        return None


def run_isolated(args):
    from robot_demo.scenes import build_isolated_scene
    from production_robot_controller import ProductionRobotController, GraspRequest

    n_trials = args.trials or 10
    results = []
    for i in range(n_trials):
        seed = args.seed + i
        scene = build_isolated_scene(seed=seed, gui=args.gui)
        ctrl = ProductionRobotController(sim=scene.sim, enable_logging=args.verbose)
        ctrl.home()
        rec_id = _start_recording(args.record) if (args.record and i == 0 and args.gui) else None
        t0 = time.time()
        req = GraspRequest(target_id=scene.cube_ids[0], timeout=30.0)
        result = ctrl.execute_grasp_request(req)
        if rec_id is not None:
            try:
                p.stopStateLogging(rec_id)
            except Exception:
                pass
        results.append(dict(
            seed=seed, status=str(result.status), failure_code=str(result.failure_code),
            elapsed=round(time.time() - t0, 3),
            bilateral_contact_success=result.bilateral_contact_success,
            friction_micro_lift_success=result.friction_micro_lift_success,
            constraint_assisted_transport_success=result.constraint_assisted_transport_success,
            message=result.error_message,
            transitions=[dict(state=t.state.value, ok=t.ok,
                              failure_code=str(t.failure_code) if t.failure_code else None,
                              message=t.message) for t in result.transitions],
        ))
        try:
            p.disconnect()
        except Exception:
            pass

    n_success = sum(1 for r in results if r["status"] == "GraspStatus.SUCCESS")
    target_n_success = max(1, round(0.9 * n_trials))
    out = dict(
        config_version=1, mode="isolated", policy="n/a (manipulation-only)",
        base_seed=args.seed, n_trials=n_trials, n_success=n_success,
        target_n_success=target_n_success,
        meets_target=n_success >= target_n_success,
        trials=results,
    )
    return out


def run_stack(args):
    from robot_demo.scenes import build_stack_scene
    from production_robot_controller import ProductionRobotController, GraspRequest

    n = args.trials or 3
    scene = build_stack_scene(seed=args.seed, gui=args.gui, n=n)
    ctrl = ProductionRobotController(sim=scene.sim, enable_logging=args.verbose)
    ctrl.home()
    rec_id = _start_recording(args.record) if (args.record and args.gui) else None

    placed = set()
    rounds = []
    for i in range(n):
        best_id, best_z = None, -1e9
        for cid in scene.cube_ids:
            if cid in placed:
                continue
            try:
                z = p.getBasePositionAndOrientation(cid)[0][2]
            except Exception:
                continue
            if z > best_z:
                best_z, best_id = z, cid
        if best_id is None:
            break
        t0 = time.time()
        req = GraspRequest(target_id=best_id, timeout=30.0)
        result = ctrl.execute_grasp_request(req)
        ok = str(result.status) == "GraspStatus.SUCCESS"
        if ok:
            placed.add(best_id)
        rounds.append(dict(
            round=i, target_id=best_id, status=str(result.status),
            failure_code=str(result.failure_code), elapsed=round(time.time() - t0, 3),
            bilateral_contact_success=result.bilateral_contact_success,
            friction_micro_lift_success=result.friction_micro_lift_success,
            constraint_assisted_transport_success=result.constraint_assisted_transport_success,
            message=result.error_message,
        ))
        ctrl.home()
        if not ok:
            break

    if rec_id is not None:
        try:
            p.stopStateLogging(rec_id)
        except Exception:
            pass
    try:
        p.disconnect()
    except Exception:
        pass

    out = dict(
        config_version=1, mode="stack", policy="n/a (manipulation-only)",
        seed=args.seed, n_requested=n, n_consecutive_success=len(placed),
        meets_target=len(placed) >= min(3, n), rounds=rounds,
    )
    return out


def run_rescue(args):
    from robot_demo.scenes import build_rescue_scene
    from robot_demo.coordinator import RescueCoordinator, DEFAULT_EXPOSED_THRESH

    thresh = args.exposed_thresh if args.exposed_thresh is not None else DEFAULT_EXPOSED_THRESH
    scene = build_rescue_scene(seed=args.seed, gui=args.gui, place_layout=args.place_layout, pile=args.pile)
    rec_id = _start_recording(args.record) if (args.record and args.gui) else None

    coord = RescueCoordinator(
        sim=scene.sim, policy=args.policy, model_path=args.model,
        exposed_thresh=thresh, max_steps=args.max_steps, enable_logging=args.verbose,
        place_layout=args.place_layout, nudge_mm=args.nudge_mm, lookahead=args.lookahead,
        actions=args.actions, push_top=args.push_top, verify_k=args.verify_k,
    )
    result = coord.run(target_id=scene.target_id, seed=args.seed)

    if rec_id is not None:
        try:
            p.stopStateLogging(rec_id)
        except Exception:
            pass
    try:
        p.disconnect()
    except Exception:
        pass

    out = dict(
        config_version=1, mode="rescue",
        policy=result.policy, checkpoint_meta=result.checkpoint_meta,
        seed=result.seed, exposed_thresh=result.exposed_thresh,
        post_action_settle_steps=400,
        nudge_mm=float(args.nudge_mm), pile=args.pile, lookahead=int(args.lookahead),
        pile_settle=getattr(scene.sim, "_settle_report", None),
        initial_exposure=result.initial_exposure, final_exposure=result.final_exposure,
        steps_completed=result.steps_completed,
        successful_placements=result.successful_placements,
        failed_attempts=result.failed_attempts,
        blacklisted_cubes=result.blacklisted,
        rescue_success=result.rescue_success,
        termination_reason=result.termination_reason,
        execution_time=round(result.execution_time, 2),
        step_log=result.step_log,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Closed-loop rescue run on the arm")
    ap.add_argument("--mode", choices=["isolated", "stack", "rescue"], required=True)
    ap.add_argument("--policy", choices=["heuristic", "cat"], default="heuristic",
                    help="rescue mode: 'cat' needs --model")
    ap.add_argument("--model", type=str, default=None, help="CAT checkpoint path (.pt)")
    ap.add_argument("--headless", dest="gui", action="store_false", default=False,
                    help="(default) run without a GUI window")
    ap.add_argument("--gui", dest="gui", action="store_true", help="open a PyBullet GUI window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exposed-thresh", type=float, default=None,
                    help="Absolute exposure threshold for rescue success (default 0.95)")
    ap.add_argument("--trials", type=int, default=None,
                    help="isolated: number of independent trials (default 10). "
                         "stack: stack height / number of consecutive transfers (default 3).")
    ap.add_argument("--max-steps", type=int, default=20, help="rescue mode: max coordinator steps")
    ap.add_argument("--place-layout", choices=["opposite", "same-side"], default="opposite")
    ap.add_argument("--pile", default="aval_het",
                    help="rescue mode: pile preset (robot_demo.scenes.PILE_PRESETS)")
    ap.add_argument("--actions", default="pick", choices=["pick", "pick_push"],
                    help="decision-layer action space (pick only, or picks plus blade pushes)")
    ap.add_argument("--push-top", type=int, default=4,
                    help="pushes are offered for the K most relevant reachable pieces")
    ap.add_argument("--verify-k", type=int, default=0,
                    help="verify the ranker's top-K actions with the live counterfactual and veto traps (0 = raw ranker)")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="pick-only mode: candidates checked in the pile model (0 = plain argmax)")
    ap.add_argument("--nudge-mm", type=float, default=0.0,
                    help="allowed intrusion of the fingers into a neighbour on the way down, mm (0 = none)")
    ap.add_argument("--record", type=str, default=None, help="Optional .mp4 path (requires --gui)")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    ap.add_argument("--out", type=str, default=None, help="output JSON path (default under output/)")
    args = ap.parse_args()

    if args.policy == "cat" and not args.model and args.mode == "rescue":
        ap.error("--policy cat requires --model PATH")
    if args.record and not args.gui:
        print("[rescue_demo] --record needs --gui; continuing without a recording")

    t0 = time.time()
    if args.mode == "isolated":
        result = run_isolated(args)
    elif args.mode == "stack":
        result = run_stack(args)
    else:
        result = run_rescue(args)
    result["wall_time_s"] = round(time.time() - t0, 2)

    out_path = args.out or os.path.join(_reports_dir(), f"rescue_demo_{args.mode}_seed{args.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[rescue_demo] mode={args.mode} seed={args.seed} -> {out_path}")
    if args.mode == "isolated":
        print(f"[rescue_demo] Stage A: {result['n_success']}/{result['n_trials']} "
              f"succeeded (target {result['target_n_success']}) "
              f"-> meets_target={result['meets_target']}")
    elif args.mode == "stack":
        print(f"[rescue_demo] Stage B: {result['n_consecutive_success']}/{result['n_requested']} "
              f"consecutive transfers -> meets_target={result['meets_target']}")
    else:
        print(f"[rescue_demo] Stage C/D ({result['policy']}): exposure "
              f"{result['initial_exposure']:.3f} -> {result['final_exposure']:.3f} "
              f"(threshold {result['exposed_thresh']}), "
              f"rescue_success={result['rescue_success']}, "
              f"termination={result['termination_reason']}")

    sys.exit(0)


if __name__ == "__main__":
    main()
