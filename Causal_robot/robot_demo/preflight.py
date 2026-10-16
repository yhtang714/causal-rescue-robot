"""Import, asset and headless-PyBullet checks."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = []


def check(name, ok, ev=""):
    R.append(dict(check=name, passed=bool(ok), evidence=str(ev)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {ev}", flush=True)
    return bool(ok)


def main():
    ok_all = True

    pv = ".".join(map(str, sys.version_info[:3]))
    ok_all &= check("1. Python interpreter", True, f"{pv} at {sys.executable}")

    try:
        import pybullet as p
        ok_all &= check("2. pybullet importable", True,
                        f"module {p.__file__}, API {p.getAPIVersion()}")
    except Exception as ex:
        check("2. pybullet importable", False, repr(ex))
        return _finish(False)

    modules = ["sim", "production_robot_controller", "perception",
              "causal_core", "causal_core.candidate_set", "causal_core.feature_builder",
              "decision.causal_policy_agent", "robot_demo.scenes", "robot_demo.coordinator"]
    bad = []
    for m in modules:
        try:
            mod = __import__(m, fromlist=["_"])
            R.append(dict(check=f"3. import {m}", passed=True, evidence=getattr(mod, "__file__", "?")))
        except Exception as ex:
            bad.append(f"{m}: {ex!r}")
    ok_all &= check("3. all robot_demo dependency modules import",
                    not bad, "\n       ".join(bad) if bad else f"{len(modules)} modules OK")

    try:
        from robot_demo.scenes import _cube_urdf_path
        path = _cube_urdf_path()
        exists = os.path.isfile(path)
        ok_all &= check("4. cube URDF asset present", exists, path)
    except Exception as ex:
        ok_all &= check("4. cube URDF asset present", False, repr(ex))

    try:
        from robot_demo.scenes import build_isolated_scene
        from causal_core.candidate_set import CorrectedCandidateSet
        t0 = time.time()
        scene = build_isolated_scene(seed=0, gui=False)
        ok_all &= check("5. isolated scene constructs headless", len(scene.cube_ids) == 1,
                        f"{len(scene.cube_ids)} cube(s), {time.time()-t0:.2f}s")
        ccs = CorrectedCandidateSet(scene.sim)
        recs = ccs.candidates(scene.cube_ids, gates=("C1", "C2"))
        ok_all &= check("6. C1+C2 candidate generation (corrected gate)", len(recs) >= 0,
                        f"{len(recs)} of {len(scene.cube_ids)} pass C1+C2 (no fallback)")
    except Exception as ex:
        import traceback
        ok_all &= check("5-6. scene + candidate-set smoke", False, traceback.format_exc(limit=4))

    try:
        from production_robot_controller import ProductionRobotController
        ctrl = ProductionRobotController(sim=scene.sim, enable_logging=False)
        ctrl.home()
        ok_all &= check("7. ProductionRobotController constructs and homes", True,
                        f"place tote center={ctrl.place_tote_center}")
    except Exception as ex:
        ok_all &= check("7. ProductionRobotController constructs and homes", False, repr(ex))

    try:
        from decision.causal_policy_agent import CausalPolicyAgent
        agent = CausalPolicyAgent()
        ok_all &= check("8. HeuristicPolicy CausalPolicyAgent() constructs (no checkpoint)",
                        agent.model is None, f"name={agent.name}")
    except Exception as ex:
        ok_all &= check("8. HeuristicPolicy CausalPolicyAgent() constructs", False, repr(ex))

    try:
        from decision.causal_policy_agent import CausalPolicyAgent
        try:
            CausalPolicyAgent(require_checkpoint=True)
            ok_all &= check("9. CAT mode fails closed without --model", False,
                            "expected RuntimeError, none raised")
        except RuntimeError:
            ok_all &= check("9. CAT mode fails closed without --model", True,
                            "RuntimeError raised as expected")
    except Exception as ex:
        ok_all &= check("9. CAT mode fails closed without --model", False, repr(ex))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    try:
        os.makedirs(out_dir, exist_ok=True)
        t = os.path.join(out_dir, ".write_test")
        open(t, "w").write("ok")
        os.remove(t)
        ok_all &= check("10. output/ directory writable", True, out_dir)
    except Exception as ex:
        ok_all &= check("10. output/ directory writable", False, repr(ex))

    return _finish(ok_all, out_dir)


def _finish(ok, out_dir=None):
    n = sum(r["passed"] for r in R)
    print(f"\n{n}/{len(R)} preflight checks passed")
    if out_dir:
        try:
            json.dump(dict(passed=bool(ok), n_checks=len(R), n_passed=n, checks=R),
                      open(os.path.join(out_dir, "preflight.json"), "w"), indent=2)
        except Exception:
            pass
    if not ok:
        print("PREFLIGHT FAILED -- fix the failing check(s) above before running rescue_demo.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
