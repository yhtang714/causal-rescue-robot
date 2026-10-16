#!/usr/bin/env python3
"""Runs one rescue on the arm and writes the demo video.

    python3 demo.py --pile aval_het --seed 101
    python3 demo.py --pile aval_hom50 --seed 100
    python3 demo.py --pile aval_het --seed 101 --no-video

The checked rule (ranker cat_v11, top-5 checked in the pile model, compliant
blade descent, success at exposure 0.95) runs until the target is exposed or
the action budget is spent. With --no-video the same run is done without
frames and only the step log is written.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def encode(frames, out, fps):
    files = sorted(glob.glob(os.path.join(frames, "f*.jpg")))
    if not files:
        sys.exit("no frames were written")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "f%05d.jpg"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return len(files)


def main():
    ap = argparse.ArgumentParser(description="rescue demo on the arm")
    ap.add_argument("--pile", default="aval_het", help="scene preset: aval_het (50 mixed pieces), aval_hom50 (50 cubes), ...")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--model", default="models/cat_v11/best.pt")
    ap.add_argument("--steps", type=int, default=24, help="action budget")
    ap.add_argument("--verify-k", type=int, default=5, help="ranker candidates checked in the pile model")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--no-video", action="store_true", help="headless run, step log only")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    tag = f"{a.pile}_seed{a.seed}"
    frames = os.path.join(out, f"frames_{tag}")
    env = dict(os.environ)
    env.pop("PILE_CACHE", None)
    env.update({
        "DEMO_PILE": a.pile, "DEMO_SEED": str(a.seed), "DEMO_POLICY": "cat",
        "DEMO_MODEL": os.path.join(ROOT, a.model), "DEMO_STEPS": str(a.steps),
        "DEMO_ACTIONS": "pick_push", "DEMO_PUSH_TOP": "4", "DEMO_VERIFY_K": str(a.verify_k),
        "DEMO_FRAMES": frames, "DEMO_ROWS": os.path.join(out, f"audit_{tag}.json"),
        "DEMO_RUN_JSON": os.path.join(out, f"run_{tag}.json"), "DEMO_META": os.path.join(out, f"meta_{tag}.json"),
        "DEMO_RENDER": "0" if a.no_video else "1",
        "DEMO_W": str(a.width), "DEMO_H": str(a.height),
        "DEMO_RENDER_W": str(a.width * 3 // 2), "DEMO_RENDER_H": str(a.height * 3 // 2),
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "1"),
    })
    log = os.path.join(out, f"log_{tag}.txt")
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "robot_demo", "record.py")], env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"run failed, see {log}")
    run = json.load(open(os.path.join(out, f"run_{tag}.json")))
    print(f"{a.pile} seed {a.seed}: exposure {run['initial_exposure']:.3f} -> {run['final_exposure']:.3f}, "
          f"{run['steps_completed']} actions, {run['successful_placements']} succeeded, {run['failed_attempts']} failed, "
          f"{run['termination_reason']}")
    for s in run["step_log"]:
        if s["state"] == "VERIFY":
            print(f"  action {s['step'] + 1}: {'ok' if s['action_success'] else 'failed'}  "
                  f"exposure {s['exposure_before']:.3f} -> {s['exposure_immediately_after']:.3f}")
    if a.no_video:
        shutil.rmtree(frames, ignore_errors=True)
        return
    video = os.path.join(out, f"demo_{tag}.mp4")
    n = encode(frames, video, a.fps)
    print(f"{video}: {n} frames, {n / a.fps:.1f} s")
    if not a.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
