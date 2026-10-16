"""Filters a labelled dataset and trains the ranker."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ONLINE_FEATURES = [
    "A_cube_x", "A_cube_y", "A_cube_z",
    "A_dx", "A_dy", "A_dz", "A_d", "A_d_xy",
    "A_exposure_proxy", "A_occluder_share", "A_indirect_occluder",
    "A_height_norm", "A_num_neighbors", "A_num_below", "A_num_above",
    "A_is_top", "A_support_depth", "A_is_graspable",
    "A_region_over_hammer", "A_region_xy_near",
]
GROUP_COLS = ["episode_id", "scene_id", "step_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--save_dir", default="models/cat_new")
    ap.add_argument("--target_col", default="cf_delta_burden")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min-spread", type=float, default=0.0,
                    help="drop decision groups whose target spread (max-min) is below this: a group in "
                         "which every candidate has the same effect carries no ranking signal")
    ap.add_argument("--access", action="store_true",
                    help="CAT v2: also train on the access-aware features (causal_core.access)")
    ap.add_argument("--causal-mode", default=None, choices=["none", "scaling", "bias", "heads", "all"],
                    help="causal attention mode passed to train_optimal_policy (default: its default, scaling)")
    ap.add_argument("--no-causal", action="store_true", help="ablation: no causal scaling/attention at all")
    ap.add_argument("--mediators", default="", help="label columns to predict with the auxiliary mediator head")
    ap.add_argument("--mediator-weight", type=float, default=0.3)
    a = ap.parse_args()
    feature_names = list(ONLINE_FEATURES)
    if a.access:
        from causal_core.access import ACCESS_FEATURE_NAMES
        feature_names += list(ACCESS_FEATURE_NAMES)

    df = pd.read_csv(a.csv)
    n_all = len(df)
    labelled = df[df[a.target_col].notna()].copy()
    sizes = labelled.groupby(GROUP_COLS).size()
    spread = labelled.groupby(GROUP_COLS)[a.target_col].agg(lambda v: float(v.max() - v.min()))
    keep = sizes[(sizes >= 2) & (spread >= a.min_spread)].index
    labelled = labelled.set_index(GROUP_COLS).loc[keep].reset_index()
    med_cols = [c.strip() for c in a.mediators.split(",") if c.strip() and c.strip() in labelled.columns]
    cols = GROUP_COLS + ["candidate_id", "is_chosen"] + feature_names + [a.target_col] + \
        [c for c in med_cols if c != a.target_col]
    filtered = labelled[cols]
    os.makedirs(a.save_dir, exist_ok=True)
    out_csv = os.path.join(a.save_dir, "train_filtered.csv")
    filtered.to_csv(out_csv, index=False)
    print(f"[train] rows {n_all} -> labelled {len(labelled)} in {len(keep)} decision groups; "
          f"features={len(feature_names)}; target={a.target_col}")
    nz = (filtered.groupby(GROUP_COLS)[a.target_col].max() > 0).mean()
    print(f"[train] groups with at least one positive-effect candidate: {nz:.1%}")

    extra = []
    for tok in a.extra:
        extra += tok.split()
    a.extra = extra
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "trainer.py"),
           "--csv", out_csv, "--save_dir", a.save_dir, "--target_col", a.target_col,
           "--epochs", str(a.epochs), "--tie_break", "any"] + \
          (["--mediators", ",".join(med_cols), "--mediator_weight", str(a.mediator_weight)] if med_cols else []) + a.extra
    if a.causal_mode:
        cmd += ["--causal_mode", a.causal_mode]
    if a.no_causal:
        cmd += ["--no_causal", "--causal_mode", "none"]
    print("[train]", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
