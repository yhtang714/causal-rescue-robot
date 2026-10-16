"""Recomputes the training target with a chosen weight on the continuation term."""
import argparse
import pandas as pd

LAYER, UNLOCK, CLEAR, OPTION, GAMMA = 0.5, 0.5, 0.25, 0.15, 0.7


def value(d, prefix, promise):
    return (d[f"{prefix}delta_E"] + LAYER * d[f"{prefix}delta_layers"]
            + promise * (UNLOCK * d[f"{prefix}unlock_L"] + CLEAR * d[f"{prefix}delta_clear"] + OPTION * d[f"{prefix}delta_options"])
            + d[f"{prefix}deadend"] + d[f"{prefix}done"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--promise", type=float, default=1.0, help="multiplier on the unlock/clear/options terms (1 = weights)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    v1 = value(d, "cf_", a.promise)
    v2c = value(d, "cf2_", a.promise)
    d["cf_value"] = v1
    d["cf_value2"] = v1 + GAMMA * v2c.clip(lower=0.0) * d["cf2_expanded"]
    d.to_csv(a.out, index=False)
    g = d.groupby(["scene_id", "step_index"])
    flat = (g["cf_value2"].max() <= 0.01).mean()
    pushopt = g.apply(lambda x: x.loc[x["cf_value2"].idxmax(), "action_type"] == "push").mean()
    print(f"promise x{a.promise}: {len(d)} rows, {g.ngroups} states, flat {flat:.1%}, optimum is a push in {pushopt:.0%} -> {a.out}")


if __name__ == "__main__":
    main()
