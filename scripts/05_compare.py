"""Put the equation types side by side and test the difference between them.

The headline number is a difference of two gaps:

    (Black missed - White missed) under the race-adjusted equation
  - (Black missed - White missed) under the race-neutral equation

Both are measured on the same children, by the same model, at the same flag
rate. So the one thing that can move it is the equation. Because it is the
same children twice, the interval comes from resampling children once and
recomputing both gaps on that resample.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src import data as D, evaluate as E, features as F, models as M


def load_oof(equation_type, model, expect_n=None):
    path = C.MODELS / f"oof_{equation_type}_{model}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run:\n"
            f"  python scripts/04_train.py --equation-type {equation_type}")
    p = np.load(path)
    if expect_n is not None and len(p) != expect_n:
        raise ValueError(
            f"{path.name} holds {len(p)} predictions but the analysis table has "
            f"{expect_n} children. Re-run 02 and then 04 so they line up.")
    return p


def paired_bootstrap(y, p_left, p_right, groups,
                     n=C.N_BOOTSTRAP, seed=C.SEED):
    """Interval for the difference of gaps, resampling children once for both."""
    rng = np.random.default_rng(seed)
    a, b = C.PRIMARY_GROUPS
    idx = np.arange(len(y))
    diffs = []

    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys, gs = y[s], groups[s]
        if ys[gs == a].sum() == 0 or ys[gs == b].sum() == 0:
            continue
        gl = E.fnr_gap(ys, p_left[s], gs, E.threshold_for_rate(p_left[s]))
        gr = E.fnr_gap(ys, p_right[s], gs, E.threshold_for_rate(p_right[s]))
        diffs.append(gl - gr)

    diffs = np.array(diffs)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float((diffs <= 0).mean()))


def fnr_by_race(y, p, groups):
    thr = E.threshold_for_rate(p)
    t = E.group_table(y, p, groups, thr)
    return t.set_index("group")["fnr"]


def make_figure(y, groups, model, path):
    """Missed-case rate by race, one group of bars per equation type."""
    races = [r for r in C.PRIMARY_GROUPS if r in set(groups)]
    types = [t for t in C.EQUATION_TYPES
             if (C.MODELS / f"oof_{t}_{model}.npy").exists()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(races), 1)
    x = np.arange(len(types))

    for i, race in enumerate(races):
        vals = [fnr_by_race(y, load_oof(t, model), groups)[race] for t in types]
        ax.bar(x + i * width, vals, width, label=race)
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)

    ax.set_xticks(x + width * (len(races) - 1) / 2)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("children with asthma who were missed")
    ax.set_title(f"Missed cases by race, at a matched {C.FLAG_RATE:.0%} flag rate"
                 f"\n({model} model, same children throughout)")
    ax.legend(title="race")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def main():
    C.ensure_dirs()
    df = D.load_analysis()

    # the rows are identical across equation types, so any one gives y and race
    _, y, groups = F.build(df, equation_type="neutral")
    y, groups = y.to_numpy(int), groups.to_numpy()

    summaries = sorted(C.TABLES.glob("04_summary_*.csv"))
    if not summaries:
        raise SystemExit("no results yet. Run scripts/04_train.py first.")
    allsum = pd.concat([pd.read_csv(f) for f in summaries], ignore_index=True)

    print("\n=== every run ======================================================")
    print(allsum[["equation_type", "model", "auc", "avg_precision",
                  "flag_rate", "fnr_gap", "gap_lo", "gap_hi"]]
          .round(4).to_string(index=False))

    print("\n=== the headline ===================================================")
    a, b = C.PRIMARY_GROUPS
    rows = []
    for left, right in C.CONTRASTS:
        for model in sorted(allsum["model"].unique()):
            try:
                pl = load_oof(left, model, expect_n=len(y))
                pr = load_oof(right, model, expect_n=len(y))
            except FileNotFoundError:
                continue
            gl = E.fnr_gap(y, pl, groups, E.threshold_for_rate(pl))
            gr = E.fnr_gap(y, pr, groups, E.threshold_for_rate(pr))
            lo, hi, p_one_sided = paired_bootstrap(y, pl, pr, groups)
            rows.append({
                "contrast": f"{left} - {right}", "model": model,
                "gap_first": gl, "gap_second": gr,
                "difference": gl - gr, "lo": lo, "hi": hi,
                "p_one_sided": p_one_sided,
            })

    head = pd.DataFrame(rows)
    print(head.round(4).to_string(index=False))
    print(f"\nA 'gap' is the {a} minus {b} share of children with asthma who")
    print("were missed. gap_first is under the first equation of the pair,")
    print("gap_second under the second. 'difference' is how much wider the gap")
    print("gets under the first one. An interval clear of zero is the result.")

    head.round(5).to_csv(C.TABLES / "05_headline.csv", index=False)
    allsum.round(5).to_csv(C.TABLES / "05_all_runs.csv", index=False)
    print(f"\nwrote {C.TABLES / '05_headline.csv'}")

    make_figure(y, groups, M.PRIMARY_MODEL, C.FIGURES / "05_missed_by_race.png")


if __name__ == "__main__":
    main()
