"""The controlled experiment, on invented children.

Two groups with identical lungs and identical asthma risk. The only thing we
change is how much the reference equation lowers its expectation for group B.
Running it at several sizes of adjustment gives a dose and a response: no
adjustment should mean no gap, and a bigger adjustment should mean a bigger one.

This is the part the real data cannot give us, because we cannot assign a
child a race.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from src import models as M, simulate as S


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=4000, help="invented children per run")
    p.add_argument("--model", default="logistic", choices=list(M.build_models()),
                   help="the sweep runs many times, so a fast model is sensible")
    p.add_argument("--seed", type=int, default=C.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    C.ensure_dirs()
    model = M.build_models(seed=args.seed)[args.model]

    print(f"[06] {args.n} invented children per run, {args.model} model")
    sweep = S.run_sweep(model, n=args.n, seed=args.seed)

    print("\n=== gap by size of race adjustment =================================")
    print(sweep[["offset", "gap_legacy", "gap_neutral", "difference"]]
          .round(4).to_string(index=False))
    print("\noffset 0 is the control: the equation is then the same for both")
    print("groups, so the difference there should sit on zero.")

    control = sweep.loc[sweep["offset"] == 0, "difference"]
    if len(control) and abs(control.iloc[0]) > 0.03:
        print(f"\nWARNING: the control run is off zero ({control.iloc[0]:+.3f}). "
              f"Something other than the equation is moving the result.")

    path = C.TABLES / "06_simulation_sweep.csv"
    sweep.round(5).to_csv(path, index=False)
    print(f"\nwrote {path}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(sweep["offset"], sweep["gap_legacy"], "o-", label="race-adjusted")
    ax.plot(sweep["offset"], sweep["gap_neutral"], "s-", label="race-neutral")
    ax.set_xlabel("how much the equation lowers its expectation for group B (z-units)")
    ax.set_ylabel("extra share of group B's\nasthma cases missed")
    ax.set_title("Identical lungs, identical risk.\nThe only thing changing is the equation.")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = C.FIGURES / "06_simulation.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
