"""Train every model on one equation type and write the results.

    python scripts/04_train.py --equation-type legacy
    python scripts/04_train.py --equation-type legacy --models stack

Same children, same folds, same seed, same settings every time. Only the
three lung-function columns change.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd

import config as C
from src import data as D, evaluate as E, features as F, models as M


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--equation-type", required=True, choices=list(C.EQUATION_TYPES),
                   help="which scoring of the breath the model sees")
    p.add_argument("--models", nargs="+", default=list(M.build_models()),
                   help="which models to run (default: all four)")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--n-repeats", type=int, default=C.N_REPEATS,
                   help="repeats of the 5-fold split. Use 2 or 3 while you are "
                        "still changing things; the default is for the final run.")
    return p.parse_args()


def main():
    args = parse_args()
    C.ensure_dirs()
    et = args.equation_type

    df = D.load_analysis()
    X, y, groups = F.build(df, equation_type=et)

    if y.sum() < 50:
        print(f"\nWARNING: only {int(y.sum())} children with asthma. "
              f"Subgroup rates will be very uncertain.\n")

    available = M.build_models(seed=args.seed)
    summaries, tables = [], []

    for name in args.models:
        if name not in available:
            raise SystemExit(f"unknown model {name!r}; choose from {list(available)}")
        print(f"\n--- {et} / {name} "
              f"{'-' * max(0, 50 - len(et) - len(name))}")

        summary, table, p = E.evaluate(available[name], X, y, groups,
                                       equation_type=et, model_name=name,
                                       seed=args.seed, n_repeats=args.n_repeats)
        summaries.append(summary)
        tables.append(table)

        print(f"AUC {summary['auc']:.3f}   "
              f"average precision {summary['avg_precision']:.3f}   "
              f"flagging {summary['flag_rate']:.1%} of children")
        print(table.drop(columns=["equation_type", "model"])
                   .round(4).to_string(index=False))
        a, b = C.PRIMARY_GROUPS
        print(f"\nmissed-case gap ({a} - {b}): {summary['fnr_gap']:+.3f} "
              f"[{summary['gap_lo']:+.3f}, {summary['gap_hi']:+.3f}]")

        np.save(C.MODELS / f"oof_{et}_{name}.npy", p)

    pd.DataFrame(summaries).round(5).to_csv(
        C.TABLES / f"04_summary_{et}.csv", index=False)
    pd.concat(tables, ignore_index=True).round(5).to_csv(
        C.TABLES / f"04_groups_{et}.csv", index=False)
    print(f"\nwrote {C.TABLES / f'04_summary_{et}.csv'}")
    print(f"wrote {C.TABLES / f'04_groups_{et}.csv'}")


if __name__ == "__main__":
    main()
