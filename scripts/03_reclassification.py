"""Who changes category when you swap the equation. No model involved.

A child is "below the lower limit of normal" when their z-score is under
-1.645, which is the bottom 5% of a healthy reference population. Swapping
the equation moves some children across that line. This script counts them,
by race, and tests whether the moves go mostly one way.

Worth saying out loud: below LLN does not mean the child has a disease. It
means their lungs measured lower than 95% of the healthy children the
equation was built from.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import binomtest

import config as C
from src import data as D

LLN_Z = -1.645


def mcnemar(b: int, c: int):
    """Exact McNemar on the children who moved. b and c are the two directions."""
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def compare(df, left, right, measure=0):
    """Cross-tabulate below-LLN under two equation types, overall and by race."""
    lcol = C.EQUATION_TYPES[left][measure]
    rcol = C.EQUATION_TYPES[right][measure]
    low_l, low_r = df[lcol] < LLN_Z, df[rcol] < LLN_Z

    rows = []
    for g in ["ALL"] + sorted(df["race"].unique()):
        m = pd.Series(True, index=df.index) if g == "ALL" else (df["race"] == g)
        l, r = low_l[m], low_r[m]
        only_left = int((l & ~r).sum())     # below LLN only under `left`
        only_right = int((~l & r).sum())    # below LLN only under `right`
        rows.append({
            "group": g,
            "n": int(m.sum()),
            f"below_{left}": int(l.sum()),
            f"below_{right}": int(r.sum()),
            f"rate_{left}": float(l.mean()),
            f"rate_{right}": float(r.mean()),
            f"only_{left}": only_left,
            f"only_{right}": only_right,
            "moved": only_left + only_right,
            "p_mcnemar": mcnemar(only_left, only_right),
        })
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    df = D.load_analysis()
    print(f"[03] {len(df)} children\n")

    all_tables = []
    for left, right in C.CONTRASTS:
        print(f"=== {left} vs {right}  (FEV1, z < {LLN_Z}) "
              f"{'=' * max(0, 40 - len(left) - len(right))}")
        t = compare(df, left, right)
        print(t.round(4).to_string(index=False))

        row = t[t["group"] == "ALL"].iloc[0]
        print(f"\n{int(row['moved'])} children changed category. "
              f"{int(row[f'only_{left}'])} were below the line only under "
              f"{left}, {int(row[f'only_{right}'])} only under {right}.")
        print()

        t.insert(0, "contrast", f"{left}_vs_{right}")
        all_tables.append(t)

    out = pd.concat(all_tables, ignore_index=True)
    path = C.TABLES / "03_reclassification.csv"
    out.round(5).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
