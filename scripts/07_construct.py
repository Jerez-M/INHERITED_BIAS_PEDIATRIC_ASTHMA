"""Does the z-score measure the same thing in every group?

A z-score is supposed to stand for "how unusual this child's lungs are". If
it does, it should track the things asthma actually does to a child - attacks,
emergency room visits - and it should track them about equally well whoever
the child is.

If a race-adjusted z-score tracks real asthma worse in Black children than in
White children, the number is not measuring the same thing in both groups.
That is a construct validity problem, and it is a different criticism from
"the model is less accurate". A less accurate model can be retrained. A
number that means different things in different groups cannot be fixed by
more data.

Convergent: the z-score should track asthma and its severity.
Discriminant: it should not track BMI nearly as strongly, because BMI is a
different construct.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
from src import data as D

# things the z-score SHOULD track, in rising order of severity
CONVERGENT = ["asthma_current", "asthma_attack_12m", "asthma_er_12m"]
# something it should track much less
DISCRIMINANT = ["BMI"]


def corr(x, y):
    """Correlation on the rows where both are present."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def main():
    C.ensure_dirs()
    df = D.load_analysis()

    targets = [c for c in CONVERGENT + DISCRIMINANT if c in df.columns]
    skipped = [c for c in CONVERGENT + DISCRIMINANT if c not in df.columns]
    if skipped:
        print(f"[07] not in the analysis table, skipping: {skipped}")
    if not targets:
        raise SystemExit("none of the comparison columns are present")

    rows = []
    for et, cols in C.EQUATION_TYPES.items():
        zcol = cols[0]                                  # the FEV1 column
        for group in ["ALL"] + C.PRIMARY_GROUPS:
            m = (pd.Series(True, index=df.index) if group == "ALL"
                 else df["race"] == group)
            if m.sum() < 50:
                continue
            row = {"equation_type": et, "group": group, "n": int(m.sum())}
            for t in targets:
                r, n = corr(df.loc[m, zcol], df.loc[m, t])
                row[f"r_{t}"] = r
                row[f"n_{t}"] = n
            rows.append(row)

    out = pd.DataFrame(rows)
    print("\n=== correlation between the FEV1 z-score and ... =================")
    print(out.round(3).to_string(index=False))
    print("\nNegative is the expected direction for the asthma columns: lower")
    print("lung function should go with more asthma.")

    a, b = C.PRIMARY_GROUPS
    print(f"\n=== does it measure the same thing in {a} and {b} children? ======")
    gaps = []
    for et in C.EQUATION_TYPES:
        sub = out[out["equation_type"] == et].set_index("group")
        if a not in sub.index or b not in sub.index:
            continue
        g = {"equation_type": et}
        for t in [c for c in CONVERGENT if f"r_{c}" in out.columns]:
            g[f"{t}_{a}"] = sub.loc[a, f"r_{t}"]
            g[f"{t}_{b}"] = sub.loc[b, f"r_{t}"]
            g[f"{t}_gap"] = sub.loc[a, f"r_{t}"] - sub.loc[b, f"r_{t}"]
        gaps.append(g)

    gapdf = pd.DataFrame(gaps)
    print(gapdf.round(3).to_string(index=False))
    print(f"\n'gap' is how much weaker the z-score's link to real asthma is in")
    print(f"{a} children than in {b} children. A gap near zero means the")
    print(f"number means the same thing in both groups. A gap that opens up")
    print(f"under one equation and not the other is the construct problem.")

    out.round(4).to_csv(C.TABLES / "07_construct_validity.csv", index=False)
    if len(gapdf):
        gapdf.round(4).to_csv(C.TABLES / "07_construct_gaps.csv", index=False)
    print(f"\nwrote {C.TABLES / '07_construct_validity.csv'}")


if __name__ == "__main__":
    main()
