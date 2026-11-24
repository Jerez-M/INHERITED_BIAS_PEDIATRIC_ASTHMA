"""Score every child with every equation and write data/processed/analysis.csv.

This is where the experiment is actually set up. After this, the five runs
differ only in which three columns they read.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import config as C
from src import data as D, equations as EQ

pd.set_option("display.width", 140)


def main():
    C.ensure_dirs()
    df = D.load_cohort()
    out = EQ.build_analysis_table(df)

    print("\n=== mean z-score by race, under each equation =====================")
    rows = {}
    for name, cols in C.EQUATION_TYPES.items():
        if name == "raw":
            continue
        rows[name] = out.groupby("race")[cols[0]].mean()  # the FEV1 column
    means = pd.DataFrame(rows)
    print(means.round(3).to_string())
    print("\n(the FEV1 column of each equation. A race-specific equation pulls")
    print("every group towards zero; a race-neutral one leaves the differences")
    print("where they are.)")

    gap = (means.loc["Black"] - means.loc["White"]).rename("Black - White")
    print("\n" + gap.round(3).to_string())

    out.to_csv(C.ANALYSIS_CSV, index=False)
    means.round(4).to_csv(C.TABLES / "02_mean_z_by_race.csv")
    print(f"\nwrote {C.ANALYSIS_CSV}  ({len(out)} children)")
    print(f"wrote {C.TABLES / '02_mean_z_by_race.csv'}")


if __name__ == "__main__":
    main()
