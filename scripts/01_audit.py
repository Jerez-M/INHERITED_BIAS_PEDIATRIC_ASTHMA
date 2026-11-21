"""Describe the cohort before any modelling. Every number here can go on a slide.

Also checks one thing that matters: are the z-score columns already in the
source file race-neutral? A race-specific equation centres each group near
zero, so the group means would all sit close together. If they are spread
out, the equation did not adjust for race.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import config as C
from src import data as D

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)


def main():
    C.ensure_dirs()
    df = D.load_cohort()

    print("\n=== cohort =========================================================")
    print(f"children            {len(df)}")
    print(f"age range           {df['Age'].min()}-{df['Age'].max()}")
    print(f"current asthma      {int(df[C.OUTCOME].sum())} "
          f"({df[C.OUTCOME].mean():.1%})")
    if "asthma_ever" in df:
        print(f"ever asthma         {int(df['asthma_ever'].sum())} "
              f"({df['asthma_ever'].mean():.1%})")

    print("\n=== by race ========================================================")
    by_race = df.groupby("race").agg(
        n=("SEQN", "size"),
        asthma=(C.OUTCOME, "sum"),
        prevalence=(C.OUTCOME, "mean"),
        mean_age=("Age", "mean"),
        mean_height=("Height", "mean"),
        mean_bmi=("BMI", "mean"),
    ).sort_values("n", ascending=False)
    by_race["asthma"] = by_race["asthma"].astype(int)
    print(by_race.round(3).to_string())

    small = by_race[by_race["asthma"] < 150].index.tolist()
    if small:
        print(f"\nfewer than 150 children with asthma in: {small}")
        print("subgroup rates for those groups will have wide intervals.")

    print(f"\nchildren scored with a borrowed Hankinson category "
          f"({', '.join(C.HANKINSON_DEFAULTED)}): "
          f"{int(df['race'].isin(C.HANKINSON_DEFAULTED).sum())}")

    print("\n=== are the z-scores in the source file race-neutral? ==============")
    zcols = [c for c in df.columns if "Zscore" in c or "zscore" in c]
    if not zcols:
        print("no z-score columns in the cohort file")
    else:
        means = df.groupby("race")[zcols].mean()
        spread = (means.max() - means.min()).sort_values(ascending=False)
        print(means.round(3).to_string())
        print("\nspread of the group means (max - min):")
        for col, s in spread.items():
            verdict = "race-specific" if s < 0.25 else "race-neutral or pooled"
            print(f"  {col:38s} {s:5.2f}   {verdict}")
        print("\nA race-specific equation centres every group near zero, so its")
        print("spread would be small. Anything above about 0.25 did not adjust.")

    print("\n=== missing ========================================================")
    gaps = df.isna().sum()
    print(gaps[gaps > 0].to_string() if (gaps > 0).any() else "none")

    by_race.round(4).to_csv(C.TABLES / "01_cohort_by_race.csv")
    print(f"\nwrote {C.TABLES / '01_cohort_by_race.csv'}")


if __name__ == "__main__":
    main()
