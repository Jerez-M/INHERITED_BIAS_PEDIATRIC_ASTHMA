"""
Step 00 - Acquire NHANES asthma + demographic data and build the pediatric cohort.

Downloads MCQ and DEMO files for NHANES 2007-08, 2009-10, 2011-12, merges them
onto the spirometry CSV by SEQN, builds both asthma outcomes, and reports the
positive counts by race that decide whether subgroup modelling is viable.

Usage:
    python scripts/00_download_and_merge.py \
        --spirometry data/raw/NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# cycle -> (year folder, file suffix, SDDSRVYR code)
CYCLES = {
    "2007-2008": ("2007", "E", 5),
    "2009-2010": ("2009", "F", 6),
    "2011-2012": ("2011", "G", 7),
}

MCQ_KEEP = ["SEQN", "MCQ010", "MCQ025", "MCQ035", "MCQ040", "MCQ050"]
DEMO_KEEP = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
             "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "SDDSRVYR"]

RIDRETH1_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "White",
    4: "Black",
    5: "Other / multiracial",
}


def download(url: str, dest: Path) -> Path:
    """Download url to dest unless it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have   {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get    {dest.name} ...", end=" ", flush=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    size = dest.stat().st_size
    if size < 10_000:
        raise RuntimeError(f"{dest.name} is only {size} bytes - download likely failed")
    print(f"ok ({size/1e6:.1f} MB)")
    return dest


def read_xpt(path: Path, keep: list) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing {missing}")
    df = df[keep].copy()
    df["SEQN"] = df["SEQN"].astype("int64")
    return df


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    MCQ010  ever told you have asthma      1=Yes 2=No 7=Refused 9=DK
    MCQ035  do you still have asthma       asked only when MCQ010 == 1
    MCQ040  attack in past 12 months       asked only when MCQ035 == 1
    MCQ050  ER / urgent care, past 12 mo   asked only when MCQ035 == 1

    Refused / don't know / unanswered stay missing. They are NOT coded as 0.
    """
    out = df.copy()

    out["asthma_ever"] = out["MCQ010"].map({1.0: 1, 2.0: 0})

    cur = pd.Series(np.nan, index=out.index, dtype="float64")
    cur[out["MCQ010"] == 2] = 0                                      # never told
    cur[(out["MCQ010"] == 1) & (out["MCQ035"] == 1)] = 1             # still has it
    cur[(out["MCQ010"] == 1) & (out["MCQ035"] == 2)] = 0             # outgrown
    out["asthma_current"] = cur

    # Structural missingness: only defined among current asthmatics.
    has_current = out["asthma_current"] == 1
    out["asthma_attack_12m"] = np.where(
        has_current, out["MCQ040"].map({1.0: 1, 2.0: 0}), np.nan)
    out["asthma_er_12m"] = np.where(
        has_current, out["MCQ050"].map({1.0: 1, 2.0: 0}), np.nan)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirometry", required=True, type=Path)
    ap.add_argument("--raw-dir", default=Path("data/raw"), type=Path)
    ap.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    # ---------- 1. download ----------
    paths = {}
    print("NHANES files")
    for cycle, (year, suffix, _) in CYCLES.items():
        for kind in ("MCQ", "DEMO"):
            name = f"{kind}_{suffix}.XPT"
            url = f"{BASE}/{year}/DataFiles/{name}"
            dest = args.raw_dir / name
            if args.skip_download:
                if not dest.exists():
                    sys.exit(f"missing {dest} and --skip-download was set")
                print(f"  have   {name}")
            else:
                download(url, dest)
            paths[(cycle, kind)] = dest

    # ---------- 2. read and stack ----------
    mcq = pd.concat([read_xpt(paths[(c, "MCQ")], MCQ_KEEP) for c in CYCLES],
                    ignore_index=True)
    demo = pd.concat([read_xpt(paths[(c, "DEMO")], DEMO_KEEP) for c in CYCLES],
                     ignore_index=True)
    print(f"\nMCQ  rows {len(mcq):,}  unique SEQN {mcq.SEQN.nunique():,}")
    print(f"DEMO rows {len(demo):,}  unique SEQN {demo.SEQN.nunique():,}")
    for name, d in (("MCQ", mcq), ("DEMO", demo)):
        if d.SEQN.duplicated().any():
            sys.exit(f"{name} has duplicate SEQN - cycles may overlap")

    # ---------- 3. spirometry ----------
    spiro = pd.read_csv(args.spirometry, low_memory=False)
    spiro["SEQN"] = spiro["SEQN"].astype("int64")
    print(f"spiro rows {len(spiro):,}  unique SEQN {spiro.SEQN.nunique():,}")

    # ---------- 4. merge ----------
    df = spiro.merge(demo, on="SEQN", how="left", validate="one_to_one")
    df = df.merge(mcq, on="SEQN", how="left", validate="one_to_one")

    unmatched = df["RIDRETH1"].isna().sum()
    print(f"\nmerged: {len(df):,} rows, {unmatched} spirometry records "
          f"with no DEMO match")
    if unmatched:
        print("  -> check these before continuing:",
              df.loc[df.RIDRETH1.isna(), "SEQN"].head(10).tolist())

    # authoritative race from DEMO; resolves the stray 'Race == 4' row
    df["race"] = df["RIDRETH1"].map(RIDRETH1_LABELS)
    mismatch = (df["race"] != df["Race"].replace(
        {"Other hispanic": "Other Hispanic",
         "Other race, including multi-racial": "Other / multiracial"})).sum()
    print(f"race label disagreements CSV vs DEMO: {mismatch}")

    # 3-cycle MEC weight (NHANES guidance: 2-year weight / number of cycles)
    df["wtmec6yr"] = df["WTMEC2YR"] / len(CYCLES)

    # ---------- 5. outcomes ----------
    df = build_outcomes(df)

    # ---------- 6. pediatric cohort ----------
    kids = df[(df["Age"] >= 6) & (df["Age"] < 18)].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_out = args.out_dir / "nhanes_merged.csv"
    kid_out = args.out_dir / "pediatric_cohort.csv"
    df.to_csv(full_out, index=False)
    kids.to_csv(kid_out, index=False)

    # ---------- 7. the counts that decide the project ----------
    print(f"\n{'='*66}\nPEDIATRIC COHORT (6-17): n = {len(kids):,}\n{'='*66}")
    for outcome in ("asthma_ever", "asthma_current"):
        s = kids[outcome]
        print(f"\n{outcome}:  positive {int(s.sum()):,} "
              f"| negative {int((s == 0).sum()):,} "
              f"| missing {int(s.isna().sum()):,}")
        tab = (kids.groupby("race")[outcome]
                   .agg(n="size", positives="sum", missing=lambda x: x.isna().sum())
                   .astype({"positives": "Int64", "missing": "Int64"}))
        tab["rate_%"] = (100 * tab["positives"] / (tab["n"] - tab["missing"])).round(1)
        print(tab.sort_values("positives", ascending=False).to_string())

    # ---------- 8. the band the models actually use ----------
    # Hankinson 1999 has no pediatric coefficients below age 8, so scoring
    # every child with every equation means modelling 8-17. These are the
    # counts that decide whether subgroup rates are estimable at all.
    band = kids[kids["Age"] >= 8]
    print(f"\n{'='*66}\nMODELLING BAND (8-17): n = {len(band):,}"
          f"   [{len(kids) - len(band):,} younger children cannot be scored "
          f"by Hankinson]\n{'='*66}")
    tab = (band.groupby("race")["asthma_current"]
               .agg(n="size", positives="sum")
               .astype({"positives": "Int64"})
               .sort_values("positives", ascending=False))
    print(tab.to_string())
    thin = tab.index[tab["positives"] < 150].tolist()
    if thin:
        print(f"\nfewer than 150 current-asthma cases in: {thin}")
        print("those groups will have wide confidence intervals.")

    print(f"\nwrote {full_out}")
    print(f"wrote {kid_out}")
    print("\nDecision point: if Black positives for asthma_current are below "
          "~150, treat subgroup results as descriptive, not predictive.")


if __name__ == "__main__":
    main()
