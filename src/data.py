"""Load the merged cohort and put the shared columns into one known shape.

Every script starts here, so the rest of the project can assume:
  Sex    0 = female, 1 = male
  Age    whole years
  Height centimetres
  race   one of the five NHANES labels
"""
import pandas as pd
import config as C

SEX_MAP = {
    "male": 1, "m": 1, "1": 1, "1.0": 1,
    "female": 0, "f": 0, "2": 0, "2.0": 0, "0": 0, "0.0": 0,
}


def normalise_sex(s: pd.Series) -> pd.Series:
    """Accept Male/Female, M/F, NHANES 1/2, or 0/1 and return 0/1."""
    out = s.astype(str).str.strip().str.lower().map(SEX_MAP)
    if out.isna().any():
        bad = sorted(s[out.isna()].astype(str).unique())[:5]
        raise ValueError(f"Sex values not understood: {bad}")
    return out.astype(int)


def check_height_cm(s: pd.Series) -> pd.Series:
    """Catch heights given in metres or inches before they poison every z-score."""
    med = s.median()
    if med < 3:
        raise ValueError(f"Height looks like metres (median {med:.2f}); need cm.")
    if med < 90:
        raise ValueError(f"Height looks like inches (median {med:.1f}); need cm.")
    return s


def load_cohort(path=None) -> pd.DataFrame:
    """Read the merged cohort, tidy the shared columns, filter to the age band."""
    path = path or C.COHORT_CSV
    df = pd.read_csv(path)

    required = ["SEQN", "Age", "Sex", "Height", "race", C.OUTCOME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing {missing}. Run 00 first.")

    df["Sex"] = normalise_sex(df["Sex"])
    df["Height"] = check_height_cm(df["Height"])
    df["race"] = df["race"].astype(str).str.strip()

    unknown = set(df["race"].unique()) - set(C.HANKINSON_ETHNICITY)
    if unknown:
        raise ValueError(f"race labels with no Hankinson mapping: {sorted(unknown)}")

    before = len(df)
    df = df[df["Age"].between(C.AGE_MIN, C.AGE_MAX)].reset_index(drop=True)
    print(f"[data] {path.name}: {before} rows -> {len(df)} aged "
          f"{C.AGE_MIN}-{C.AGE_MAX} ({before - len(df)} dropped)")

    if df["SEQN"].duplicated().any():
        raise ValueError("duplicate SEQN in the cohort; the merge went wrong")
    return df


def load_analysis(path=None) -> pd.DataFrame:
    """Read the table that already has every equation's z-scores."""
    path = path or C.ANALYSIS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/02_equations.py first.")
    return pd.read_csv(path)


def healthy_mask(df: pd.DataFrame) -> pd.Series:
    """Children a reference equation may be fitted on.

    Reference equations are built from people without respiratory disease, so
    we use children who have never been told they had asthma.
    """
    if "asthma_ever" not in df.columns:
        raise KeyError("asthma_ever is needed to pick the healthy reference sample")
    return df["asthma_ever"] == 0
