"""Build the feature matrix for one equation type, with a leakage guard
that cannot be skipped."""
import pandas as pd
import config as C


class LeakageError(AssertionError):
    pass


def _check(cols):
    bad = [c for c in cols
           if c in C.BANNED_EXACT
           or any(s.lower() in c.lower() for s in C.BANNED_SUBSTRINGS)]
    if bad:
        raise LeakageError(
            f"These columns must never be model inputs: {bad}\n"
            f"Race is the moderator, not a feature. Flags are the outcome in disguise."
        )


def build(df: pd.DataFrame, equation_type: str):
    """Return X, y, groups for one equation type.

    The rows are identical across equation types. Only the three
    lung-function columns change.
    """
    if equation_type not in C.EQUATION_TYPES:
        raise KeyError(
            f"equation_type must be one of {list(C.EQUATION_TYPES)}"
        )

    cols = C.COMMON_FEATURES + C.EQUATION_TYPES[equation_type]
    _check(cols)                                   # always runs

    missing = [c for c in cols + [C.OUTCOME, "race"] if c not in df.columns]
    if missing:
        raise KeyError(f"missing from the analysis table: {missing}")

    keep = df[C.OUTCOME].notna()
    for c in cols:
        keep &= df[c].notna()

    X = df.loc[keep, cols].copy()
    X["Sex"] = (X["Sex"].astype(str).str.lower() == "male").astype(int)
    y = df.loc[keep, C.OUTCOME].astype(int)
    groups = df.loc[keep, "race"]

    print(f"[{equation_type}] n={len(X)}  positives={int(y.sum())}  "
          f"features={list(X.columns)}")
    return X, y, groups
