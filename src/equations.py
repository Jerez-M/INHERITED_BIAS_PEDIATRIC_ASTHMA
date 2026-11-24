"""Turn litres of breath into z-scores, four different ways.

legacy         Hankinson 1999, the published race-specific equation (pyspiro)
neutral        GLI Global 2022, already in the source file as *_GLOBAL columns (Bowerman Research)
ownfit_race    our own equation, same form as Hankinson, fitted per race
ownfit_norace  our own equation, same form, race left out

The last two exist so the comparison does not depend on two outside research
teams having made the same modelling choices. Same specification, same data,
one difference: race.
"""

import numpy as np
import pandas as pd
import pyspiro

import config as C
from . import data as D

_HANK = pyspiro.HANKINSON_1999()
_P = pyspiro.HANKINSON_1999.Parameters

# the three measurements, and what each one is called in the source file
MEASURES = {
    "FEV1": ("Baseline_FEV1_L", _P.FEV1),
    "FVC": ("Baseline_FVC_L", _P.FVC),
    "Ratio": ("Baseline_FEV1_FVC_Ratio", _P.FEV1FVC),
}


def _ratio_as_percent(s: pd.Series) -> pd.Series:
    """Hankinson's table 6 is in percent; the source file may hold a fraction."""
    return s * 100 if s.median() < 2 else s


# 1. the published race-specific equation
def add_hankinson(df: pd.DataFrame) -> pd.DataFrame:
    """Add FEV1_z_hank, FVC_z_hank, Ratio_z_hank (race-specific, published)."""
    out = df.copy()
    out["_eth"] = out["race"].map(C.HANKINSON_ETHNICITY)

    for name, (col, param) in MEASURES.items():
        value = _ratio_as_percent(out[col]) if name == "Ratio" else out[col]
        work = pd.DataFrame(
            {
                "sex": out["Sex"],
                "age": out["Age"],
                "height": out["Height"],
                "eth": out["_eth"],
                "val": value,
            }
        )
        res = _HANK.compute(
            work,
            parameter=param,
            sex_col="sex",
            age_col="age",
            height_col="height",
            value_col="val",
            ethnicity_col="eth",
            metrics=("zscore",),
        )
        out[f"{name}_z_hank"] = pd.to_numeric(res["zscore"], errors="coerce").values

    lost = out["FEV1_z_hank"].isna().sum()
    if lost:
        print(
            f"[equations] Hankinson returned nothing for {lost} children "
            f"(outside its published age or height range)"
        )
    return out.drop(columns="_eth")


# 2. our own equation, fitted here
def _design(df: pd.DataFrame, name: str) -> np.ndarray:
    """Hankinson's own functional form.

    volumes  b0 + b1*age + b2*age^2 + b3*height^2
    ratio    b0 + b1*age                            (no height, as in table 6)
    """
    age = df["Age"].to_numpy(float)
    if name == "Ratio":
        return np.column_stack([np.ones_like(age), age])
    ht2 = df["Height"].to_numpy(float) ** 2
    return np.column_stack([np.ones_like(age), age, age**2, ht2])


class OwnEquation:
    """Least squares on healthy children, fitted per sex (and per race if asked).

    Mirrors how Hankinson built his: one equation per sex per race group, with
    the spread of the healthy residuals as the z-score denominator.
    """

    def __init__(self, race_aware: bool):
        self.race_aware = race_aware
        self.coef_ = {}  # (measure, sex, race) -> beta
        self.sd_ = {}  # (measure, sex, race) -> residual sd

    def _keys(self, df: pd.DataFrame) -> pd.Series:
        race = df["race"] if self.race_aware else pd.Series("all", index=df.index)
        return list(zip(df["Sex"], race))

    def fit(self, healthy: pd.DataFrame):
        for name, (col, _) in MEASURES.items():
            value = _ratio_as_percent(healthy[col]) if name == "Ratio" else healthy[col]
            frame = healthy.assign(_y=value)
            frame["_key"] = self._keys(frame)

            for key, grp in frame.groupby("_key"):
                X, y = _design(grp, name), grp["_y"].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
                X, y = X[ok], y[ok]
                if len(y) < 30:
                    raise ValueError(
                        f"only {len(y)} healthy children for {name} {key}; "
                        f"too few to fit a reference equation"
                    )
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                self.coef_[(name, *key)] = beta
                self.sd_[(name, *key)] = resid.std(ddof=X.shape[1])
        return self

    def transform(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        out = df.copy()
        keys = self._keys(df)
        for name, (col, _) in MEASURES.items():
            value = _ratio_as_percent(df[col]) if name == "Ratio" else df[col]
            X = _design(df, name)
            z = np.full(len(df), np.nan)
            for i, key in enumerate(keys):
                beta = self.coef_.get((name, *key))
                if beta is None:
                    continue
                pred = X[i] @ beta
                z[i] = (value.iloc[i] - pred) / self.sd_[(name, *key)]
            out[f"{name}_z_{suffix}"] = z
        return out


def add_own_equations(df: pd.DataFrame) -> pd.DataFrame:
    """Add both own-fit pairs: race in and race out."""
    healthy = df[D.healthy_mask(df)]
    print(f"[equations] fitting our own equations on {len(healthy)} healthy children")

    out = OwnEquation(race_aware=True).fit(healthy).transform(df, "own_race")
    out = OwnEquation(race_aware=False).fit(healthy).transform(out, "own_norace")
    return out


def build_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every equation's z-scores on one table, then drop children any of them
    could not score, so all five runs see exactly the same children."""
    out = add_own_equations(add_hankinson(df))

    needed = [c for cols in C.EQUATION_TYPES.values() for c in cols]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise KeyError(f"these columns were never built: {missing}")

    before = len(out)
    out = out.dropna(subset=needed + [C.OUTCOME]).reset_index(drop=True)
    print(f"[equations] {before} -> {len(out)} children scored by every equation")
    return out
