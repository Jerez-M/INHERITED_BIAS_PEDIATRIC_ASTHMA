"""A controlled version of the study, on children we invent ourselves.

We cannot assign a child a race, so in the real data we can never be certain
the gap comes from the equation and not from something else about the two
groups. Here we can. The two invented groups have identical lungs and
identical asthma risk by construction. The only thing we change is how much
the reference equation lowers its expectation for group B.

If a gap appears, the equation made it. There is nothing else left.
"""
import numpy as np
import pandas as pd

import config as C


def make_children(n=4000, offset=0.30, seed=C.SEED) -> pd.DataFrame:
    """Invent n children whose biology does not depend on their group.

    offset: how much the race-adjusted equation lowers what it expects of
            group B, in z-units. 0.30 is close to what the published
            pediatric equations imply.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(C.AGE_MIN, C.AGE_MAX + 1, n)
    sex = rng.integers(0, 2, n)
    height = 95 + 5.6 * age + 4.0 * sex + rng.normal(0, 5, n)
    bmi = rng.normal(19, 3.2, n)
    group = rng.choice(["A", "B"], n)

    # true lung function: age, height, sex only. Group is nowhere in here.
    true_z = rng.normal(0, 1, n)
    fev1 = 0.00022 * height ** 2 + 0.035 * age + 0.10 * sex + 0.35 * true_z
    fvc = fev1 * (1.18 + rng.normal(0, 0.03, n))

    # asthma risk depends on the truth, not on the group
    logit = -2.1 - 1.05 * true_z + 0.02 * (bmi - 19)
    asthma = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "Age": age, "Sex": sex, "Height": height, "BMI": bmi,
        "race": group, "true_z": true_z,
        "Baseline_FEV1_L": fev1, "Baseline_FVC_L": fvc,
        "Baseline_FEV1_FVC_Ratio": fev1 / fvc,
        C.OUTCOME: asthma,
    })

    # the race-neutral equation: one expectation for every child
    df["FEV1_z_neutral"] = _standardise(df, "Baseline_FEV1_L")
    df["FVC_z_neutral"] = _standardise(df, "Baseline_FVC_L")
    df["Ratio_z_neutral"] = _standardise(df, "Baseline_FEV1_FVC_Ratio")

    # the race-adjusted equation: expects less of group B, so group B's
    # z-scores rise and their lungs look healthier than they are
    shift = np.where(df["race"] == "B", offset, 0.0)
    df["FEV1_z_legacy"] = df["FEV1_z_neutral"] + shift
    df["FVC_z_legacy"] = df["FVC_z_neutral"] + shift
    df["Ratio_z_legacy"] = df["Ratio_z_neutral"]      # ratio barely moves
    return df


def _standardise(df, col):
    """Residual z-score against age, sex and height squared - one equation for all."""
    X = np.column_stack([np.ones(len(df)), df["Age"], df["Sex"],
                         df["Height"] ** 2])
    y = df[col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return resid / resid.std(ddof=X.shape[1])


SIM_EQUATION_TYPES = {
    "legacy":  ["FEV1_z_legacy", "FVC_z_legacy", "Ratio_z_legacy"],
    "neutral": ["FEV1_z_neutral", "FVC_z_neutral", "Ratio_z_neutral"],
}


def run_once(model, offset=0.30, n=4000, seed=C.SEED) -> dict:
    """Train on both scorings of the same invented children, return the gaps."""
    from sklearn.metrics import roc_auc_score
    from . import evaluate as E

    df = make_children(n=n, offset=offset, seed=seed)
    y, groups = df[C.OUTCOME].to_numpy(int), df["race"].to_numpy()

    result = {"offset": offset}
    for name, cols in SIM_EQUATION_TYPES.items():
        X = df[C.COMMON_FEATURES + cols].to_numpy(float)
        p, _ = E.oof_probabilities(model, X, y, seed=seed, n_repeats=2)
        thr = E.threshold_for_rate(p)
        result[f"gap_{name}"] = E.fnr_gap(y, p, groups, thr, pair=("B", "A"))
        result[f"auc_{name}"] = float(roc_auc_score(y, p))
    result["difference"] = result["gap_legacy"] - result["gap_neutral"]
    return result


def run_sweep(model, offsets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
              n=4000, seed=C.SEED) -> pd.DataFrame:
    """Dose and response: a bigger race adjustment should mean a bigger gap.

    The offset = 0 row is the control. The equation is then the same for both
    groups, so the gap there should sit on zero.
    """
    rows = [run_once(model, offset=o, n=n, seed=seed) for o in offsets]
    return pd.DataFrame(rows)
