"""Out-of-fold prediction, a matched flag rate, and who gets missed.

The one idea worth holding on to: every equation type is forced to flag the
same number of children. If one of them then misses more Black children with
asthma than another, that is not because it flagged fewer children overall.
It is because it flagged different ones.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import config as C


# ---------------------------------------------------------------------------
def oof_probabilities(
    model, X, y, seed=C.SEED, n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS
):
    """Out-of-fold probability for every child, averaged over the repeats.

    Every child is predicted by a model that never saw them. Repeating the
    split and averaging stops one lucky fold from carrying the result.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    per_repeat = np.full((len(y), n_repeats), np.nan)
    for i, (train, test) in enumerate(cv.split(X, y)):
        est = clone(model).fit(X[train], y[train])
        per_repeat[test, i // n_splits] = est.predict_proba(X[test])[:, 1]

    if np.isnan(per_repeat).any():
        raise RuntimeError("some children were never held out")
    return per_repeat.mean(axis=1), per_repeat


# ---------------------------------------------------------------------------
def threshold_for_rate(p, rate=C.FLAG_RATE) -> float:
    """The cut-off that flags exactly `rate` of the children."""
    return float(np.quantile(p, 1.0 - rate))


def _rates(y, flagged):
    """False negative and false positive rate for one set of children."""
    pos, neg = y == 1, y == 0
    fnr = float((~flagged[pos]).mean()) if pos.any() else np.nan
    fpr = float(flagged[neg].mean()) if neg.any() else np.nan
    return fnr, fpr


def group_table(y, p, groups, thr) -> pd.DataFrame:
    """One row per race group: how many, how many missed."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    flagged = p >= thr

    rows = []
    for g in ["ALL"] + sorted(pd.unique(groups)):
        m = np.ones(len(y), bool) if g == "ALL" else (groups == g)
        fnr, fpr = _rates(y[m], flagged[m])
        rows.append(
            {
                "group": g,
                "n": int(m.sum()),
                "asthma": int(y[m].sum()),
                "flag_rate": float(flagged[m].mean()),
                "missed": int((y[m] == 1).sum() - (flagged[m] & (y[m] == 1)).sum()),
                "fnr": fnr,
                "fpr": fpr,
            }
        )
    return pd.DataFrame(rows)


def fnr_gap(y, p, groups, thr, pair=None) -> float:
    """False negative rate in the first group minus the second.

    Positive means the first group's children with asthma are missed more often.
    """
    a, b = pair or C.PRIMARY_GROUPS
    y, p, groups = np.asarray(y, int), np.asarray(p, float), np.asarray(groups)
    flagged = p >= thr
    fa, _ = _rates(y[groups == a], flagged[groups == a])
    fb, _ = _rates(y[groups == b], flagged[groups == b])
    return fa - fb


# ---------------------------------------------------------------------------
def bootstrap_gap(
    y, p, groups, rate=C.FLAG_RATE, n=C.N_BOOTSTRAP, seed=C.SEED, pair=None
):
    """Confidence interval for the gap, resampling children.

    The threshold is recomputed inside every resample, so the matched flag
    rate holds there too.
    """
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y, int), np.asarray(p, float), np.asarray(groups)
    a, b = pair or C.PRIMARY_GROUPS

    gaps = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys, ps, gs = y[s], p[s], groups[s]
        if (ys[gs == a].sum() == 0) or (ys[gs == b].sum() == 0):
            gaps[i] = np.nan
            continue
        gaps[i] = fnr_gap(ys, ps, gs, threshold_for_rate(ps, rate), pair)

    gaps = gaps[~np.isnan(gaps)]
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)), gaps


# ---------------------------------------------------------------------------
def evaluate(
    model, X, y, groups, equation_type, model_name, seed=C.SEED, n_repeats=C.N_REPEATS
):
    """Everything about one (equation type, model) pair."""
    p, _ = oof_probabilities(model, X, y, seed=seed, n_repeats=n_repeats)
    thr = threshold_for_rate(p)

    table = group_table(y, p, groups, thr)
    table.insert(0, "model", model_name)
    table.insert(0, "equation_type", equation_type)

    gap = fnr_gap(y, p, groups, thr)
    lo, hi, _ = bootstrap_gap(y, p, groups, seed=seed)

    summary = {
        "equation_type": equation_type,
        "model": model_name,
        "n": len(y),
        "n_repeats": n_repeats,
        "auc": roc_auc_score(y, p),
        "avg_precision": average_precision_score(y, p),
        "threshold": thr,
        "flag_rate": float((p >= thr).mean()),
        "fnr_gap": gap,
        "gap_lo": lo,
        "gap_hi": hi,
    }
    return summary, table, p
