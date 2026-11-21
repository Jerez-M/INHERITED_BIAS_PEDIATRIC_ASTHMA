"""The models. Four of them, identical for every equation type.

Three single learners, then a stack that learns how much to trust each one.
Nothing here knows anything about race or about which equation type it is
looking at - that is the point.
"""
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C


def _logistic(seed):
    """Penalised logistic regression. The one a reviewer can read off the page."""
    return Pipeline([
        ("scale", StandardScaler()),
        # C=1.0 is L2 at the library default; naming the penalty is deprecated
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", random_state=seed)),
    ])


def _forest(seed, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed,
    )


def _boosting(seed):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        class_weight="balanced", random_state=seed,
    )


def build_models(seed: int = C.SEED) -> dict:
    """Return {name: estimator}. Same settings for every equation type."""
    stack = StackingClassifier(
        estimators=[
            ("logistic", _logistic(seed)),
            # single-threaded on purpose: the stack below does the parallelism,
            # and nesting the two is about five times slower
            ("forest", _forest(seed, n_jobs=1)),
            ("boosting", _boosting(seed)),
        ],
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced",
                                           random_state=seed),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    return {
        "logistic": _logistic(seed),
        "forest": _forest(seed),
        "boosting": _boosting(seed),
        "stack": stack,
    }


PRIMARY_MODEL = "stack"
