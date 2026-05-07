"""
model_logreg.py

Regularized Logistic Regression (L1 and L2 penalties) with cross-validated
hyperparameter selection.

Tuning:
  - C (inverse regularization strength): log-spaced grid
  - penalty: L1 (sparse, feature selection) and L2 (ridge-like, shrinkage)
  - Primary metric: AUROC

Class imbalance handled via class_weight='balanced'.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings

from preprocessing import get_logreg_pipeline, labels_to_binary


# ── Hyperparameter grid ───────────────────────────────────────────────────────

C_RANGE    = np.logspace(-4, 4, 17)   # 17 values from 1e-4 to 1e4
PENALTIES  = ["l2", "l1"]
SOLVER_MAP = {"l2": "lbfgs", "l1": "liblinear"}


def select_hyperparams(
    X: pd.DataFrame,
    y: np.ndarray,
    metric: str = "roc_auc",
    k: int = 5,
    verbose: bool = True,
) -> tuple[float, str]:
    """
    Grid search over C x penalty using k-fold stratified CV.

    Returns best (C, penalty) pair maximizing the given metric.
    """
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    best_score = -np.inf
    best_C     = 1.0
    best_pen   = "l2"
    results    = []

    preprocessor = get_logreg_pipeline()

    for penalty in PENALTIES:
        solver = SOLVER_MAP[penalty]
        for C in C_RANGE:
            clf = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf",  clf),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X, y, cv=kfold, scoring=metric, n_jobs=-1)

            mean_score = scores.mean()
            results.append({"C": C, "penalty": penalty, "mean_score": mean_score, "std": scores.std()})

            if mean_score > best_score:
                best_score = mean_score
                best_C     = C
                best_pen   = penalty

    if verbose:
        print(f"\n[LogReg] Best CV {metric}: {best_score:.4f}")
        print(f"[LogReg] Best params: C={best_C:.5g}, penalty={best_pen}")

    return best_C, best_pen, pd.DataFrame(results)


def build_model(C: float, penalty: str) -> Pipeline:
    """
    Returns a full sklearn Pipeline (preprocessor + classifier)
    ready to be fit on training data.
    """
    solver = SOLVER_MAP[penalty]
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    return Pipeline([
        ("prep", get_logreg_pipeline()),
        ("clf",  clf),
    ])


def get_feature_weights(pipeline: Pipeline, feature_names: list) -> pd.Series:
    """
    Extract learned weights from a fitted LogReg pipeline.
    Accounts for columns dropped by the imputer (all-NaN features).
    """
    clf = pipeline.named_steps["clf"]
    imputer = pipeline.named_steps["prep"].named_steps["imputer"]

    # Imputer marks all-NaN columns with nan in statistics_ — get surviving indices
    surviving_mask = ~np.isnan(imputer.statistics_)
    surviving_names = [name for name, keep in zip(feature_names, surviving_mask) if keep]

    weights = clf.coef_[0]
    return pd.Series(weights, index=surviving_names).sort_values(key=abs, ascending=False)


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    tune: bool = True,
    k: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Full training routine:
      1. (Optional) Hyperparameter tuning via CV
      2. Fit final model on full training set

    Returns fitted pipeline and a dict of training metadata.
    """
    if tune:
        best_C, best_pen, cv_results = select_hyperparams(X_train, y_train, k=k)
    else:
        best_C, best_pen = 1.0, "l2"
        cv_results = None

    model = build_model(best_C, best_pen)
    model.fit(X_train, y_train)

    meta = {
        "model_name": "LogisticRegression",
        "best_C": best_C,
        "best_penalty": best_pen,
        "cv_results": cv_results,
    }
    return model, meta