"""
model_xgboost.py

XGBoost gradient boosted trees for ICU mortality prediction.

Key advantages over LogReg for this task:
  - Handles missing values natively (learns optimal split direction for NaN)
  - Captures nonlinear feature interactions automatically
  - scale_pos_weight handles class imbalance without throwing away signal
  - Built-in regularization (L1 alpha, L2 lambda, min_child_weight)

Tuning via RandomizedSearchCV over the most impactful XGBoost hyperparameters.
Primary metric: AUROC.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import warnings

from preprocessing import XGBoostPreprocessor, labels_to_binary


# ── Hyperparameter search space ───────────────────────────────────────────────

PARAM_DIST = {
    "clf__n_estimators":      randint(100, 800),
    "clf__max_depth":         randint(3, 9),
    "clf__learning_rate":     uniform(0.01, 0.29),    # 0.01 – 0.30
    "clf__subsample":         uniform(0.6, 0.4),       # 0.6 – 1.0
    "clf__colsample_bytree":  uniform(0.5, 0.5),       # 0.5 – 1.0
    "clf__min_child_weight":  randint(1, 10),
    "clf__reg_alpha":         uniform(0, 1),            # L1
    "clf__reg_lambda":        uniform(0.5, 4.5),        # L2, default 1
    "clf__gamma":             uniform(0, 0.5),
}

N_ITER   = 50   # number of random candidates
CV_FOLDS = 5


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    XGBoost's built-in class imbalance handler.
    scale_pos_weight = count(negative) / count(positive)
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg) / float(n_pos)


def build_pipeline(scale_pos_weight: float = 1.0, **xgb_kwargs) -> Pipeline:
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **xgb_kwargs,
    )
    return Pipeline([
        ("prep", XGBoostPreprocessor()),
        ("clf",  clf),
    ])


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    tune: bool = True,
    n_iter: int = N_ITER,
    k: int = CV_FOLDS,
) -> tuple[Pipeline, dict]:
    """
    Train XGBoost with optional RandomizedSearchCV hyperparameter tuning.

    Parameters
    ----------
    X_train, y_train : training data (y in {0,1})
    X_val, y_val     : unused here (CV handles validation), kept for API consistency
    tune             : whether to run RandomizedSearchCV
    n_iter           : number of random candidates to evaluate
    k                : number of CV folds

    Returns
    -------
    fitted Pipeline, metadata dict
    """
    spw = _compute_scale_pos_weight(y_train)
    print(f"\n[XGBoost] scale_pos_weight = {spw:.2f}")

    if tune:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        base_pipe = build_pipeline(scale_pos_weight=spw)

        search = RandomizedSearchCV(
            base_pipe,
            param_distributions=PARAM_DIST,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=kfold,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        print(f"[XGBoost] Best CV AUROC : {search.best_score_:.4f}")
        print(f"[XGBoost] Best params   : {best_params}")

        model = search.best_estimator_
        meta = {
            "model_name": "XGBoost",
            "best_params": best_params,
            "best_cv_auroc": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }

    else:
        # Default hyperparameters — sensible baseline
        model = build_pipeline(
            scale_pos_weight=spw,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X_train, y_train)
        meta = {"model_name": "XGBoost", "best_params": "defaults"}

    return model, meta


def get_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.Series:
    """Extract feature importances from the fitted XGBoost model."""
    clf = pipeline.named_steps["clf"]
    importance = clf.feature_importances_
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)