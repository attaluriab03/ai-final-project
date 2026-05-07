"""
preprocessing.py

Model-specific preprocessing pipelines built on top of the raw feature matrix
produced by feature_engineering.py.

Each model has different requirements:
  - LogisticRegression : needs imputation + standard scaling
  - XGBoost            : handles NaN natively, no scaling needed, label-encode categoricals
  - FT-Transformer     : needs imputation + scaling (like LogReg), separate cat/num treatment
  - TabPFN             : needs imputation (can't pass NaN), light scaling recommended

All pipelines are sklearn-compatible (fit / transform / fit_transform).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# ── Identify feature groups ───────────────────────────────────────────────────

def get_feature_groups(X: pd.DataFrame) -> tuple[list, list]:
    """
    Returns (categorical_cols, numerical_cols).
    ICUType and Gender are treated as categorical (low-cardinality integers).
    """
    cat_cols = [c for c in X.columns if c in ("ICUType", "Gender")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


# ── Shared utility ────────────────────────────────────────────────────────────

class DataFrameOutput(BaseEstimator, TransformerMixin):
    """Wraps a sklearn transformer and preserves DataFrame column names."""
    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        if self.columns is None:
            if hasattr(X, "columns"):
                self.columns = list(X.columns)
        return self

    def transform(self, X):
        arr = self.transformer.transform(X)
        cols = self.columns if self.columns is not None else range(arr.shape[1])
        return pd.DataFrame(arr, columns=cols, index=X.index if hasattr(X, "index") else None)


# ── Model 1: Logistic Regression pipeline ────────────────────────────────────

def get_logreg_pipeline() -> Pipeline:
    """
    Median imputation → Standard scaling.
    Returns a fitted-ready sklearn Pipeline that outputs a numpy array.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── Model 2: XGBoost pipeline ─────────────────────────────────────────────────

def get_xgboost_pipeline() -> Pipeline:
    """
    XGBoost handles NaN natively, so only minimal preprocessing:
    - Replace -1 sentinel values (already done in feature_engineering, but safety check)
    - No scaling needed
    - Returns numpy array (XGBoost accepts NaN directly)
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=np.nan)),
        # identity scaler — keeps NaN passthrough for XGBoost
    ])


class XGBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Minimal preprocessor for XGBoost:
    - Converts DataFrame to numpy float32
    - XGBoost handles missing values internally
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return X.astype(np.float32)


# ── Model 3: FT-Transformer pipeline ─────────────────────────────────────────

class FTTransformerPreprocessor(BaseEstimator, TransformerMixin):
    """
    Prepares data for FT-Transformer (rtdl library):
    - Separates numerical and categorical features
    - Imputes numericals with median, categoricals with mode
    - Scales numericals with StandardScaler
    - Label-encodes categoricals to integer indices

    After transform(), returns a dict with keys:
        'x_num' : np.ndarray float32  (n_samples, n_num_features)
        'x_cat' : np.ndarray int64    (n_samples, n_cat_features)  or None
    """
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler      = StandardScaler()
        self.cat_encoders = {}
        self.num_cols_ = None
        self.cat_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_cols_, self.num_cols_ = get_feature_groups(X)

        # Numerical
        X_num = X[self.num_cols_].values
        self.num_imputer.fit(X_num)
        X_num_imp = self.num_imputer.transform(X_num)
        self.scaler.fit(X_num_imp)

        # Categorical
        if self.cat_cols_:
            X_cat = X[self.cat_cols_].values
            self.cat_imputer.fit(X_cat)
            X_cat_imp = self.cat_imputer.transform(X_cat)
            for i, col in enumerate(self.cat_cols_):
                le = LabelEncoder()
                le.fit(X_cat_imp[:, i].astype(str))
                self.cat_encoders[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> dict:
        # Numerical
        X_num = X[self.num_cols_].values
        X_num = self.num_imputer.transform(X_num)
        X_num = self.scaler.transform(X_num).astype(np.float32)

        # Categorical
        if self.cat_cols_:
            X_cat = X[self.cat_cols_].values
            X_cat = self.cat_imputer.transform(X_cat)
            X_cat_enc = np.zeros_like(X_cat, dtype=np.int64)
            for i, col in enumerate(self.cat_cols_):
                le = self.cat_encoders[col]
                col_vals = X_cat[:, i].astype(str)
                # Handle unseen labels gracefully
                known = set(le.classes_)
                col_vals = np.array([v if v in known else le.classes_[0] for v in col_vals])
                X_cat_enc[:, i] = le.transform(col_vals)
        else:
            X_cat_enc = None

        return {"x_num": X_num, "x_cat": X_cat_enc}

    def get_cardinalities(self, X: pd.DataFrame) -> list[int]:
        """Returns list of cardinalities for categorical features (needed by FT-Transformer)."""
        if not self.cat_cols_:
            return []
        return [len(self.cat_encoders[col].classes_) for col in self.cat_cols_]


# ── Model 4: TabPFN pipeline ──────────────────────────────────────────────────

def get_tabpfn_pipeline() -> Pipeline:
    """
    TabPFN requires:
    - No NaN values
    - Numerical inputs (categoricals encoded as integers are fine)
    - StandardScaler recommended for stability

    Simple median imputation + scaling is sufficient.
    TabPFN handles the rest internally via its meta-learned prior.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── Label conversion utilities ────────────────────────────────────────────────

def labels_to_binary(y: pd.Series) -> np.ndarray:
    """Convert {1, -1} labels to {1, 0} for sklearn/XGBoost/TabPFN compatibility."""
    return ((y + 1) // 2).values.astype(int)  # 1→1, -1→0


def binary_to_labels(y: np.ndarray) -> np.ndarray:
    """Convert {1, 0} back to {1, -1}."""
    return np.where(y == 1, 1, -1)