"""
model_tabpfn.py

TabPFN v2 for ICU mortality prediction.
Reference: Hollmann et al., "TabPFN: A Transformer That Solves Small Tabular
           Classification Problems in a Second" (2022); v2 (2024).

What makes TabPFN fundamentally different from the other three models:

  - It is a PRETRAINED transformer meta-learned on millions of synthetic
    classification datasets sampled from a prior distribution over Bayesian
    networks and decision trees.
  - At inference time, the entire training set is passed as CONTEXT — the model
    does in-context learning (like a large language model) rather than gradient
    descent weight updates on your data.
  - .fit() stores the training data; .predict_proba() passes both training and
    test data together through the transformer in a single forward pass.
  - No explicit hyperparameter tuning loop is needed: the model adapts to the
    dataset structure automatically via attention over training examples.

Practical considerations:
  - TabPFN v2 supports datasets up to ~10,000 samples and ~500 features.
  - If the dataset exceeds this, we subsample for the "context" shown to TabPFN.
  - Preprocessing: median imputation + StandardScaler (no NaN allowed).
  - The model internally handles class imbalance to some degree.

TabPFN v2 install: pip install tabpfn
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

from preprocessing import get_tabpfn_pipeline, labels_to_binary


# Maximum context size TabPFN v2 can handle comfortably on CPU
TABPFN_MAX_SAMPLES  = 10_000
TABPFN_MAX_FEATURES = 500


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around TabPFN v2.

    Handles:
      - Preprocessing (imputation + scaling) via sklearn Pipeline
      - Subsampling if dataset is too large for TabPFN's context window
      - Graceful fallback error messages if tabpfn is not installed
    """

    def __init__(
        self,
        max_samples: int = TABPFN_MAX_SAMPLES,
        ignore_pretraining_limits: bool = True,
        random_state: int = 42,
    ):
        self.max_samples = max_samples
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

        self.preprocessor_ = None
        self.tabpfn_        = None
        self.classes_       = np.array([0, 1])
        self._X_train_proc  = None
        self._y_train       = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise ImportError(
                "TabPFN not installed. Run: pip install tabpfn\n"
                "Note: requires Python 3.9+ and ~500MB download on first use."
            )

        # Preprocessing
        self.preprocessor_ = get_tabpfn_pipeline()
        X_proc = self.preprocessor_.fit_transform(X)

        # Subsample if needed (TabPFN context window limit)
        n = len(X_proc)
        if n > self.max_samples:
            print(f"[TabPFN] Dataset ({n} samples) exceeds max context size ({self.max_samples}). "
                  f"Subsampling training context.")
            rng = np.random.default_rng(self.random_state)
            # Stratified subsample
            idx_pos = np.where(y == 1)[0]
            idx_neg = np.where(y == 0)[0]
            frac    = self.max_samples / n
            sel_pos = rng.choice(idx_pos, size=int(len(idx_pos) * frac), replace=False)
            sel_neg = rng.choice(idx_neg, size=int(len(idx_neg) * frac), replace=False)
            sel_idx = np.concatenate([sel_pos, sel_neg])
            X_proc  = X_proc[sel_idx]
            y       = y[sel_idx]
            print(f"[TabPFN] Context size after subsampling: {len(X_proc)}")

        # Feature truncation if needed
        if X_proc.shape[1] > TABPFN_MAX_FEATURES:
            print(f"[TabPFN] Truncating features to {TABPFN_MAX_FEATURES} (from {X_proc.shape[1]})")
            X_proc = X_proc[:, :TABPFN_MAX_FEATURES]

        # Initialise TabPFN — no actual training happens here beyond storing context
        self.tabpfn_ = TabPFNClassifier(
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            random_state=self.random_state,
        )

        print(f"\n[TabPFN] Storing training context: {X_proc.shape[0]} samples × {X_proc.shape[1]} features")
        print("[TabPFN] Note: TabPFN does in-context learning — no gradient descent is run.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tabpfn_.fit(X_proc, y)

        # Cache preprocessed training data shape for feature truncation in predict
        self._n_features = X_proc.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.preprocessor_.transform(X)
        if X_proc.shape[1] > self._n_features:
            X_proc = X_proc[:, :self._n_features]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.tabpfn_.predict_proba(X_proc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    tune: bool = True,   # TabPFN has minimal hyperparameters; tune flag is mostly ignored
) -> tuple[TabPFNWrapper, dict]:
    """
    'Train' TabPFN — in practice this stores the training context.

    TabPFN has no conventional hyperparameters to tune (the model weights are
    frozen from pretraining). The only meaningful knob is the size of the
    training context shown to the model, which we handle via max_samples.
    """
    print("\n[TabPFN] TabPFN uses in-context learning — hyperparameter tuning is not applicable.")
    print("[TabPFN] The model's weights are frozen; adaptation happens via attention over training examples.")

    model = TabPFNWrapper(
        max_samples=TABPFN_MAX_SAMPLES,
        ignore_pretraining_limits=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    meta = {
        "model_name": "TabPFN v2",
        "paradigm": "in-context learning (no gradient descent)",
        "context_size": min(len(X_train), TABPFN_MAX_SAMPLES),
    }
    return model, meta