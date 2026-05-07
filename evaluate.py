"""
evaluate.py

Unified evaluation for all four models.

Computes the full suite of clinical + ML metrics and produces:
  1. A comparison table (saved to results/comparison_table.csv)
  2. ROC curve overlay for all models (results/roc_curves.png)
  3. Precision-Recall curve overlay (results/pr_curves.png)
  4. Confusion matrices (results/confusion_matrices.png)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
import os


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Consistent colour palette across all plots
MODEL_COLORS = {
    "LogisticRegression": "#2196F3",   # blue
    "XGBoost":            "#4CAF50",   # green
    "FT-Transformer":     "#FF9800",   # orange
    "TabPFN v2":          "#9C27B0",   # purple
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for a single model.

    Parameters
    ----------
    y_true : ground truth labels {0, 1}
    y_pred : binary predictions {0, 1}
    y_prob : predicted probability of positive class

    Returns
    -------
    dict of metric_name -> value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Precision":         precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity":       specificity,
        "F1-Score":          f1_score(y_true, y_pred, zero_division=0),
        "AUROC":             roc_auc_score(y_true, y_prob),
        "Avg Precision (PR-AUC)": average_precision_score(y_true, y_prob),
    }


def evaluate_all(
    models: dict,          # {model_name: fitted_model}
    X_test: pd.DataFrame,
    y_test: np.ndarray,    # {0, 1}
    save: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all models on the test set and produce comparison outputs.

    Parameters
    ----------
    models  : dict mapping model name to fitted classifier
    X_test  : test feature matrix
    y_test  : test labels {0, 1}
    save    : whether to save plots and CSV to results/

    Returns
    -------
    pd.DataFrame with one row per model, one column per metric
    """
    results = {}
    probs   = {}
    preds   = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = compute_metrics(y_test, pred, prob)
        results[name] = metrics
        probs[name]   = prob
        preds[name]   = pred

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"

    # Format for display
    display_df = comparison_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map("{:.4f}".format)

    print("\n" + "="*70)
    print("MODEL COMPARISON — TEST SET PERFORMANCE")
    print("="*70)
    print(comparison_df.to_string(float_format="{:.4f}".format))
    print("="*70)

    if save:
        comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
        _plot_roc_curves(models, probs, y_test)
        _plot_pr_curves(models, probs, y_test)
        _plot_confusion_matrices(models, preds, y_test)
        print(f"\nResults saved to: {RESULTS_DIR}/")

    return comparison_df


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _get_color(name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#607D8B"


def _plot_roc_curves(models: dict, probs: dict, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUROC = 0.50)")

    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auroc = roc_auc_score(y_test, probs[name])
        ax.plot(fpr, tpr, color=_get_color(name), lw=2,
                label=f"{name} (AUROC = {auroc:.3f})")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — ICU In-Hospital Mortality Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plots] ROC curves saved.")


def _plot_pr_curves(models: dict, probs: dict, y_test: np.ndarray):
    baseline = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Random baseline (AP = {baseline:.3f})")

    for name in models:
        prec, rec, _ = precision_recall_curve(y_test, probs[name])
        ap = average_precision_score(y_test, probs[name])
        ax.plot(rec, prec, color=_get_color(name), lw=2,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — ICU In-Hospital Mortality", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plots] PR curves saved.")


def _plot_confusion_matrices(models: dict, preds: dict, y_test: np.ndarray):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, preds[name])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Survived", "Died"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — ICU In-Hospital Mortality", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plots] Confusion matrices saved.")


def plot_feature_importance_logreg(weights: pd.Series, top_n: int = 20):
    """Bar chart of top LogReg feature weights (positive = mortality risk)."""
    top = weights.head(top_n)
    colors = ["#F44336" if w > 0 else "#2196F3" for w in top.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Weight magnitude", fontsize=11)
    ax.set_title(f"Top {top_n} LogReg Feature Weights\n(red = increases mortality risk)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "logreg_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plots] LogReg weights saved.")