"""
train_all.py

End-to-end training and evaluation pipeline for ICU mortality prediction.

Usage:
    python train_all.py --data_dir data/files --labels data/labels.csv \
                        --config config.yaml --results results/

Arguments:
    --data_dir   : directory containing per-patient CSV files
    --labels     : path to labels.csv
    --config     : path to config.yaml
    --results    : output directory for plots and comparison table
    --tune       : run hyperparameter tuning (default: True)
    --no_tune    : skip hyperparameter tuning (faster, uses sensible defaults)
    --ft_epochs  : number of training epochs for FT-Transformer (default: 50)
    --skip       : comma-separated list of models to skip
                   e.g. --skip tabpfn,ft_transformer
    --max_patients : limit number of patients (useful for quick testing)

Pipeline:
  1. Feature extraction from raw per-patient CSV files
  2. Train/test split (80/20 stratified)
  3. Train each model with its own preprocessing + hyperparameter tuning
  4. Evaluate all models on the held-out test set
  5. Save comparison table + plots to results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

# Add src/ to path
# sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import build_feature_matrix
from preprocessing import labels_to_binary
import model_logreg as model_logreg
import model_xgboost as model_xgboost
import model_ft_transformer
import model_tabpfn as model_tabpfn
from evaluate import evaluate_all, plot_feature_importance_logreg


def parse_args():
    p = argparse.ArgumentParser(description="ICU Mortality Prediction — 4-Model Comparison")
    p.add_argument("--data_dir",     default="data/files",    help="Directory with patient CSVs")
    p.add_argument("--labels",       default="data/labels.csv", help="Path to labels.csv")
    p.add_argument("--config",       default="config.yaml",   help="Path to config.yaml")
    p.add_argument("--results",      default="results",       help="Output directory")
    p.add_argument("--tune",         action="store_true",  default=True)
    p.add_argument("--no_tune",      action="store_true",  default=False,
                   help="Skip hyperparameter tuning (faster, uses defaults)")
    p.add_argument("--ft_epochs",    type=int, default=50,    help="FT-Transformer training epochs")
    p.add_argument("--skip",         type=str, default="",
                   help="Comma-separated model keys to skip: logreg,xgboost,ft_transformer,tabpfn")
    p.add_argument("--max_patients", type=int, default=None,
                   help="Limit dataset size (useful for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    tune = not args.no_tune
    skip = set(s.strip().lower() for s in args.skip.split(",") if s.strip())

    print("=" * 65)
    print("  ICU MORTALITY PREDICTION — 4-MODEL COMPARISON")
    print("=" * 65)
    print(f"  Data dir      : {args.data_dir}")
    print(f"  Labels        : {args.labels}")
    print(f"  Tune          : {tune}")
    print(f"  FT epochs     : {args.ft_epochs}")
    print(f"  Skip models   : {skip if skip else 'none'}")
    print("=" * 65)

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    print("\n[Step 1] Extracting features from patient CSV files...")
    X, y_raw = build_feature_matrix(
        data_dir=args.data_dir,
        labels_path=args.labels,
        config_path=args.config,
        max_patients=args.max_patients,
    )

    # Convert labels from {1, -1} to {1, 0}
    y = labels_to_binary(y_raw)
    print(f"\nLabel distribution: positive (died)={y.sum()} ({100*y.mean():.1f}%), "
          f"negative (survived)={(1-y).sum()} ({100*(1-y).mean():.1f}%)")

    # ── 2. Train/test split ───────────────────────────────────────────────────
    print("\n[Step 2] Splitting into train/test (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Train models ───────────────────────────────────────────────────────
    trained_models = {}
    all_meta       = {}

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    if "logreg" not in skip:
        print("\n" + "─"*50)
        print("[Step 3a] Training Logistic Regression...")
        print("─"*50)
        logreg_model, logreg_meta = model_logreg.train(
            X_train, y_train, tune=tune, k=5
        )
        trained_models["LogisticRegression"] = logreg_model
        all_meta["LogisticRegression"]       = logreg_meta

        # Feature weights for the interpretability plot
        feature_names = list(X.columns)
        weights = model_logreg.get_feature_weights(logreg_model, feature_names)
        plot_feature_importance_logreg(weights, top_n=20)
        print(f"\nTop 5 mortality risk features (LogReg):")
        print(weights.head(5).to_string())

    # ── Model 2: XGBoost ──────────────────────────────────────────────────────
    if "xgboost" not in skip:
        print("\n" + "─"*50)
        print("[Step 3b] Training XGBoost...")
        print("─"*50)
        xgb_model, xgb_meta = model_xgboost.train(
            X_train, y_train, tune=tune, n_iter=30 if tune else 0
        )
        trained_models["XGBoost"] = xgb_model
        all_meta["XGBoost"]       = xgb_meta

    # ── Model 3: FT-Transformer ───────────────────────────────────────────────
    if "ft_transformer" not in skip:
        print("\n" + "─"*50)
        print("[Step 3c] Training FT-Transformer...")
        print("─"*50)
        ft_model, ft_meta = model_ft_transformer.train(
            X_train, y_train, tune=tune, n_epochs=args.ft_epochs
        )
        trained_models["FT-Transformer"] = ft_model
        all_meta["FT-Transformer"]       = ft_meta

    # ── Model 4: TabPFN ───────────────────────────────────────────────────────
    if "tabpfn" not in skip:
        print("\n" + "─"*50)
        print("[Step 3d] Running TabPFN (in-context learning)...")
        print("─"*50)
        try:
            tabpfn_model, tabpfn_meta = model_tabpfn.train(X_train, y_train)
            trained_models["TabPFN v2"] = tabpfn_model
            all_meta["TabPFN v2"]       = tabpfn_meta
        except ImportError as e:
            print(f"\n[Warning] {e}")
            print("[TabPFN] Skipping TabPFN — install with: pip install tabpfn")

    if not trained_models:
        print("\nNo models were trained. Check --skip argument.")
        return

    # ── 4. Save models ────────────────────────────────────────────────────────
    print("\n[Step 4] Saving fitted models...")
    os.makedirs(args.results, exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        path = os.path.join(args.results, f"{safe_name}_model.pkl")
        try:
            joblib.dump(model, path)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  [Warning] Could not save {name}: {e}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[Step 5] Evaluating all models on test set...")
    comparison_df = evaluate_all(trained_models, X_test, y_test, save=True)

    # Print training metadata summary
    print("\n[Training Summary]")
    for name, meta in all_meta.items():
        print(f"\n  {name}:")
        for k, v in meta.items():
            if k != "cv_results":
                print(f"    {k}: {v}")

    print("\n" + "="*65)
    print("  DONE. Results saved to:", os.path.abspath(args.results))
    print("="*65)

    return comparison_df


if __name__ == "__main__":
    main()