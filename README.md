# ICU Mortality Prediction — 4-Model Comparison

Predicts **in-hospital death** from the first 48 hours of ICU admission data.

## Models

| Model | Paradigm | Key property |
|---|---|---|
| Regularized LogReg (L1/L2) | Classical linear | Interpretable weights, CV-tuned C and penalty |
| XGBoost | Gradient boosted trees | Handles missing values natively, nonlinear interactions |
| FT-Transformer | Deep learning (Transformer) | Dense self-attention over all feature pairs |
| TabPFN v2 | Meta-learned in-context learning | No gradient descent on ICU data — pretrained on synthetic datasets |

## Project Structure

```
icu_mortality/
├── config.yaml              # variable definitions (static vs time-series)
├── requirements.txt
├── data/
│   ├── files/               # per-patient CSV files (e.g. 132539.csv)
│   └── labels.csv           # RecordID, In-hospital_death, 30-day_mortality
├── src/
│   ├── feature_engineering.py   # raw CSV → flat feature matrix
│   ├── preprocessing.py         # model-specific preprocessing pipelines
│   ├── model_logreg.py          # Model 1: Logistic Regression
│   ├── model_xgboost.py         # Model 2: XGBoost
│   ├── model_ft_transformer.py  # Model 3: FT-Transformer
│   ├── model_tabpfn.py          # Model 4: TabPFN v2
│   ├── evaluate.py              # unified metrics + plots
│   └── train_all.py             # main entry point
└── results/                 # output: plots, CSV, saved models
```

## Running

```bash
cd src

# Full run with hyperparameter tuning (recommended)
python train_all.py

# Quick test on 500 patients, no tuning
python train_all.py --max_patients 500 --no_tune

# Skip TabPFN (if not installed) or FT-Transformer (if no GPU)
python train_all.py --skip tabpfn
python train_all.py --skip ft_transformer,tabpfn

# Custom paths
python train_all.py \
    --data_dir /path/to/data/files \
    --labels   /path/to/labels.csv \
    --config   /path/to/config.yaml \
    --results  /path/to/results
```

## Feature Engineering

Each patient's 48-hour record is summarized into a flat feature vector:

**Static features** (5): Age, Gender, Height, ICUType, Weight

**Per time-series variable** (35 variables × 9 features = 315):
- `mean`, `std`, `min`, `max`, `last` — summary statistics
- `count` — number of observations (proxy for clinical attention)
- `mean_0_24`, `mean_24_48` — 24-hour window means (trend signal)
- `missing` — binary flag if never observed

**Total**: ~320 features per patient

## Outputs (in `results/`)

| File | Description |
|---|---|
| `comparison_table.csv` | All metrics for all models |
| `roc_curves.png` | ROC curve overlay |
| `pr_curves.png` | Precision-Recall curve overlay |
| `confusion_matrices.png` | Confusion matrix for each model |
| `logreg_weights.png` | Top LogReg feature weights |
| `*_model.pkl` | Saved fitted models |

## Metrics

- **AUROC** (primary) — area under ROC curve, threshold-independent
- **Avg Precision / PR-AUC** — better for imbalanced classes
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score

## Notes on Class Imbalance

ICU mortality is a rare outcome (~10-15% positive rate). Each model handles this:
- **LogReg**: `class_weight='balanced'`
- **XGBoost**: `scale_pos_weight = n_negative / n_positive`
- **FT-Transformer**: `pos_weight` in BCEWithLogitsLoss
- **TabPFN**: handled internally by the pretrained prior