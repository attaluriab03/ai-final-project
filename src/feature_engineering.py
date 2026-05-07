"""
feature_engineering.py

Converts raw per-patient CSV files into a flat feature matrix.

Each patient CSV has three columns: Time, Variable, Value.
  - Static variables (Age, Gender, Height, ICUType, Weight) appear at time 00:00.
    Unknown values are encoded as -1 in the raw data → converted to NaN here.
  - Time-series variables have one or more timestamped observations.

Feature extraction strategy (richer than the original project):
  For each time-series variable we extract:
    - mean, std, min, max, last observed value
    - count of observations (proxy for measurement frequency / clinical attention)
    - mean over first 24h vs second 24h (trend signal)
  Additionally:
    - missingness indicator per time-series variable (1 if never observed)

This gives a much richer feature set than the project's simple max aggregation,
while remaining fully interpretable.
"""

import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_time_to_hours(time_str: str) -> float:
    """Convert 'HH:MM' string to fractional hours."""
    h, m = time_str.strip().split(":")
    return int(h) + int(m) / 60.0


def extract_features_single(df: pd.DataFrame, static_vars: list, ts_vars: list) -> dict:
    """
    Extract feature dictionary from a single patient DataFrame.

    Parameters
    ----------
    df : pd.DataFrame with columns [Time, Variable, Value]
    static_vars : list of static variable names
    ts_vars : list of time-series variable names

    Returns
    -------
    dict mapping feature_name -> value (NaN where missing)
    """
    features = {}

    # ── Static variables ──────────────────────────────────────────────────────
    static_rows = df[df["Time"] == "00:00"]
    for var in static_vars:
        row = static_rows[static_rows["Variable"] == var]
        if len(row) == 0:
            features[var] = np.nan
        else:
            val = float(row["Value"].iloc[0])
            # -1 encodes unknown in the raw data
            features[var] = np.nan if val == -1.0 else val

    # ── Time-series variables ─────────────────────────────────────────────────
    # Add fractional hours column for windowing
    ts_df = df[df["Time"] != "00:00"].copy()
    if len(ts_df) > 0:
        ts_df["hours"] = ts_df["Time"].apply(parse_time_to_hours)
        ts_df["Value"] = pd.to_numeric(ts_df["Value"], errors="coerce")

    for var in ts_vars:
        var_df = ts_df[ts_df["Variable"] == var] if len(ts_df) > 0 else pd.DataFrame()
        vals = var_df["Value"].dropna() if len(var_df) > 0 else pd.Series(dtype=float)

        if len(vals) == 0:
            # Variable never observed → all stats NaN, missingness flag = 1
            features[f"{var}_mean"]    = np.nan
            features[f"{var}_std"]     = np.nan
            features[f"{var}_min"]     = np.nan
            features[f"{var}_max"]     = np.nan
            features[f"{var}_last"]    = np.nan
            features[f"{var}_count"]   = 0.0
            features[f"{var}_mean_0_24"] = np.nan
            features[f"{var}_mean_24_48"] = np.nan
            features[f"{var}_missing"] = 1.0
        else:
            features[f"{var}_mean"]  = float(vals.mean())
            features[f"{var}_std"]   = float(vals.std()) if len(vals) > 1 else 0.0
            features[f"{var}_min"]   = float(vals.min())
            features[f"{var}_max"]   = float(vals.max())
            features[f"{var}_last"]  = float(vals.iloc[-1])
            features[f"{var}_count"] = float(len(vals))
            features[f"{var}_missing"] = 0.0

            # 24-hour window means
            hrs = var_df["hours"]
            w1 = var_df.loc[hrs <= 24, "Value"].dropna()
            w2 = var_df.loc[hrs > 24,  "Value"].dropna()
            features[f"{var}_mean_0_24"]  = float(w1.mean()) if len(w1) > 0 else np.nan
            features[f"{var}_mean_24_48"] = float(w2.mean()) if len(w2) > 0 else np.nan

    return features


def build_feature_matrix(
    data_dir: str,
    labels_path: str,
    config_path: str,
    max_patients: int = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load all patient files, extract features, and align with labels.

    Parameters
    ----------
    data_dir    : path to folder containing per-patient CSV files
    labels_path : path to labels.csv
    config_path : path to config.yaml
    max_patients: optional limit (useful for quick testing)

    Returns
    -------
    X : pd.DataFrame  — feature matrix, one row per patient
    y : pd.Series     — binary labels {1, -1} for in-hospital death
    """
    config = load_config(config_path)
    static_vars = config["static"]
    ts_vars     = config["timeseries"]

    labels_df = pd.read_csv(labels_path)
    # Drop patients with missing in-hospital death label
    labels_df = labels_df.dropna(subset=["In-hospital_death"])
    labels_df["RecordID"] = labels_df["RecordID"].astype(int)

    if max_patients is not None:
        labels_df = labels_df.head(max_patients)

    records = []
    record_ids = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
        rid = int(row["RecordID"])
        filepath = os.path.join(data_dir, f"{rid}.csv")

        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)
        feat = extract_features_single(df, static_vars, ts_vars)
        records.append(feat)
        record_ids.append(rid)

    X = pd.DataFrame(records, index=record_ids)
    y = labels_df.set_index("RecordID").loc[record_ids, "In-hospital_death"]
    y = y.astype(int)  # already {1, -1}

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Class distribution   : {y.value_counts().to_dict()}")
    print(f"Missing values (%)   : {X.isna().mean().mean()*100:.1f}% overall")

    return X, y


if __name__ == "__main__":
    # Quick smoke test on the 7 sample files
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    X, y = build_feature_matrix(
        data_dir="/mnt/user-data/uploads",
        labels_path="data/labels.csv",
        config_path="config.yaml",
    )
    print(X.head())
    print(y.head())