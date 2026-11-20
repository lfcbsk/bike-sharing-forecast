"""
Step 1: Data Processing (src/data_processing.py)
- Load raw CSV dataset (hour.csv or day.csv)
- Sort chronologically and impute any missing values
- Engineer cyclical time features (sin/cos transforms for hour, month, weekday)
- Perform strict chronological Train / Validation / Test split
- Standardize features using StandardScaler fitted ONLY on the Training split (preventing data leakage)
- Construct causal sliding windows for Sequence-to-Sequence 5-step-ahead forecasting
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config


def load_and_engineer_features(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the raw CSV file, create cyclical (sin/cos) temporal features, drop leakage columns,
    and place `pred_column` as the final column in the returned DataFrame.
    """
    df = pd.read_csv(cfg.data_path)

    # Ensure strict chronological ordering
    sort_cols = ["dteday", "hr"] if "hr" in df.columns else ["dteday"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Defensive imputation if any missing values exist
    if df.isna().sum().sum() > 0:
        df = df.ffill().bfill()

    # Cyclical encoding for periodic temporal variables
    if "hr" in df.columns:
        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24.0)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24.0)

    if "mnth" in df.columns:
        df["mnth_sin"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12.0)
        df["mnth_cos"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12.0)

    if "weekday" in df.columns:
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)

    if cfg.pred_column not in df.columns:
        raise ValueError(
            f"Target column '{cfg.pred_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop non-feature / leakage columns while preserving pred_column
    cols_to_drop = [
        c for c in cfg.drop_columns if c in df.columns and c != cfg.pred_column
    ]
    df_clean = df.drop(columns=cols_to_drop)

    # Move pred_column to the last column index for convenient slicing
    other_cols = [c for c in df_clean.columns if c != cfg.pred_column]
    feature_cols = other_cols + [cfg.pred_column]
    df_clean = df_clean[feature_cols].astype(np.float32)

    return df_clean, feature_cols


def create_sliding_windows(
    features: np.ndarray,
    targets: np.ndarray,
    input_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Sequence-to-Sequence sliding windows (X, y_init, Y):
    - Indexing Convention A (history ends at t-1, forecast starts from current step t):
        Encoder input:  [t - input_len, ..., t - 1]  (length = input_len)
        Decoder target: [t, t + 1, t + 2, t + 3, t + 4] (5 consecutive steps starting from t)
    - Indexing Convention B (history ends at current step t, forecast next 5 future steps):
        Encoder input:  [t - input_len + 1, ..., t]
        Decoder target: [t + 1, t + 2, t + 3, t + 4, t + 5]
      Both conventions yield the exact same causal window pair (X_seq, Y_seq) with zero look-ahead leakage.

    Returns:
        X:      (N, input_len, num_features) - Historical multivariate sequence for the Encoder
        y_init: (N, 1)                       - Last observed target value at the end of Encoder window
                                               used as the initial Decoder input token (y_0)
        Y:      (N, pred_len)                - Ground-truth target values for the next 5 consecutive steps
    """
    total_len = len(features)
    num_samples = total_len - input_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Dataset length ({total_len}) is smaller than input_len ({input_len}) + pred_len ({pred_len})."
        )

    X_list, y_init_list, Y_list = [], [], []
    for i in range(num_samples):
        enc_end = i + input_len
        dec_end = enc_end + pred_len

        X_list.append(features[i:enc_end])
        y_init_list.append(targets[enc_end - 1 : enc_end])
        Y_list.append(targets[enc_end:dec_end])

    X = np.array(X_list, dtype=np.float32)
    y_init = np.array(y_init_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, y_init, Y


def prepare_data(cfg: Config) -> Dict:
    """
    Execute the complete data preparation pipeline:
    1. Load CSV & engineer cyclical features
    2. Split chronologically into Train / Validation / Test sets
    3. Fit StandardScaler strictly on Train and transform Train / Val / Test
    4. Generate Seq2Seq sliding windows for each split
    """
    df, feature_cols = load_and_engineer_features(cfg)
    values = df.values  # Shape: (T, num_features), last column is pred_column

    n_total = len(values)
    n_train = int(n_total * cfg.train_ratio)
    n_val = int(n_total * (cfg.train_ratio + cfg.val_ratio))

    train_raw = values[:n_train]
    # Include `input_len` preceding context steps for Val and Test so the first target window
    # starts right at the split boundary without leaking any target labels across splits.
    val_raw = values[n_train - cfg.input_len : n_val]
    test_raw = values[n_val - cfg.input_len :]

    # Standardize all input features (fit on training data only)
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_raw)

    # Dedicated scaler for pred_column to inverse_transform predictions back to original scale
    target_scaler = StandardScaler()
    target_scaler.fit(train_raw[:, -1:])

    train_scaled = feature_scaler.transform(train_raw)
    val_scaled = feature_scaler.transform(val_raw)
    test_scaled = feature_scaler.transform(test_raw)

    train_target_scaled = target_scaler.transform(train_raw[:, -1:]).squeeze(-1)
    val_target_scaled = target_scaler.transform(val_raw[:, -1:]).squeeze(-1)
    test_target_scaled = target_scaler.transform(test_raw[:, -1:]).squeeze(-1)

    X_train, y_init_train, Y_train = create_sliding_windows(
        train_scaled, train_target_scaled, cfg.input_len, cfg.pred_len
    )
    X_val, y_init_val, Y_val = create_sliding_windows(
        val_scaled, val_target_scaled, cfg.input_len, cfg.pred_len
    )
    X_test, y_init_test, Y_test = create_sliding_windows(
        test_scaled, test_target_scaled, cfg.input_len, cfg.pred_len
    )

    return {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train": (X_train, y_init_train, Y_train),
        "val": (X_val, y_init_val, Y_val),
        "test": (X_test, y_init_test, Y_test),
    }
