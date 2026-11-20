"""
Step 4: Evaluation Metrics for Sequence-to-Sequence Forecasting (src/metrics.py)
Includes:
1. R² (Coefficient of Determination & Squared Pearson Correlation Coefficient)
2. NSE (Nash–Sutcliffe Efficiency)
3. MAE (Mean Absolute Error)
4. RMSE (Root Mean Squared Error)

How these metrics are applied to Sequence-to-Sequence (H = 5 steps) models:
- Per-Step (Horizon-Specific) Evaluation: Computed independently for each forecast horizon
  h in {t+1, t+2, t+3, t+4, t+5} across all N test windows to quantify error growth over time.
- Macro-Average Evaluation: Unweighted mean of the metric across the 5 forecast horizons.
- Overall (Pooled / Flattened) Evaluation: Computed across all N x 5 predictions flattened together.
"""

from typing import Dict
import numpy as np


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error:
        MAE = (1 / N) * sum(|y_i - y_hat_i|)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error:
        RMSE = sqrt((1 / N) * sum((y_i - y_hat_i)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash–Sutcliffe Model Efficiency Coefficient:
        NSE = 1 - [ sum((y_i - y_hat_i)^2) / sum((y_i - mean(y))^2) ]
    Range: (-inf, 1.0]. NSE = 1.0 indicates a perfect forecast; NSE = 0.0 indicates that
    the model is no more accurate than predicting the observed mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard Coefficient of Determination (R²):
        R² = 1 - [ sum((y_i - y_hat_i)^2) / sum((y_i - mean(y))^2) ]
    Note: Under standard regression definitions (e.g., sklearn.metrics.r2_score),
    the formula for R² is mathematically identical to NSE.
    """
    return calc_nse(y_true, y_pred)


def calc_r2_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Squared Pearson Correlation Coefficient (r²):
    In hydrology and environmental forecasting literature where both R² and NSE are reported
    side-by-side:
    - R² is often defined as the squared Pearson correlation coefficient (measuring phase
      alignment and linear collinearity, invariant to additive/multiplicative bias).
    - NSE measures exact 1:1 agreement around the identity line (penalizing bias and scale errors).
    """
    y_t = y_true.reshape(-1)
    y_p = y_pred.reshape(-1)
    t_diff = y_t - np.mean(y_t)
    p_diff = y_p - np.mean(y_p)
    denom = np.sqrt(np.sum(t_diff ** 2) * np.sum(p_diff ** 2))
    if denom == 0:
        return 0.0
    r = np.sum(t_diff * p_diff) / denom
    return float(r ** 2)


def evaluate_seq2seq_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict:
    """
    Compute R², R² (Pearson), NSE, MAE, and RMSE for Seq2Seq outputs of shape (N, pred_len).

    Args:
        y_true: (N, pred_len) - Ground-truth values inverse-transformed to original scale
        y_pred: (N, pred_len) - Predicted values inverse-transformed to original scale
    Returns:
        Dictionary containing Overall (pooled), Macro-Average, and Per-Step (step_1..step_H) metrics.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}!"
        )

    pred_len = y_true.shape[1]
    per_step = {}

    for step in range(pred_len):
        yt_s = y_true[:, step]
        yp_s = y_pred[:, step]
        per_step[f"step_{step + 1}"] = {
            "R2": calc_r2(yt_s, yp_s),
            "R2_Pearson": calc_r2_pearson(yt_s, yp_s),
            "NSE": calc_nse(yt_s, yp_s),
            "MAE": calc_mae(yt_s, yp_s),
            "RMSE": calc_rmse(yt_s, yp_s),
        }

    overall = {
        "R2": calc_r2(y_true.reshape(-1), y_pred.reshape(-1)),
        "R2_Pearson": calc_r2_pearson(y_true.reshape(-1), y_pred.reshape(-1)),
        "NSE": calc_nse(y_true.reshape(-1), y_pred.reshape(-1)),
        "MAE": calc_mae(y_true.reshape(-1), y_pred.reshape(-1)),
        "RMSE": calc_rmse(y_true.reshape(-1), y_pred.reshape(-1)),
    }

    macro_avg = {
        metric: float(np.mean([per_step[f"step_{s + 1}"][metric] for s in range(pred_len)]))
        for metric in ["R2", "R2_Pearson", "NSE", "MAE", "RMSE"]
    }

    return {
        "overall": overall,
        "macro_average": macro_avg,
        "per_step": per_step,
    }
