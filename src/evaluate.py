"""
Step 6: Test Set Evaluation, Metric Reporting (R², NSE, MAE, RMSE), and Visualization (src/evaluate.py)
- Inverse-transform scaled predictions back to original bike rental counts
- Compute R², R² (Pearson), NSE, MAE, and RMSE per forecast horizon (t+1 .. t+5), Macro-Average, and Overall
- Export reports to CSV and JSON, and generate 3 diagnostic plots:
  1. Training vs. Validation MSE Loss curve
  2. Multi-step Forecast vs. Ground Truth comparison on the Test set
  3. Average Attention Weights Heatmap (Decoder Forecast Steps vs. Encoder Historical Steps)
"""

import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.metrics import evaluate_seq2seq_predictions
from src.model import Seq2SeqAttentionForecaster


@torch.no_grad()
def predict_dataset(
    model: Seq2SeqAttentionForecaster,
    loader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run autoregressive inference across a DataLoader and inverse-transform predictions to the original scale.
    Returns:
        y_true_orig: (N, pred_len)
        y_pred_orig: (N, pred_len)
        attentions:  (N, pred_len, input_len)
    """
    model.eval()
    preds_list, trues_list, attn_list = [], [], []

    for x_enc, y_init, y_dec in loader:
        x_enc = x_enc.to(device)
        y_init = y_init.to(device)

        preds, attn = model(
            x_enc=x_enc,
            y_init=y_init,
            y_target=None,
            teacher_forcing_ratio=0.0,
        )
        preds_list.append(preds.cpu().numpy())
        trues_list.append(y_dec.numpy())
        attn_list.append(attn.cpu().numpy())

    y_pred_scaled = np.concatenate(preds_list, axis=0)  # (N, pred_len)
    y_true_scaled = np.concatenate(trues_list, axis=0)  # (N, pred_len)
    attentions = np.concatenate(attn_list, axis=0)      # (N, pred_len, input_len)

    # Inverse-transform back to original physical units (rental bike counts)
    n_samples, pred_len = y_pred_scaled.shape
    y_pred_orig = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(n_samples, pred_len)
    # Clip negative predictions since bike rental counts are non-negative
    y_pred_orig = np.clip(y_pred_orig, a_min=0.0, a_max=None)

    y_true_orig = target_scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)
    ).reshape(n_samples, pred_len)

    return y_true_orig, y_pred_orig, attentions


def print_and_save_metrics(metrics_report: Dict, cfg: Config) -> pd.DataFrame:
    """Print the formatted R², NSE, MAE, RMSE table and save results to CSV and JSON."""
    cfg.ensure_dirs()

    rows = []
    for step_idx in range(1, cfg.pred_len + 1):
        step_key = f"step_{step_idx}"
        m = metrics_report["per_step"][step_key]
        rows.append(
            {
                "Horizon": f"Step t+{step_idx}",
                "R2": m["R2"],
                "R2_Pearson": m["R2_Pearson"],
                "NSE": m["NSE"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
            }
        )

    macro = metrics_report["macro_average"]
    rows.append(
        {
            "Horizon": "Macro Average (5 steps)",
            "R2": macro["R2"],
            "R2_Pearson": macro["R2_Pearson"],
            "NSE": macro["NSE"],
            "MAE": macro["MAE"],
            "RMSE": macro["RMSE"],
        }
    )

    overall = metrics_report["overall"]
    rows.append(
        {
            "Horizon": "Overall (All 5 steps pooled)",
            "R2": overall["R2"],
            "R2_Pearson": overall["R2_Pearson"],
            "NSE": overall["NSE"],
            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
        }
    )

    df_metrics = pd.DataFrame(rows)

    csv_path = cfg.output_dir / "metrics" / "test_metrics.csv"
    json_path = cfg.output_dir / "metrics" / "test_metrics.json"
    df_metrics.to_csv(csv_path, index=False, float_format="%.4f")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 82)
    print(
        f"SEQ2SEQ + ATTENTION EVALUATION RESULTS (Target Column: '{cfg.pred_column}')"
    )
    print("=" * 82)
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 82)
    print(f"[Metrics] Saved evaluation reports to: {csv_path} and {json_path}")
    return df_metrics


def save_plots(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attentions: np.ndarray,
    cfg: Config,
) -> None:
    """Generate and save diagnostic plots for model evaluation."""
    cfg.ensure_dirs()
    fig_dir = cfg.output_dir / "figures"

    # 1. Training & Validation Loss Curve
    plt.figure(figsize=(8, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train MSE Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation MSE Loss")
    plt.title("Training & Validation Loss Curve (Seq2Seq + Attention)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Standardized Scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=150)
    plt.close()

    # 2. Actual vs. Predicted Comparison at Step t+1 and Step t+5 (168 hours = 1 week on Test Set)
    num_plot = min(168, len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    axes[0].plot(y_true[:num_plot, 0], label="Actual", color="#1f77b4", linewidth=1.8)
    axes[0].plot(
        y_pred[:num_plot, 0],
        label="Predicted (Step t+1)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.6,
    )
    axes[0].set_title(f"Forecast Comparison on Test Set - Step t+1 ({cfg.pred_column})")
    axes[0].set_ylabel(cfg.pred_column)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(
        y_true[:num_plot, -1], label="Actual", color="#1f77b4", linewidth=1.8
    )
    axes[1].plot(
        y_pred[:num_plot, -1],
        label=f"Predicted (Step t+{cfg.pred_len})",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.6,
    )
    axes[1].set_title(
        f"Forecast Comparison on Test Set - Step t+{cfg.pred_len} ({cfg.pred_column})"
    )
    axes[1].set_xlabel("Time Steps (Hours in Test Sample Window)")
    axes[1].set_ylabel(cfg.pred_column)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_comparison.png", dpi=150)
    plt.close()

    # 3. Average Attention Weights Heatmap across the Test Set
    mean_attn = np.mean(attentions, axis=0)  # (pred_len, input_len)
    plt.figure(figsize=(10, 4))
    im = plt.imshow(mean_attn, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Attention Weight")
    plt.yticks(
        range(cfg.pred_len),
        [f"t+{i + 1}" for i in range(cfg.pred_len)],
    )
    plt.xticks(
        range(cfg.input_len),
        [f"t-{cfg.input_len - 1 - i}" for i in range(cfg.input_len)],
        rotation=45,
    )
    plt.xlabel("Encoder Historical Time Steps")
    plt.ylabel("Decoder Forecast Horizon")
    plt.title("Average Attention Weights Heatmap (Test Set)")
    plt.tight_layout()
    plt.savefig(fig_dir / "attention_heatmap.png", dpi=150)
    plt.close()

    print(f"[Figures] Saved diagnostic visualizations to: {fig_dir}")
