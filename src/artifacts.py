"""Save the inputs needed to reconstruct a trained forecasting model."""

import hashlib
import json
from pathlib import Path
from typing import Dict

from src.config import Config


def save_checkpoint_metadata(cfg: Config, data_dict: Dict) -> Path:
    """Write architecture, feature order, and fitted scaler parameters as JSON."""
    cfg.ensure_dirs()
    data_hash = hashlib.sha256(cfg.data_path.read_bytes()).hexdigest()
    checkpoint_path = cfg.output_dir / "checkpoints" / "best_seq2seq_attention.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    feature_scaler = data_dict["feature_scaler"]
    target_scaler = data_dict["target_scaler"]
    metadata = {
        "checkpoint": "best_seq2seq_attention.pt",
        "checkpoint_sha256": checkpoint_hash,
        "data_path": str(cfg.data_path),
        "data_sha256": data_hash,
        "pred_column": cfg.pred_column,
        "seed": cfg.seed,
        "feature_cols": data_dict["feature_cols"],
        "model": {
            "input_size": data_dict["num_features"],
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "pred_len": cfg.pred_len,
            "dropout": cfg.dropout,
            "attention_type": cfg.attention_type,
        },
        "preprocessing": {
            "input_len": cfg.input_len,
            "time_gap_policy": "drop_crossing_windows",
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "drop_columns": cfg.drop_columns,
            "feature_scaler_mean": feature_scaler.mean_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
            "target_scaler_mean": target_scaler.mean_.tolist(),
            "target_scaler_scale": target_scaler.scale_.tolist(),
        },
    }
    path = cfg.output_dir / "checkpoints" / "best_seq2seq_attention.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
