"""Regression checks for the data and checkpoint safeguards."""

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.artifacts import save_checkpoint_metadata
from src.data_processing import create_sliding_windows, prepare_data
from src.metrics import calc_nse, calc_r2
from src.model import Seq2SeqAttentionForecaster
from src.train import train_model


class PipelineSafeguardsTest(unittest.TestCase):
    def test_no_attention_keeps_decoder_shapes_and_uniform_weights(self):
        model = Seq2SeqAttentionForecaster(
            input_size=3, hidden_size=8, num_layers=1, pred_len=2,
            dropout=0.0, attention_type="none",
        )
        predictions, weights = model(
            torch.randn(4, 6, 3), torch.randn(4, 1),
        )
        self.assertEqual(tuple(predictions.shape), (4, 2))
        self.assertEqual(tuple(weights.shape), (4, 2, 6))
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 6))

    def test_mean_context_keeps_decoder_shapes_and_uniform_weights(self):
        model = Seq2SeqAttentionForecaster(
            input_size=3, hidden_size=8, num_layers=1, pred_len=2,
            dropout=0.0, attention_type="mean",
        )
        predictions, weights = model(
            torch.randn(4, 6, 3), torch.randn(4, 1),
        )
        self.assertEqual(tuple(predictions.shape), (4, 2))
        self.assertEqual(tuple(weights.shape), (4, 2, 6))
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 6))

    def test_checkpoint_metadata_uses_project_relative_data_path(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(output_dir=Path(directory))
            cfg.ensure_dirs()
            (cfg.output_dir / "checkpoints" / "best_seq2seq_attention.pt").write_bytes(b"test")
            scaler = SimpleNamespace(mean_=np.array([0.0]), scale_=np.array([1.0]))
            path = save_checkpoint_metadata(cfg, {
                "feature_scaler": scaler,
                "target_scaler": scaler,
                "feature_cols": ["cnt"],
                "num_features": 1,
            })
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["data_path"], "data/hour.csv")

    def test_targets_stay_within_their_chronological_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour.csv"
            timestamps = pd.date_range("2020-01-01", periods=80, freq="h")
            pd.DataFrame({
                "dteday": timestamps.strftime("%Y-%m-%d"),
                "hr": timestamps.hour,
                "cnt": np.arange(80, dtype=np.float32),
            }).to_csv(path, index=False)
            cfg = Config(data_path=path, input_len=3, pred_len=2)
            prepared = prepare_data(cfg)
            scaler = prepared["target_scaler"]
            boundaries = {"train": (0, 56), "val": (56, 68), "test": (68, 80)}
            for split, (start, stop) in boundaries.items():
                targets = prepared[split][2]
                original = scaler.inverse_transform(targets.reshape(-1, 1))
                self.assertTrue(np.all(original >= start - 1e-4), split)
                self.assertTrue(np.all(original < stop + 1e-4), split)

    def test_windows_crossing_a_missing_hour_are_excluded(self):
        values = np.arange(10, dtype=np.float32)
        features = values[:, None]
        time = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h"))
        time.iloc[5:] += pd.Timedelta(hours=1)
        x, _, y = create_sliding_windows(
            features, values, 3, 2, time, pd.Timedelta(hours=1)
        )
        self.assertEqual(len(y), 2)
        self.assertEqual(x.shape, (2, 3, 1))

    def test_constant_target_r2_and_nse(self):
        truth = np.array([5.0, 5.0, 5.0])
        self.assertEqual(calc_r2(truth, truth), 1.0)
        self.assertEqual(calc_nse(truth, truth + 1), 0.0)

    def test_zero_epochs_cannot_load_an_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(output_dir=Path(directory), epochs=0)
            with self.assertRaisesRegex(ValueError, "epochs must be at least 1"):
                train_model(None, None, None, cfg)


if __name__ == "__main__":
    unittest.main()
