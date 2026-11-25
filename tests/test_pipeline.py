"""Regression checks for the data and checkpoint safeguards."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_processing import create_sliding_windows
from src.metrics import calc_nse, calc_r2
from src.train import train_model


class PipelineSafeguardsTest(unittest.TestCase):
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
