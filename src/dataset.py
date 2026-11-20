"""
Step 2: PyTorch Dataset and DataLoader Wrapper (src/dataset.py)
"""

from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config


class BikeSeq2SeqDataset(Dataset):
    """
    PyTorch Dataset for Sequence-to-Sequence time-series forecasting:
    - x_enc:  (input_len, num_features) - Historical multivariate sequence fed into the Encoder
    - y_init: (1,)                      - Last observed pred_column value from the Encoder window
    - y_dec:  (pred_len,)               - Next 5 consecutive ground-truth values of pred_column
    """

    def __init__(
        self,
        X: np.ndarray,
        y_init: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_init = torch.tensor(y_init, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_init[idx], self.Y[idx]


def build_dataloaders(
    data_dict: Dict, cfg: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build PyTorch DataLoaders for Train, Validation, and Test splits."""
    train_ds = BikeSeq2SeqDataset(*data_dict["train"])
    val_ds = BikeSeq2SeqDataset(*data_dict["val"])
    test_ds = BikeSeq2SeqDataset(*data_dict["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
