from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch


@dataclass
class Config:
    """Configuration for the Seq2Seq with Attention time-series forecasting pipeline."""

    # 1. File paths & output directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "hour.csv")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")

    # 2. Time-Series Forecasting setup
    pred_column: str = "cnt"          # Target column to forecast (can be 'cnt', 'casual', or 'registered')
    input_len: int = 24               # Encoder input sequence length (24 hours = 1 full day of history)
    pred_len: int = 5                 # Forecast horizon: next 5 consecutive time steps

    # Chronological dataset split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1.0 - train_ratio - val_ratio = 0.15

    # Columns dropped to prevent data leakage when forecasting 'cnt'
    # Note: cnt = casual + registered, so casual and registered must be excluded from input features
    drop_columns: List[str] = field(
        default_factory=lambda: ["instant", "dteday", "casual", "registered"]
    )

    # 3. Encoder-Decoder + Attention architecture hyperparameters
    hidden_size: int = 64             # Hidden state dimension for Encoder & Decoder LSTMs
    num_layers: int = 2               # Number of stacked LSTM layers
    dropout: float = 0.2              # Dropout probability for regularization
    attention_type: str = "bahdanau"  # 'bahdanau' (Additive) or 'luong' (Multiplicative/General)

    # 4. Training hyperparameters
    batch_size: int = 128
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    teacher_forcing_ratio: float = 0.5
    grad_clip: float = 1.0
    patience: int = 8                 # Early stopping patience
    seed: int = 42

    # Execution device (GPU if available, otherwise CPU)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def ensure_dirs(self) -> None:
        """Create output subdirectories if they do not exist."""
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
