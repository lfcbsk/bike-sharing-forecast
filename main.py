"""
End-to-End Execution Script (main.py):
  Step 1: Data preprocessing, feature engineering & sliding windows (src/data_processing.py)
  Step 2: PyTorch Dataset & DataLoader construction (src/dataset.py)
  Step 3: Encoder-Decoder + Attention model initialization (src/model.py)
  Step 4 & 5: Model training, validation & checkpointing (src/train.py)
  Step 6: Test set evaluation with R², NSE, MAE, RMSE & diagnostic plots (src/evaluate.py)
"""

import argparse
from pathlib import Path
import torch

from src.config import Config
from src.data_processing import prepare_data
from src.dataset import build_dataloaders
from src.evaluate import predict_dataset, print_and_save_metrics, save_plots
from src.metrics import evaluate_seq2seq_predictions
from src.model import Seq2SeqAttentionForecaster
from src.train import set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seq2Seq + Attention Bike Sharing Forecasting (Next 5 Consecutive Time Steps)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hour.csv",
        help="Path to the input CSV dataset (default: data/hour.csv)",
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default="cnt",
        help="Target column to forecast (pred_column), default: cnt",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=24,
        help="Historical sequence length fed into the Encoder (default: 24 steps)",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=5,
        help="Number of consecutive future time steps to forecast (default: 5)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden state dimension of Encoder and Decoder LSTMs (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of stacked LSTM layers in Encoder and Decoder (default: 2)",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="bahdanau",
        choices=["bahdanau", "luong"],
        help="Attention mechanism type: 'bahdanau' or 'luong' (default: bahdanau)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Maximum number of training epochs (default: 35)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints, metrics, and figures (default: outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        pred_column=args.pred_column,
        input_len=args.input_len,
        pred_len=args.pred_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attention_type=args.attention,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    set_seed(cfg.seed)

    print("=" * 72)
    print("STEP 1: DATA PREPROCESSING & SLIDING WINDOW CONSTRUCTION")
    print("=" * 72)
    data_dict = prepare_data(cfg)
    print(f"- Dataset path        : {cfg.data_path}")
    print(f"- Target column       : {cfg.pred_column}")
    print(f"- Input features ({data_dict['num_features']}) : {data_dict['feature_cols']}")
    print(f"- Train split shapes  : X={data_dict['train'][0].shape}, Y={data_dict['train'][2].shape}")
    print(f"- Val split shapes    : X={data_dict['val'][0].shape}, Y={data_dict['val'][2].shape}")
    print(f"- Test split shapes   : X={data_dict['test'][0].shape}, Y={data_dict['test'][2].shape}")

    print("\n" + "=" * 72)
    print("STEP 2: PYTORCH DATASET & DATALOADER INITIALIZATION")
    print("=" * 72)
    train_loader, val_loader, test_loader = build_dataloaders(data_dict, cfg)
    print(f"- Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    print("\n" + "=" * 72)
    print("STEP 3: BUILDING ENCODER-DECODER + ATTENTION NETWORK")
    print("=" * 72)
    model = Seq2SeqAttentionForecaster(
        input_size=data_dict["num_features"],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        pred_len=cfg.pred_len,
        dropout=cfg.dropout,
        attention_type=cfg.attention_type,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Attention mechanism : {cfg.attention_type.capitalize()} Attention")
    print(f"- Trainable parameters: {num_params:,}")

    print("\n" + "=" * 72)
    print("STEP 4: TRAINING & VALIDATION")
    print("=" * 72)
    model, history = train_model(model, train_loader, val_loader, cfg)

    print("\n" + "=" * 72)
    print("STEP 5: TEST SET EVALUATION (R², NSE, MAE, RMSE) & VISUALIZATION")
    print("=" * 72)
    device = torch.device(cfg.device)
    y_true, y_pred, attentions = predict_dataset(
        model=model,
        loader=test_loader,
        target_scaler=data_dict["target_scaler"],
        device=device,
    )
    metrics_report = evaluate_seq2seq_predictions(y_true, y_pred)
    print_and_save_metrics(metrics_report, cfg)
    save_plots(history, y_true, y_pred, attentions, cfg)


if __name__ == "__main__":
    main()
