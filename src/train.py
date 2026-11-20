"""
Step 5: Training & Validation Pipeline for Seq2Seq + Attention (src/train.py)
- Scheduled Teacher Forcing ratio decay across epochs
- Gradient Norm Clipping to prevent exploding gradients in deep LSTMs
- ReduceLROnPlateau learning rate scheduler & Early Stopping with best checkpoint saving
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.model import Seq2SeqAttentionForecaster


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: Seq2SeqAttentionForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
    grad_clip: float,
) -> float:
    """Train the Seq2Seq model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_enc, y_init, y_dec in loader:
        x_enc = x_enc.to(device)
        y_init = y_init.to(device)
        y_dec = y_dec.to(device)

        optimizer.zero_grad()
        preds, _ = model(
            x_enc=x_enc,
            y_init=y_init,
            y_target=y_dec,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss = criterion(preds, y_dec)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = x_enc.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: Seq2SeqAttentionForecaster,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate MSE loss on the Validation set (strictly autoregressive, teacher_forcing_ratio=0.0)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x_enc, y_init, y_dec in loader:
        x_enc = x_enc.to(device)
        y_init = y_init.to(device)
        y_dec = y_dec.to(device)

        preds, _ = model(
            x_enc=x_enc,
            y_init=y_init,
            y_target=None,
            teacher_forcing_ratio=0.0,
        )
        loss = criterion(preds, y_dec)

        batch_size = x_enc.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_model(
    model: Seq2SeqAttentionForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> Tuple[Seq2SeqAttentionForecaster, Dict[str, List[float]]]:
    """
    Full training loop with learning rate scheduling, early stopping, and best checkpoint restoration.
    """
    cfg.ensure_dirs()
    device = torch.device(cfg.device)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = cfg.output_dir / "checkpoints" / "best_seq2seq_attention.pt"

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    print(
        f"\n[Train] Starting training on {device} | "
        f"Max Epochs: {cfg.epochs} | Batch Size: {cfg.batch_size}"
    )
    print("-" * 72)

    for epoch in range(1, cfg.epochs + 1):
        # Linearly decay Teacher Forcing ratio to transition smoothly to autoregressive inference
        tf_ratio = max(
            0.0, cfg.teacher_forcing_ratio * (1.0 - (epoch - 1) / max(cfg.epochs, 1))
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=tf_ratio,
            grad_clip=cfg.grad_clip,
        )
        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            improved = " * [Best saved]"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"TF: {tf_ratio:.2f} | LR: {lr_now:.6f} | "
            f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}{improved}"
        )

        if patience_counter >= cfg.patience:
            print(
                f"\n[Early Stopping] Validation loss did not improve for {cfg.patience} epochs. Stopping at epoch {epoch}."
            )
            break

    print("-" * 72)
    print(f"[Train] Finished! Restoring best weights (Val MSE = {best_val_loss:.4f}) from: {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    return model, history
