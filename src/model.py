"""
Step 3: Encoder-Decoder (Sequence-to-Sequence) Network with Attention in PyTorch (src/model.py)
- Encoder: Multi-layer LSTM that encodes the historical input sequence X (batch, input_len, num_features)
           into a sequence of hidden states encoder_outputs (batch, input_len, hidden_size)
           and final recurrent state (hidden, cell).
- Attention: Bahdanau (Additive) or Luong (Multiplicative/General) Attention mechanism
             that dynamically computes alignment weights alpha_s over all Encoder time steps
             at each decoding step s.
- Decoder: Autoregressive multi-layer LSTM Decoder that unrolls for `pred_len = 5` consecutive steps,
           combining the previous step's prediction and the Attention context vector to forecast pred_column.
"""

import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Multi-layer LSTM Encoder for historical multivariate time-series sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, input_len, input_size)
        Returns:
            encoder_outputs: (batch_size, input_len, hidden_size)
            (hidden, cell):  each tensor of shape (num_layers, batch_size, hidden_size)
        """
        encoder_outputs, (hidden, cell) = self.lstm(x)
        return encoder_outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism:
        score(s_{t-1}, h_i) = v^T * tanh(W_s * s_{t-1} + W_h * h_i)
        alpha_{t, i} = softmax(score(s_{t-1}, h_i))
        context_t = sum_i(alpha_{t, i} * h_i)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden:  (batch_size, hidden_size) - Top-layer hidden state of the Decoder
            encoder_outputs: (batch_size, input_len, hidden_size)
        Returns:
            context:      (batch_size, hidden_size)
            attn_weights: (batch_size, input_len)
        """
        # Expand time dimension for decoder query: (batch_size, 1, hidden_size)
        query = self.W_s(decoder_hidden).unsqueeze(1)
        keys = self.W_h(encoder_outputs)  # (batch_size, input_len, hidden_size)

        energy = torch.tanh(query + keys)  # (batch_size, input_len, hidden_size)
        scores = self.v(energy).squeeze(-1)  # (batch_size, input_len)

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, input_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class LuongAttention(nn.Module):
    """
    General multiplicative attention score with a pre-decoder-step query.

    The score has Luong's general form, but this decoder queries with the
    preceding hidden state, so the overall computation is not canonical Luong.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_keys = self.W_a(encoder_outputs)  # (batch_size, input_len, hidden_size)
        scores = torch.bmm(proj_keys, decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class AttentionDecoder(nn.Module):
    """
    Autoregressive LSTM Decoder equipped with an Attention mechanism.
    At each decoding step:
    1. Compute the `context` vector from `decoder_hidden` and `encoder_outputs` via Attention.
    2. Concatenate `[y_prev, context]` and feed into the Decoder LSTM.
    3. Project `[lstm_out, context, y_prev]` through an MLP readout head to predict `y_hat`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_type: str = "bahdanau",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if attention_type.lower() == "none":
            self.attention = None
        elif attention_type.lower() == "luong":
            self.attention = LuongAttention(hidden_size)
        elif attention_type.lower() == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Decoder LSTM input consists of previous scalar prediction (1) + context vector (hidden_size)
        self.lstm = nn.LSTM(
            input_size=1 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward_step(
        self,
        y_prev: torch.Tensor,
        hidden_cell: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Execute a single decoding time step.
        Args:
            y_prev:          (batch_size, 1) - Target value from the preceding step
            hidden_cell:     (hidden, cell), each tensor (num_layers, batch_size, hidden_size)
            encoder_outputs: (batch_size, input_len, hidden_size)
        Returns:
            pred:         (batch_size, 1) - Forecasted target value for the current step
            hidden_cell:  Updated (hidden, cell) state tuple
            attn_weights: (batch_size, input_len) - Attention weights over Encoder steps
        """
        hidden, cell = hidden_cell
        # Both score types use the pre-step top-layer hidden state as query.
        top_hidden = hidden[-1]  # (batch_size, hidden_size)
        if self.attention is None:
            context = encoder_outputs.new_zeros(encoder_outputs.size(0), self.hidden_size)
            attn_weights = encoder_outputs.new_full(
                (encoder_outputs.size(0), encoder_outputs.size(1)),
                1.0 / encoder_outputs.size(1),
            )
        else:
            context, attn_weights = self.attention(top_hidden, encoder_outputs)

        # Concatenate y_prev and context as LSTM input: (batch_size, 1, 1 + hidden_size)
        lstm_input = torch.cat([y_prev, context], dim=-1).unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_out = self.dropout(lstm_out.squeeze(1))  # (batch_size, hidden_size)

        # Readout projection combining LSTM output, attention context, and residual y_prev
        combined = torch.cat([lstm_out, context, y_prev], dim=-1)
        pred = self.fc_out(combined)  # (batch_size, 1)
        return pred, (hidden, cell), attn_weights


class Seq2SeqAttentionForecaster(nn.Module):
    """
    Complete Encoder-Decoder (Sequence-to-Sequence) Network with Attention
    for multi-step forecasting of `pred_len` (5) consecutive time steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 5,
        dropout: float = 0.2,
        attention_type: str = "bahdanau",
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = AttentionDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_type=attention_type,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        y_init: torch.Tensor,
        y_target: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_enc:                 (batch_size, input_len, input_size)
            y_init:                (batch_size, 1) - Last observed pred_column value from Encoder window
            y_target:              (batch_size, pred_len) - Ground-truth targets (for Teacher Forcing)
            teacher_forcing_ratio: Probability of feeding ground-truth target as next step's input
        Returns:
            predictions:  (batch_size, pred_len)
            attentions:   (batch_size, pred_len, input_len)
        """
        encoder_outputs, hidden_cell = self.encoder(x_enc)

        preds = []
        attn_maps = []
        y_prev = y_init  # (batch_size, 1)

        for step in range(self.pred_len):
            pred, hidden_cell, attn_weights = self.decoder.forward_step(
                y_prev, hidden_cell, encoder_outputs
            )
            preds.append(pred)
            attn_maps.append(attn_weights.unsqueeze(1))

            if (
                y_target is not None
                and teacher_forcing_ratio > 0.0
                and random.random() < teacher_forcing_ratio
            ):
                y_prev = y_target[:, step : step + 1]
            else:
                y_prev = pred

        predictions = torch.cat(preds, dim=1)  # (batch_size, pred_len)
        attentions = torch.cat(attn_maps, dim=1)  # (batch_size, pred_len, input_len)
        return predictions, attentions
