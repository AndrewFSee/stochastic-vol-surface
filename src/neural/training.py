"""Walk-forward training loop for the NeuralVolModel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.neural.neural_vol_model import NeuralVolModel
from src.neural.losses import weighted_mse_loss, arbitrage_penalty

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: NeuralVolModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ts: torch.Tensor,
    atm_weight: float = 3.0,
    arb_lambda: float = 0.1,
    device: torch.device | None = None,
) -> float:
    device = device or torch.device("cpu")
    model.train()
    total_loss = 0.0

    for batch in loader:
        snapshots, target_iv = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        pred = model(snapshots, ts.to(device))
        # Use last predicted time step vs target
        pred_last = pred[:, -1, :, :]

        loss = weighted_mse_loss(pred_last, target_iv, atm_weight=atm_weight)
        loss += arb_lambda * arbitrage_penalty(pred_last)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def walk_forward_train(
    surfaces: torch.Tensor,         # (N, 1, n_k, n_t)  – time-ordered
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_window: int = 252,
    val_window: int = 63,
    checkpoint_dir: Optional[Path] = None,
    device: torch.device | None = None,
) -> NeuralVolModel:
    """Walk-forward training over rolling (train_window, val_window) splits.

    Parameters
    ----------
    surfaces : (N, 1, n_k, n_t)  – all normalised IV surface snapshots
    """
    device = device or torch.device("cpu")
    N, _, n_k, n_t = surfaces.shape

    model = NeuralVolModel(n_k=n_k, n_t=n_t).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    seq_len = 5  # number of conditioning snapshots

    for fold_start in range(0, N - train_window - val_window, val_window):
        train_end = fold_start + train_window
        val_end = train_end + val_window

        def make_dataset(start: int, end: int) -> TensorDataset:
            xs, ys = [], []
            for i in range(start + seq_len, end):
                snap = surfaces[i - seq_len: i].unsqueeze(0)  # (1, seq, 1, n_k, n_t)
                target = surfaces[i, 0]                        # (n_k, n_t)
                xs.append(snap)
                ys.append(target)
            if not xs:
                return TensorDataset(torch.empty(0), torch.empty(0))
            return TensorDataset(torch.cat(xs, dim=0), torch.stack(ys, dim=0))

        train_ds = make_dataset(fold_start, train_end)
        val_ds = make_dataset(train_end, val_end)

        if len(train_ds) == 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        ts = torch.linspace(0, 1, seq_len, device=device)

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, ts, device=device)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Fold %d | Epoch %d/%d | Train loss: %.6f",
                    fold_start, epoch + 1, n_epochs, train_loss,
                )

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(checkpoint_dir) / f"fold_{fold_start}.pt")

    return model
