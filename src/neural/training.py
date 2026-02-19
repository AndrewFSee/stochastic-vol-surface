"""
Walk-forward training loop for the neural vol model.

Trains the model to predict next-day IV surface residuals from the
current surface using a sliding-window walk-forward validation scheme.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def train_neural_vol_model(
    surfaces: np.ndarray,
    moneyness_grid: np.ndarray,
    tenor_grid: np.ndarray,
    model: Optional["torch.nn.Module"] = None,
    latent_dim: int = 32,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.15,
    patience: int = 20,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple["torch.nn.Module", dict]:
    """Train the Neural Vol Model on a sequence of surface snapshots.

    Parameters
    ----------
    surfaces:
        Array of shape (T, n_moneyness, n_tenors) — sequence of daily
        IV surface snapshots (NaN where no data).
    moneyness_grid:
        Log-moneyness values, shape (n_moneyness,).
    tenor_grid:
        Tenor values in years, shape (n_tenors,).
    model:
        Pre-initialised NeuralVolModel.  If None, one is created.
    epochs:
        Maximum training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam optimizer learning rate.
    validation_split:
        Fraction of data for validation (most recent samples).
    patience:
        Early stopping patience in epochs.
    save_path:
        If provided, save the best model weights here.
    device:
        ``"cpu"`` or ``"cuda"`` (auto-detected if None).

    Returns
    -------
    (model, history)
        Trained model and a dict with ``train_loss`` and ``val_loss`` lists.
    """
    from src.neural.neural_vol_model import NeuralVolModel
    from src.neural.losses import VolSurfaceLoss

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)
    logger.info("Training on device: %s", dev)

    n_snapshots, n_moneyness, n_tenors = surfaces.shape

    if model is None:
        model = NeuralVolModel(
            n_moneyness=n_moneyness,
            n_tenors=n_tenors,
            latent_dim=latent_dim,
        )
    model = model.to(dev)

    # Replace NaN with 0 for training
    surfaces_clean = np.where(np.isnan(surfaces), 0.0, surfaces).astype(np.float32)

    # Create (input_t, target_t+1) pairs
    X = torch.from_numpy(surfaces_clean[:-1])  # (T-1, n_m, n_T)
    Y = torch.from_numpy(surfaces_clean[1:])   # (T-1, n_m, n_T)

    # Walk-forward split: keep last validation_split % for validation
    n_val = max(1, int(len(X) * validation_split))
    n_train = len(X) - n_val

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_train.to(dev), Y_train.to(dev)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val.to(dev), Y_val.to(dev)),
        batch_size=batch_size,
    )

    m_grid = torch.from_numpy(moneyness_grid.astype(np.float32)).to(dev)
    t_grid = torch.from_numpy(tenor_grid.astype(np.float32)).to(dev)
    criterion = VolSurfaceLoss(m_grid, t_grid).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            logger.info("Epoch %d/%d — train=%.6f val=%.6f", epoch, epochs, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=dev))

    return model, history
