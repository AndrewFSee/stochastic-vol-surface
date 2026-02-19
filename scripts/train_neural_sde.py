"""Train the neural SDE on historical vol surface snapshots."""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural SDE on historical vol surfaces")
    parser.add_argument("--ticker", default="SPY", help="Ticker (default: SPY)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, dest="latent_dim")
    parser.add_argument("--save-path", default="data/models/neural_vol.pt", dest="save_path")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    import numpy as np

    from src.data.storage import OptionsStorage

    storage = OptionsStorage()
    dates = storage.list_dates(args.ticker)

    if not dates:
        logger.error("No surface data found for %s. Run the scraper first.", args.ticker)
        return

    # Load surfaces
    surfaces_list = []
    for dt in dates:
        surf = storage.load_surface(args.ticker, dt)
        if not surf.empty:
            surfaces_list.append(surf.values.astype("float32"))

    if len(surfaces_list) < 10:
        logger.error("Need at least 10 surface snapshots to train. Got %d.", len(surfaces_list))
        return

    surfaces = np.stack(surfaces_list)  # (T, n_m, n_T)

    # Build grids from the first surface
    first_surf = storage.load_surface(args.ticker, dates[0])
    moneyness_grid = first_surf.index.values.astype("float32")
    tenor_grid = first_surf.columns.values.astype("float32")

    from src.neural.training import train_neural_vol_model

    model, history = train_neural_vol_model(
        surfaces=surfaces,
        moneyness_grid=moneyness_grid,
        tenor_grid=tenor_grid,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device,
    )

    best_val = min(history["val_loss"]) if history["val_loss"] else float("nan")
    logger.info("Training complete. Best validation loss: %.6f", best_val)


if __name__ == "__main__":
    main()
