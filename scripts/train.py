#!/usr/bin/env python
"""CLI: train the Neural SDE on historical surface snapshots."""

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--data-dir", default="data/surfaces", show_default=True)
@click.option("--ticker", "-t", default="SPY", show_default=True)
@click.option("--epochs", "-e", default=50, show_default=True)
@click.option("--batch-size", "-b", default=32, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--checkpoint-dir", default="checkpoints", show_default=True)
def main(data_dir, ticker, epochs, batch_size, lr, checkpoint_dir):
    """Train the NeuralVolModel on stored surface snapshots."""
    import torch
    import numpy as np
    from pathlib import Path
    from src.neural.training import walk_forward_train

    # Load surfaces from Parquet store
    surfaces_dir = Path(data_dir) / f"ticker={ticker}"
    parquet_files = sorted(surfaces_dir.glob("date=*/surface.parquet"))

    if not parquet_files:
        click.echo(f"No surfaces found in {surfaces_dir}", err=True)
        raise SystemExit(1)

    import pandas as pd
    frames = [pd.read_parquet(p) for p in parquet_files]
    click.echo(f"Loaded {len(frames)} surface snapshots for {ticker}")

    # Pivot each surface to a grid
    n_k, n_t = 25, 8
    grids = []
    for df in frames:
        if "implied_vol" in df.columns and len(df) > 0:
            pivot = df.pivot_table(
                index="log_moneyness", columns="tenor", values="implied_vol", aggfunc="mean"
            )
            grid = pivot.to_numpy()
            if grid.shape == (n_k, n_t):
                grids.append(grid)

    if not grids:
        click.echo("Could not build surface grids from data.", err=True)
        raise SystemExit(1)

    surfaces = torch.tensor(np.stack(grids), dtype=torch.float32).unsqueeze(1)
    model = walk_forward_train(
        surfaces,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        checkpoint_dir=Path(checkpoint_dir),
    )

    final_path = Path(checkpoint_dir) / f"{ticker}_final.pt"
    torch.save(model.state_dict(), final_path)
    click.echo(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
