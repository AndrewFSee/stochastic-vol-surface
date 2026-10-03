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
    import numpy as np
    import torch
    from pathlib import Path
    from experimental.neural.training import walk_forward_train

    surfaces_dir = Path(data_dir) / f"ticker={ticker}"
    if not surfaces_dir.exists():
        click.echo(
            f"No surfaces found in {surfaces_dir}. "
            f"Run: python scripts/build_surfaces.py -t {ticker}",
            err=True,
        )
        raise SystemExit(1)

    from src.surface.batch import load_surface_history

    dates, k_grid, t_grid, grids = load_surface_history(ticker, surfaces_dir=data_dir)
    if grids.size == 0:
        click.echo(f"Could not build surface grids from {surfaces_dir}", err=True)
        raise SystemExit(1)

    click.echo(
        f"Loaded {len(dates)} surface snapshots for {ticker} "
        f"({dates[0]} -> {dates[-1]}), grid {grids.shape[1]}x{grids.shape[2]}"
    )

    # Guard the training set: a non-finite cell silently produces NaN loss.
    if not np.isfinite(grids).all():
        n_bad = int((~np.isfinite(grids)).any(axis=(1, 2)).sum())
        click.echo(f"Dropping {n_bad} snapshot(s) with non-finite values", err=True)
        keep = np.isfinite(grids).all(axis=(1, 2))
        grids = grids[keep]
        if grids.size == 0:
            click.echo("No usable surfaces remain.", err=True)
            raise SystemExit(1)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    surfaces = torch.tensor(grids, dtype=torch.float32).unsqueeze(1)
    model = walk_forward_train(
        surfaces,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        checkpoint_dir=ckpt_dir,
    )

    final_path = ckpt_dir / f"{ticker}_final.pt"
    torch.save(model.state_dict(), final_path)
    click.echo(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
