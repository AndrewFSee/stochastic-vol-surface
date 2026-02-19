#!/usr/bin/env python
"""CLI: generate HTML tearsheet from backtest results CSV."""

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--input", "-i", "input_file", default="backtest_results.csv",
              show_default=True, help="Backtest results CSV")
@click.option("--output", "-o", default="reports/tearsheet.html",
              show_default=True, help="Output HTML path")
@click.option("--title", default="Strategy Tearsheet", show_default=True)
def main(input_file, output, title):
    """Generate Plotly HTML tearsheet from backtest results."""
    import pandas as pd
    from src.backtest.tearsheet import generate_tearsheet

    df = pd.read_csv(input_file, parse_dates=["date"])
    if "cumulative_pnl" not in df.columns:
        df["cumulative_pnl"] = df["net_pnl"].cumsum()

    out = generate_tearsheet(df, output_path=output, title=title)
    click.echo(f"Tearsheet written to {out}")


if __name__ == "__main__":
    main()
