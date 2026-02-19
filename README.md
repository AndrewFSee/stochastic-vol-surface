# Deep Stochastic Volatility Surface Modeler

A production-grade Python framework that scrapes live options chains, builds arbitrage-free implied-volatility surfaces, calibrates parametric models (SABR, Heston, SVI, rough-Bergomi), and trains a Neural SDE to learn residual dynamics.  Tradeable signals and a walk-forward backtest engine are included, together with a Streamlit dashboard.

---

## Architecture

```
stochastic-vol-surface/
├── config/                  # YAML configuration
│   ├── default.yaml         # Scraper, model, signal, backtest params
│   └── surface_grid.yaml    # Log-moneyness × tenor grid
├── src/
│   ├── data/                # Ingestion: yfinance, FRED, Kaggle
│   ├── surface/             # IV inversion, grid builder, interpolation, filters
│   ├── models/              # SABR, Heston, SVI, rough-Bergomi, calibration
│   ├── neural/              # Neural SDE, encoder, decoder, training
│   ├── signals/             # Skew, term-structure, butterfly, regime signals
│   ├── backtest/            # Walk-forward engine, strategies, Greeks, metrics
│   ├── dashboard/           # Streamlit app
│   └── agents/              # LLM-based analyst / recommender / narrator stubs
├── scripts/                 # CLI entry-points
└── tests/                   # Pytest test suite
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-org/stochastic-vol-surface.git
cd stochastic-vol-surface
pip install -e ".[dev]"

# 2. Configure secrets
cp .env.example .env
# edit .env with your FRED_API_KEY and OPENAI_API_KEY

# 3. Scrape today's options chains
python scripts/scrape.py --tickers SPY QQQ AAPL MSFT

# 4. Backfill historical data from Kaggle
python scripts/backfill.py path/to/spy_options_iv.csv

# 5. Calibrate parametric models
python scripts/calibrate.py --start 2023-01-01 --end 2023-12-31

# 6. Train the Neural SDE
python scripts/train.py --epochs 100 --batch-size 32

# 7. Run walk-forward backtest
python scripts/backtest.py --start 2022-01-01 --end 2023-12-31

# 8. Generate HTML tearsheet
python scripts/tearsheet.py --output reports/tearsheet.html

# 9. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Data Pipeline

| Source | Module | Description |
|---|---|---|
| yfinance | `src/data/scraper.py` | Daily SPY/QQQ/AAPL/MSFT options chains |
| yfinance | `src/data/vix_family.py` | VIX, VIX3M, VIX9D, SKEW, VVIX |
| FRED API | `src/data/rates.py` | Risk-free rates (DGS1MO…DGS10) |
| Kaggle | `src/data/kaggle_loader.py` | Historical SPY IV dataset |
| Parquet | `src/data/storage.py` | Partitioned storage with dedup |

---

## Surface Construction

1. **IV Inversion** (`src/surface/implied_vol.py`) – Brent's method on Black-Scholes
2. **Grid Builder** (`src/surface/grid_builder.py`) – raw chain → log-moneyness × tenor grid
3. **Interpolation** (`src/surface/interpolation.py`) – SVI per slice, cubic spline across tenors, RBF for 2D
4. **Filters** (`src/surface/filters.py`) – butterfly (convexity) + calendar (monotonicity) arbitrage removal
5. **VolSurface** (`src/surface/surface.py`) – `RegularGridInterpolator`-backed query object

---

## Parametric Models

| Model | Module | Method |
|---|---|---|
| SABR | `src/models/sabr.py` | Hagan (2002) approximation, L-BFGS-B calibration |
| Heston | `src/models/heston.py` | Characteristic function + FFT pricing |
| SVI | `src/models/svi.py` | Gatheral raw SVI, quasi-explicit fit |
| rough-Bergomi | `src/models/rough_bergomi.py` | Monte Carlo pricing |

Model selection via AIC/BIC is in `src/models/model_selection.py`.

---

## Neural SDE

The `NeuralSDE` (`src/neural/neural_sde.py`) models the latent vol-surface dynamics:

```
dY_t = f_θ(t, Y_t) dt + g_θ(t, Y_t) dW_t
```

where `f` and `g` are MLPs.  A `SurfaceEncoder` (Conv2D) compresses daily snapshots to a latent vector, and a `SurfaceDecoder` (MLP) reconstructs predicted surfaces.  The model learns residuals between market IV and best-fit parametric IV.

---

## Signals

| Signal | Module |
|---|---|
| 25-delta skew | `src/signals/skew_signals.py` |
| Term-structure slope/curvature | `src/signals/term_structure.py` |
| Butterfly mispricing | `src/signals/butterfly.py` |
| Calendar spread arbitrage | `src/signals/calendar_spread.py` |
| Vol regime (Low/Normal/High/Crisis) | `src/signals/regime_vol.py` |
| Composite recommendation | `src/signals/composite.py` |

---

## Backtest

Walk-forward engine in `src/backtest/engine.py` with daily rebalancing.  Strategies include delta-hedged straddles, risk-reversals, and butterflies.  Metrics: Sharpe, Sortino, max drawdown, win rate, P&L attribution.

---

## Configuration

Edit `config/default.yaml` to change tickers, model hyperparameters, signal thresholds, and backtest parameters. Edit `config/surface_grid.yaml` to change the grid resolution.

---

## Environment Variables

| Variable | Description |
|---|---|
| `FRED_API_KEY` | FRED API key (free registration) |
| `OPENAI_API_KEY` | OpenAI API key for agent stubs |
| `KAGGLE_USERNAME` | Kaggle username for backfill loader |
| `KAGGLE_KEY` | Kaggle API key |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT – see `LICENSE`.
