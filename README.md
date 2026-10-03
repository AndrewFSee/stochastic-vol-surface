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

# 3. Collect today's data (options + VIX + rates + prices + surfaces + features)
python scripts/schedule_scraper.py --once

# 4. Backfill the risk-free curve and underlying prices (both serve full history)
python scripts/backfill_rates.py --start 2026-01-01
python scripts/backfill_underlying.py --start 2024-01-01

# 5. Build the surface corpus from every stored chain
python scripts/build_surfaces.py --report data/logs/surface_build.csv

# 6. Build the feature table (the main output)
python scripts/build_features.py

# 7. Check corpus health and surface accuracy at any time
python scripts/data_health.py
python scripts/validate_surfaces.py --strict

# 7. Calibrate parametric models
python scripts/calibrate.py --start 2026-02-19 --end 2026-07-31

# 8. Train the Neural SDE
python scripts/train.py --ticker SPY --epochs 100 --batch-size 32

# 9. Run walk-forward backtest
python scripts/backtest.py --start 2026-02-19 --end 2026-07-31

# 10. Generate HTML tearsheet
python scripts/tearsheet.py --output reports/tearsheet.html

# 11. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Data Pipeline

| Source | Module | Description |
|---|---|---|
| yfinance | `src/data/scraper.py` | Daily options chains (8 tickers) |
| yfinance | `src/data/vix_family.py` | VIX, VIX3M, VIX9D, SKEW, VVIX |
| FRED API | `src/data/rates.py` | Risk-free curve (DGS1MO…DGS10) |
| yfinance | `src/data/underlying.py` | Daily OHLC for the tickers (realised vol) |
| Kaggle | `src/data/kaggle_loader.py` | Historical SPY IV dataset |
| Parquet | `src/data/storage.py` | Partitioned storage with dedup |
| — | `src/data/health.py` | Coverage / freshness / quality reporting |

### Storage layout

```
data/
├── options/ticker={T}/date={D}/chain.parquet    # raw chains
├── surfaces/ticker={T}/date={D}/surface.parquet # 25 × 8 IV grids
├── vix/vix_{D}.parquet                          # rolling 5-day snapshots
│   └── vix_history.parquet                      # consolidated series
├── rates/rates_history.parquet                  # merged FRED curve
├── underlying/prices.parquet                    # daily OHLC, (date, ticker)
├── features/surface_features.parquet            # one row per (ticker, date)
└── logs/scrape_runs.jsonl                       # one line per collection run
```

Each daily VIX snapshot stores a rolling 5-day window, so consecutive files
overlap. `load_vix_history()` merges them, which recovers values that had not
settled when first collected (notably `SKEW`). Always read VIX through that
function rather than a single snapshot file.

### Daily automation

`scripts/schedule_scraper.py --once` runs one full cycle — chains, VIX, rates,
underlying prices, surfaces, then the feature table — skipping non-trading
days via the NYSE calendar. On Windows it
is driven by a Task Scheduler entry (`\StochasticVolSurface\DailyScraper`) at
16:30 ET, Mon–Fri. Each run appends to `data/logs/scrape_runs.jsonl`.

---

## Surface Corpus

`src/surface/batch.py` turns the raw chain store into the standardised surface
store consumed by the neural, calibration, and backtest layers.

* **Resumable** — existing surfaces are skipped unless `--overwrite`, so the
  daily incremental run costs one build per ticker.
* **Real rates** — the discount rate is the FRED curve interpolated to each
  snapshot's tenor, not a hard-coded constant.
* **Quality-gated** — every grid is scored before it is written; non-finite or
  implausible grids are rejected rather than poisoning training data.

```python
from src.surface.batch import load_surface_history

dates, k_grid, t_grid, grids = load_surface_history("SPY")
# grids.shape == (n_dates, 25, 8)
```

---

## Surface Construction

Surfaces are built **one expiry at a time** (`src/surface/slices.py`, builder
`expiry-svi/1`):

1. **Clean quotes** – two-sided markets only, bounded relative spread.
2. **Implied forward** – from put-call parity near the money, per expiry.
   This carries dividends and borrow; `S·e^{rT}` is ~0.9% too high for SPY at
   two years and ~2% for XLF.
3. **OTM Black-76 IVs** – inverted from bid/ask mids (vectorised solver in
   `src/surface/implied_vol.py`); puts below the forward, calls above. Yahoo's
   own `impliedVolatility` is not used when prices are available.
4. **SVI per expiry** – quasi-explicit fit, weighted by bid-ask spread in vol
   terms, constrained to positive variance and Lee's wing bound, with one
   round of outlier rejection.
5. **Across tenors** – total variance linear in T between the two bracketing
   expiries (VIX's constant-maturity construction), then a calendar
   monotonicity clean-up (`src/surface/filters.py`).
6. **VolSurface** (`src/surface/surface.py`) – grid plus an `observed` mask
   (False where a cell is extrapolated beyond quoted strikes or expiries) and
   the per-expiry fits (forward, SVI params, fit RMSE) in the file metadata.

Each saved surface records its builder. `build_surfaces.py` rebuilds
surfaces from an older builder instead of skipping them, and
`load_surface_history` loads only the current builder by default, so a
history never silently mixes methods. The old pooled-bin pipeline
(`grid_builder.build_surface_grid` / `interpolate_to_grid`) is still
reachable with `method="rbf"` but should not feed features or models.

### Checking accuracy

```bash
python scripts/validate_surfaces.py --strict
```

Rebuilds SPY's 30-day variance-swap vol from the fitted smiles and compares
it with VIX (same definition, computed by CBOE from SPX). It also reports
daily-change noise for every ticker. A strongly negative lag-1
autocorrelation of daily changes means jumps that revert the next day, which
is measurement noise.

---

## Feature Table

`data/features/surface_features.parquet` is the project's main output: one
point-in-time row per (ticker, trading date), for the dashboard and for
downstream models. `scripts/build_features.py` rebuilds it in full from the
surface, price and VIX stores, and the daily run does the same.

```python
from src.features import load_feature_table

df = load_feature_table(tickers=["SPY", "QQQ"], start="2026-06-01")
```

**Timing.** Row *t* uses only what was known at about 16:30 ET on *t*: that
day's option snapshot, the underlying's close and the VIX closes. To predict
anything over `(t, t+h]`, use row *t*. Rolling statistics are trailing and
include *t*. A test checks that appending later rows never changes earlier
ones.

| Group | Columns | Notes |
|---|---|---|
| ATM term structure | `atm_{7,14,30,60,91,182,365}d` | Forward-ATM implied vol at constant maturity |
| Smile (30d, 91d) | `iv_{p10,p25,c25,c10}_*`, `rr25_*`, `bf25_*`, `rr10_*`, `bf10_*`, `atm_skew_*`, `atm_curv_*` | Exact forward-delta strikes; `rr` = call − put |
| Variance swap | `vs_30d`, `vs_91d` | Model-free, VIX's definition |
| Term spreads | `ts_7_30`, `ts_30_91`, `ts_30_365` | Longer minus shorter ATM vol |
| Carry | `fwd_carry_1y` | ln(F/S)/T: rate − dividend − borrow. Level only; daily changes are mostly noise |
| Realised | `ret_{1,5,21}d`, `rv_cc_{10,21,63}d`, `rv_yz_21d` | Zero-mean close-to-close and Yang-Zhang |
| Premia | `vrp_30d`, `vrp_var_30d` | ATM − RV in vol; VS² − RV² in variance |
| Market | `mkt_vix`, `mkt_vix3m`, `mkt_vix9d`, `mkt_vvix`, `mkt_skew` | Same for every ticker |
| Dynamics | `<col>_d1`, `_d5`, `_z63`, `_pct252` | For the key columns; laid on the NYSE calendar, so a missing day is a gap, not a 2-day change |
| Quality | `n_expiries`, `nearest_expiry_days`, `fit_rmse`, `parity_fraction` | Use to down-weight weak days |

Vols are decimals (0.15 = 15%). **Nothing is extrapolated.** A feature whose
tenor falls outside the listed expiries, or whose strike falls outside the
quotes of the bracketing expiries, is NaN. For example, XLF often lists no
expiry near 7 days. Butterflies are the noisiest features: they are
second-order and small, so quote noise is a larger share of them.

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

Walk-forward engine in `src/backtest/engine.py`.

Every position is a bundle of `OptionLeg`s carrying a full contract spec —
type, strike, expiry, signed quantity. That is what makes honest P&L possible:
each open leg is repriced daily against the new surface at its *own* moneyness
and *remaining* tenor, instead of being approximated from one ATM vol change.

Each day the engine reprices open legs (settling expired ones at intrinsic),
marks the delta hedge against the spot move, rebalances the hedge, rolls
positions that hit their holding period, and opens a new one when flat.

```bash
python scripts/backtest.py --ticker SPY --start 2026-02-19 --end 2026-07-31 \
    --strategy straddle --tenor 0.25 --holding-days 5
```

Strategies (`src/backtest/strategies.py`): `straddle`, `short_straddle`,
`risk_reversal`, `butterfly`. All are sized to a target **gross** vega
(`--target-vega`, cash P&L per vol point) so their results are comparable.
Gross rather than net matters: a risk reversal's net vega is structurally near
zero, and sizing off it divides by ~0.

Output includes a P&L attribution — delta, gamma, vega, theta, hedge, and an
unexplained residual. The residual is a diagnostic: it measures how much of
the day's move the second-order Greek expansion fails to capture, so a large
residual means the surface moved in a way the Greeks did not describe.

> The engine assumes you can trade at the surface's interpolated mid with a
> flat bps cost. It does not model bid-ask by strike, early assignment, or
> liquidity limits — treat results as a signal-quality measure, not a P&L
> forecast.

---

## Historical backfill

The live corpus starts in Feb 2026 and contains no crisis regime. Historical
SPY chains fill that gap without waiting:

```bash
python scripts/backfill_kaggle.py --list          # inspect the dataset
python scripts/backfill_kaggle.py --download      # all years
python scripts/backfill_kaggle.py --download --years 2018 2020 2022

# Build surfaces over the historical store
python scripts/build_surfaces.py \
    --options-dir data/historical/options \
    --surfaces-dir data/historical/surfaces
```

Historical data lands in **`data/historical/`**, deliberately separate from the
live yfinance corpus. The two differ in source, quote timing, and IV
convention, so keeping them apart preserves provenance — and stops a
2010-dated partition from breaking the daily freshness check. Pool them only
deliberately, and re-run `scripts/backfill_rates.py --start 2009-01-01` first
so historical forwards use the rates of their own era rather than today's.

---

## Dashboard

```bash
streamlit run src/dashboard/app.py
```

Reads the built surface corpus and the consolidated VIX history: pick a ticker
and as-of date to inspect the 3-D surface, a skew slice, the ATM term
structure, ATM vol through time, and the current vol regime. If no surfaces
have been built it falls back to a synthetic surface, so the app always runs.

---

## Configuration

Edit `config/default.yaml` to change tickers, model hyperparameters, signal thresholds, and backtest parameters. Edit `config/surface_grid.yaml` to change the grid resolution.

The grid in `config/surface_grid.yaml` (25 log-moneyness knots × 8 tenors) is
the contract between the surface, neural, and backtest layers — changing it
invalidates every surface already built, so re-run
`scripts/build_surfaces.py --overwrite` afterwards.

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

The suite is offline — every test builds its own synthetic chains in a
temporary store, so no network access or collected data is required.

> **Note:** `pyproject.toml` pins pytest's scratch space to `.pytest_tmp/`
> inside the project. The default Windows location
> (`%LOCALAPPDATA%\Temp\pytest-of-<user>`) is not readable on this machine,
> which breaks every `tmp_path`-based test. pytest wipes that directory on each
> run, so do not put anything else there.

### Checking the live corpus

`pytest` validates the code; `scripts/data_health.py` validates the *data*:

```bash
python scripts/data_health.py                     # full report
python scripts/data_health.py --strict            # exit 1 if stale (for cron/CI)
python scripts/data_health.py --csv-dir data/logs/health
```

It reports collection freshness, per-ticker gaps against the NYSE calendar,
snapshot quality, surface-build coverage, and the VIX / rates stores.

---

## License

MIT – see `LICENSE`.
