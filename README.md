# Deep Stochastic Volatility Surface Modeler

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A research-grade system that scrapes options chain data daily, constructs implied volatility surfaces, calibrates parametric stochastic volatility models (SABR, Heston, rBergomi, SVI), trains a neural SDE to learn residuals from parametric fits, generates vol-arb trading signals, and backtests delta-hedged volatility strategies — all using only **free, publicly available data**.

### Key Capabilities

- **Daily options chain scraping** for SPY, QQQ, AAPL, MSFT via yfinance
- **Implied volatility surface construction** with arbitrage-free interpolation (SVI per-slice, cubic spline, RBF)
- **Parametric model calibration**: SABR, Heston, rBergomi, SVI
- **Neural SDE residual learning**: learns market IV − parametric IV dynamics
- **Vol-arb signal generation**: skew, term structure, butterfly, calendar spread anomalies
- **Walk-forward backtesting** of delta-hedged straddles, risk reversals, calendar spreads
- **Live dashboard** (Phase 2) with 3D surface visualization and real-time signal monitoring
- **LLM agentic layer** (Phase 3) for natural-language surface interpretation and trade recommendations

---

## Architecture

```
Data Layer (Phase 1 & 2)
    ├── yfinance scraper  ──────────────────────────────────────┐
    ├── Kaggle backfill (SPY IV 2019-2024)                      │
    ├── VIX family (VIX, VIX3M, VIX9D, SKEW, VVIX)             │
    └── FRED risk-free rates (DGS1MO, DGS3MO, ...)             │
                                                                ▼
Surface Construction (Phase 1)                      Parquet storage
    ├── Black-Scholes IV inversion (Brent's method)            │
    ├── Log-moneyness × tenor grid construction                 │
    ├── Arbitrage filters (butterfly, calendar)                 │
    └── VolSurface class (query any (m, τ) point)              │
                                                                ▼
Parametric Calibration (Phase 1)                   Calibrated surface
    ├── SABR (Hagan 2002)                                       │
    ├── Heston (characteristic function + FFT)                  │
    ├── rBergomi (rough volatility, Monte Carlo)                │
    └── SVI (Gatheral 2006, quasi-explicit fit)                │
                                                                ▼
Neural SDE (Phase 1)                           Residuals: market − parametric
    ├── Surface encoder (Conv2D → latent state)                │
    ├── Neural SDE (drift + diffusion as MLPs)                  │
    └── Surface decoder (latent → predicted grid)              │
                                                                ▼
Signal Generation (Phase 1)                      Vol predictions
    ├── 25Δ skew steepening / flattening                       │
    ├── Term structure inversion detection                      │
    ├── Butterfly mispricing (model vs market)                  │
    └── Vol regime classifier (Low/Normal/High/Crisis)         │
                                                                ▼
Backtesting (Phase 1)                             Trading signals
    ├── Walk-forward engine                                     │
    ├── Delta-hedged straddles, risk reversals                  │
    ├── Greeks P&L attribution                                  │
    └── Sharpe / Sortino / max drawdown tearsheet              │
                                                                ▼
Dashboard (Phase 2)                              Backtest results
    ├── 3D vol surface (Plotly)
    ├── VIX term structure with regime shading
    ├── 25Δ skew time series + Z-score bands
    └── Live signal monitoring panel

Agentic AI (Phase 3)
    ├── LLM surface anomaly analyst
    ├── LLM trade recommender
    └── LLM regime narrator
```

---

## Project Phases

### Phase 1 — Historical Foundation (Immediate)
Build the complete data pipeline using free data sources, calibrate parametric models on historical surfaces, train the neural SDE, generate trading signals, and validate via backtesting.

### Phase 2 — Live System (Ongoing)
Deploy the daily scraper on a schedule (4:30 PM ET), accumulate real options data, and monitor the system through a live Streamlit dashboard.

### Phase 3 — Agentic AI (Later)
Add an LLM-powered interpretation layer that narrates surface anomalies in plain English, suggests specific vol-arb trades with quantitative rationale, and contextualizes the current vol regime against historical episodes.

---

## Data Sources (Free)

| Source | Data | Module |
|--------|------|--------|
| yfinance | Daily options chains (SPY, QQQ, AAPL, MSFT) | `src/data/scraper.py` |
| Kaggle | SPY IV surface dataset 2019–2024 | `src/data/kaggle_loader.py` |
| yfinance | VIX, VIX3M, VIX9D, SKEW, VVIX | `src/data/vix_family.py` |
| FRED API | Risk-free rates (DGS1MO, DGS3MO, DGS6MO, DGS1) | `src/data/rates.py` |

No Bloomberg, Refinitiv, or paid data vendors required.

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/AndrewFSee/stochastic-vol-surface.git
cd stochastic-vol-surface
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 3. Backfill historical data

```bash
# Load Kaggle SPY IV dataset (download from Kaggle first)
python scripts/backfill_kaggle.py --path data/kaggle/spy_iv.csv

# Download VIX family
python scripts/run_scraper.py --tickers VIX --mode vix

# Calibrate parametric models on stored surfaces
python scripts/calibrate_surface.py --ticker SPY --start 2020-01-01
```

### 4. Train neural SDE

```bash
python scripts/train_neural_sde.py --ticker SPY --epochs 100
```

### 5. Run backtest

```bash
python scripts/run_backtest.py --ticker SPY --strategy straddle --start 2020-01-01
python scripts/generate_tearsheet.py --output reports/backtest.html
```

### 6. Launch dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Repository Structure

```
stochastic_vol_surface/
├── config/              # YAML configuration files
├── src/
│   ├── data/            # Data acquisition and storage
│   ├── surface/         # IV surface construction
│   ├── models/          # Parametric stochastic vol models
│   ├── neural/          # Neural SDE residual learning
│   ├── signals/         # Vol-arb signal generation
│   ├── backtest/        # Backtesting engine
│   ├── agents/          # Phase 3: LLM agentic layer
│   └── dashboard/       # Phase 2: Streamlit dashboard
├── scripts/             # CLI entrypoints
├── notebooks/           # Exploratory analysis
└── tests/               # Unit and integration tests
```

---

## Academic References

- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk. *Wilmott Magazine*, 1, 84–108.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies*, 6(2), 327–343.
- Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887–904.
- Kidger, P., Foster, J., Li, X., & Lyons, T. (2021). Neural SDEs as infinite-dimensional GANs. *ICML 2021*. arXiv:2102.03657.

---

## License

MIT License. See [LICENSE](LICENSE).