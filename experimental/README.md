# Experimental

Research code moved out of the main package because it does not yet serve the
project's goal (vol surfaces, features, dashboard) and has not been checked
against market data. It still runs and its tests pass. Treat its outputs as
untested.

## Status

| Area | Path | Status |
|---|---|---|
| Parametric models | `models/` | SABR, Heston, SVI, rough Bergomi. Unit-tested on synthetic smiles; not fitted to the live corpus. `models/svi.py` duplicates the SVI in `src/surface/`. |
| Neural SDE | `neural/` | Trains on the 25×8 surface grids. About 150 days of live data is too little history to learn dynamics. |
| Signals | `signals/` | Rule-based skew, term-structure and regime calls. For modelling, the feature table's z-scores and percentiles supersede them. |
| Backtest | `backtest/` | Walk-forward engine pricing at the surface mid with a flat cost. Prices legs off `S·e^(rT)`, not the parity forwards the surfaces now use. |
| Agents | `agents/` | LLM narrative stubs (optional `openai`). |

Promoting any of these means validating it against data first, the way
`scripts/validate_surfaces.py` checks the surfaces against VIX.

## Running

From the repository root:

```bash
pip install -e ".[experimental]"        # torch, hmmlearn, openai
pytest experimental/tests

python -m experimental.scripts.calibrate --start 2026-02-19 --end 2026-07-31
python -m experimental.scripts.train --ticker SPY --epochs 100
python -m experimental.scripts.backtest --ticker SPY --strategy straddle
python -m experimental.scripts.tearsheet --output reports/tearsheet.html
python -m experimental.scripts.regime_analysis --surfaces-dir data/surfaces
```

Parameters are in `config.yaml` (moved here from `config/default.yaml`).
Experimental code may import from `src`; nothing in `src` imports from here.

---

## Parametric Models

| Model | Module | Method |
|---|---|---|
| SABR | `experimental/models/sabr.py` | Hagan (2002) approximation, L-BFGS-B calibration |
| Heston | `experimental/models/heston.py` | Characteristic function + FFT pricing |
| SVI | `experimental/models/svi.py` | Gatheral raw SVI, quasi-explicit fit |
| rough-Bergomi | `experimental/models/rough_bergomi.py` | Monte Carlo pricing |

Model selection via AIC/BIC is in `experimental/models/model_selection.py`.

---

## Neural SDE

The `NeuralSDE` (`experimental/neural/neural_sde.py`) models the latent vol-surface dynamics:

```
dY_t = f_θ(t, Y_t) dt + g_θ(t, Y_t) dW_t
```

where `f` and `g` are MLPs.  A `SurfaceEncoder` (Conv2D) compresses daily snapshots to a latent vector, and a `SurfaceDecoder` (MLP) reconstructs predicted surfaces.  The model learns residuals between market IV and best-fit parametric IV.

---

## Signals

| Signal | Module |
|---|---|
| 25-delta skew | `experimental/signals/skew_signals.py` |
| Term-structure slope/curvature | `experimental/signals/term_structure.py` |
| Butterfly mispricing | `experimental/signals/butterfly.py` |
| Calendar spread arbitrage | `experimental/signals/calendar_spread.py` |
| Vol regime (Low/Normal/High/Crisis) | `experimental/signals/regime_vol.py` |
| Composite recommendation | `experimental/signals/composite.py` |

---

## Backtest

Walk-forward engine in `experimental/backtest/engine.py`.

Every position is a bundle of `OptionLeg`s carrying a full contract spec —
type, strike, expiry, signed quantity. That is what makes honest P&L possible:
each open leg is repriced daily against the new surface at its *own* moneyness
and *remaining* tenor, instead of being approximated from one ATM vol change.

Each day the engine reprices open legs (settling expired ones at intrinsic),
marks the delta hedge against the spot move, rebalances the hedge, rolls
positions that hit their holding period, and opens a new one when flat.

```bash
python -m experimental.scripts.backtest --ticker SPY --start 2026-02-19 --end 2026-07-31 \
    --strategy straddle --tenor 0.25 --holding-days 5
```

Strategies (`experimental/backtest/strategies.py`): `straddle`, `short_straddle`,
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
