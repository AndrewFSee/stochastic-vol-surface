"""One-shot script to test the full surface construction pipeline with live scraped data."""

import logging
import sys
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s %(name)s: %(message)s",
)

from src.data.storage import load_options_chain
from src.surface.surface import VolSurface

# 1. Load the SPY chain
chain = load_options_chain("SPY", start="2026-02-19", end="2026-02-19")
print(f"Loaded {len(chain)} rows")
print(f"Columns: {list(chain.columns)}")
spot = float(chain["underlying_price"].iloc[0])
print(f"Spot: {spot}")

# 2. Build the full surface via from_chain
vs = VolSurface.from_chain(
    chain,
    ticker="SPY",
    as_of=date(2026, 2, 19),
    spot=spot,
    r=0.04,
    method="svi",
)
print(f"Grid shape: {vs.iv_grid.shape}")
print(f"ATM vol (3m): {vs.atm_vol(0.25):.4f}")
print(f"ATM vol (1y): {vs.atm_vol(1.0):.4f}")
print(f"Skew (3m):    {vs.skew(0.25):.4f}")
ts = vs.term_structure()
print(f"Term structure: {[f'{v:.4f}' for v in ts]}")

# 3. Save and reload
path = vs.save("data/surfaces/SPY_2026-02-19.parquet")
print(f"Saved to {path}")

vs2 = VolSurface.load(path)
print(f"Loaded back: {vs2.ticker} {vs2.as_of}  grid={vs2.iv_grid.shape}")
print(f"  ATM 3m check: {vs2.atm_vol(0.25):.4f}")
print("DONE")
