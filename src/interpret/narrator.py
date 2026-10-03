"""Plain-language interpretation of a ticker's vol surface, written by Claude.

On demand only: the dashboard calls :func:`interpret` when asked, and each
result is cached on disk per (ticker, date), keyed by a hash of the exact
snapshot, prompt version and model.  Re-opening the same day costs nothing;
the snapshot changing (data rebuilt) marks the cached text as stale.

The model is told to use only the numbers in the snapshot, to separate what
the surface shows from what it might mean, and to carry the project's known
caveats (noise floors, short histories, the forecast's track record) rather
than overstate them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.interpret.snapshot import snapshot_hash

MODEL = "claude-opus-5-5"
EFFORT = "medium"
PROMPT_VERSION = "interpret/1"
DEFAULT_CACHE_DIR = "data/interpretations"

SYSTEM_PROMPT = """\
You write the interpretation panel of an options-volatility dashboard. Each \
request gives you one JSON snapshot for one ticker on one date. Write a short \
read of what the implied-volatility surface, realised volatility and the \
volatility forecast show, for a reader who knows options.

Ground rules:
- Use only numbers that appear in the snapshot. Never invent levels, dates, \
events or news; you do not know why markets moved, so do not attribute moves to \
causes the snapshot does not contain. If something you would want is missing \
(null), say so briefly instead of guessing.
- Distinguish observation from interpretation: state what the numbers show, \
then what that usually indicates, with hedged language for the latter.
- This is analysis, not advice. Do not recommend trades or positions.

Conventions in the snapshot:
- Vols are annualised percent; differences are vol points. "atm" is \
forward-at-the-money implied vol at constant maturity; "variance_swap_30d" is a \
VIX-style model-free 30-day vol (for SPY it tracks VIX closely).
- Risk reversal = 25-delta call vol minus 25-delta put vol, so negative means \
puts are richer (normal for equity indices). Butterfly = average of the 25-delta \
wings minus ATM (smile curvature).
- Term structure values are longer minus shorter maturity: positive is the usual \
upward-sloping (contango) shape; negative (inversion) usually accompanies stress.
- The variance risk premium is 30-day ATM implied minus 21-day realised vol.
- Percentiles are 0-100 within the ticker's trailing year, or within all \
available history when shorter; history_available_days says how much. Treat \
percentiles from under ~120 days as rough.

Known limits to respect:
- Measurement noise is about 0.3 vol points per day in 30-day risk reversals \
and about 0.5 in 30-day ATM vol; do not read meaning into single-day changes \
smaller than that. Butterflies are precise (under 0.1 pts of noise) but small.
- The volatility forecast is HAR plus implied vol, pooled across tickers and \
mostly estimated on SPY. Its track_record shows how it has done here: compare \
rmse_forecast_pts with rmse_implied_pts. For index ETFs it has usually beaten \
raw implied vol; for single stocks raw implied vol has often been as good or \
better, partly because implied vol prices known events such as earnings that \
the model cannot see. Say which applies, using the numbers.
- A large data_quality median_fit_error_pts (above ~0.6) or few expiries means \
the surface is less reliable that day; mention it if so.

Format: Markdown, about 200-300 words. Start with a one-sentence headline in \
bold. Then three short sections with these level-4 headings: "Surface", \
"Realised vs implied", "Forecast". Use peers only where they add contrast. No \
tables, no preamble, no closing summary."""


class MissingCredentials(RuntimeError):
    """No Anthropic API credentials are configured."""


@dataclass
class Interpretation:
    ticker: str
    as_of: str
    text: str
    model: str
    prompt_version: str
    snapshot_hash: str
    created_at: str
    usage: dict = field(default_factory=dict)
    served_by: Optional[str] = None


def _cache_path(ticker: str, as_of: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"ticker={ticker}" / f"{as_of}.json"


def load_cached(snapshot: dict, cache_dir: str = DEFAULT_CACHE_DIR) -> tuple[Optional[Interpretation], bool]:
    """The cached interpretation for the snapshot's (ticker, date), if any.

    Returns ``(interpretation, is_current)``: *is_current* is False when the
    cached text was written from different data, prompt or model.
    """
    p = _cache_path(snapshot["ticker"], snapshot["as_of"], cache_dir)
    if not p.exists():
        return None, False
    try:
        it = Interpretation(**json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None, False
    return it, it.snapshot_hash == snapshot_hash(snapshot, PROMPT_VERSION, MODEL)


def _client():
    """An Anthropic client, after loading the project's .env."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise MissingCredentials(
            "No Anthropic API key found. Add ANTHROPIC_API_KEY=... to the project's .env "
            "file (create a key at console.anthropic.com), then reload the dashboard.")
    import anthropic

    # Keys not scoped to a workspace must name one on every request.  The SDK
    # keeps a client-level anthropic-workspace-id header on each call.
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    return anthropic.Anthropic(default_headers=headers)


def _explain(exc: Exception) -> Exception:
    """Turn account-setup API errors into instructions; pass others through."""
    msg = str(exc)
    if "anthropic-workspace-id" in msg or "not scoped to a workspace" in msg:
        return MissingCredentials(
            "This API key isn't tied to a workspace, so requests must name one. Either add "
            "ANTHROPIC_WORKSPACE_ID=wrkspc_... to .env (Console → Settings → Workspaces), or "
            "create a key inside a workspace and use that instead. Then reload the dashboard.")
    if "credit balance is too low" in msg:
        return MissingCredentials(
            "The Anthropic account has no API credit. Add credit under Plans & Billing at "
            "console.anthropic.com, then try again.")
    return exc


def interpret(
    snapshot: dict,
    *,
    client: Any = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Interpretation:
    """Interpretation of *snapshot*, from the cache or a new Claude call.

    Raises :class:`MissingCredentials` without a key, and ``RuntimeError``
    when the request is declined (after the server-side fallback).
    """
    cached, current = load_cached(snapshot, cache_dir)
    if cached is not None and current and not force:
        return cached

    client = client or _client()
    user = ("Snapshot (JSON):\n\n"
            + json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False))
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": EFFORT},
            # Re-run a classifier-declined request on Anthropic's recommended
            # fallback model, server-side, instead of returning the refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            cache_control={"type": "ephemeral"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:
        raise _explain(exc) from exc
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise RuntimeError(f"Claude declined to interpret this snapshot (category: {category}).")
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise RuntimeError(f"Empty response (stop_reason: {response.stop_reason}).")

    usage = getattr(response, "usage", None)
    it = Interpretation(
        ticker=snapshot["ticker"], as_of=snapshot["as_of"], text=text, model=MODEL,
        prompt_version=PROMPT_VERSION,
        snapshot_hash=snapshot_hash(snapshot, PROMPT_VERSION, MODEL),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        usage={k: getattr(usage, k, None) for k in
               ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")} if usage is not None else {},
        served_by=getattr(response, "model", MODEL),
    )
    p = _cache_path(it.ticker, it.as_of, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(it), indent=1), encoding="utf-8")
    return it
