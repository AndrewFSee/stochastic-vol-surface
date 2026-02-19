"""Market narrative generator agent stub."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketNarrative:
    date: str
    headline: str
    body: str
    key_observations: list[str]


def generate_narrative(
    date: str,
    surface_metrics: dict,
    signals: dict,
    openai_api_key: Optional[str] = None,
) -> MarketNarrative:
    """Generate a human-readable market narrative for the current vol surface.

    Uses rule-based text if openai is not available.
    """
    obs: list[str] = []

    atm = surface_metrics.get("atm_vol", 0.18)
    skew = surface_metrics.get("skew", 0.0)
    regime = signals.get("regime", "Normal")
    recommendation = signals.get("recommendation", "NEUTRAL")

    obs.append(f"ATM vol is {atm:.1%} ({regime} regime)")
    if abs(skew) > 0.02:
        direction = "put-heavy" if skew > 0 else "call-heavy"
        obs.append(f"25d skew is {direction} ({skew:+.4f})")

    obs.append(f"Signal: {recommendation}")

    headline = f"Vol Surface Update – {date}"
    body = (
        f"As of {date}, the implied volatility surface shows "
        f"{'elevated' if atm > 0.25 else 'muted'} overall vol "
        f"with a {'steep' if abs(skew) > 0.03 else 'moderate'} skew. "
        f"Regime classification: {regime}. "
        f"Recommended position: {recommendation}."
    )

    if openai_api_key:
        try:
            headline, body = _llm_narrative(date, surface_metrics, signals, obs, openai_api_key)
        except Exception as exc:
            logger.warning("LLM narrative failed: %s", exc)

    return MarketNarrative(
        date=date,
        headline=headline,
        body=body,
        key_observations=obs,
    )


def _llm_narrative(
    date: str,
    metrics: dict,
    signals: dict,
    observations: list[str],
    api_key: str,
) -> tuple[str, str]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"You are a vol market commentator. Write a short market narrative for {date}. "
            f"Metrics: {metrics}. Signals: {signals}. Key observations: {observations}. "
            f"Return a JSON object with keys 'headline' and 'body'."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        return data["headline"], data["body"]
    except Exception:
        return f"Vol Surface Update – {date}", "Narrative unavailable."
