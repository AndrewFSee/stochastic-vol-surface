"""Surface anomaly analyst agent stub."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    ticker: str
    date: str
    anomalies: list[str]
    severity: str   # "low" | "medium" | "high"
    narrative: str


def analyse_surface_anomalies(
    ticker: str,
    date: str,
    surface_metrics: dict,
    openai_api_key: Optional[str] = None,
) -> AnomalyReport:
    """Analyse a vol surface for anomalies.

    Uses OpenAI GPT if an API key is provided; otherwise returns a
    rule-based analysis.

    Parameters
    ----------
    surface_metrics : dict with keys like 'atm_vol', 'skew', 'term_slope', ...
    """
    anomalies: list[str] = []

    # Rule-based detection
    atm = surface_metrics.get("atm_vol", 0.2)
    skew = surface_metrics.get("skew", 0.0)
    term_slope = surface_metrics.get("term_slope", 0.0)

    if atm > 0.40:
        anomalies.append(f"Elevated ATM vol: {atm:.2%}")
    if abs(skew) > 0.05:
        direction = "steep" if skew > 0 else "inverted"
        anomalies.append(f"Unusual skew ({direction}): {skew:.4f}")
    if term_slope < -0.02:
        anomalies.append(f"Inverted term structure (slope={term_slope:.4f})")

    severity = "low"
    if len(anomalies) >= 2:
        severity = "medium"
    if len(anomalies) >= 3 or atm > 0.50:
        severity = "high"

    narrative = f"Surface for {ticker} on {date}: {len(anomalies)} anomaly(ies) detected."
    if anomalies:
        narrative += " " + " | ".join(anomalies)

    if openai_api_key:
        try:
            narrative = _llm_narrative(ticker, date, surface_metrics, anomalies, openai_api_key)
        except Exception as exc:
            logger.warning("LLM narrative failed: %s", exc)

    return AnomalyReport(
        ticker=ticker,
        date=date,
        anomalies=anomalies,
        severity=severity,
        narrative=narrative,
    )


def _llm_narrative(
    ticker: str,
    date: str,
    metrics: dict,
    anomalies: list[str],
    api_key: str,
) -> str:
    """Generate narrative using OpenAI (stub – requires openai package)."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"You are a quantitative analyst. Summarise the following vol surface "
            f"anomalies for {ticker} on {date} in 2-3 sentences:\n"
            f"Metrics: {metrics}\nAnomalies: {anomalies}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return "openai package not installed."
