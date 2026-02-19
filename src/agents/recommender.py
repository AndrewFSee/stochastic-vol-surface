"""Trade recommendation agent stub."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.signals.composite import CompositeSignal, TradeRecommendation

logger = logging.getLogger(__name__)


@dataclass
class TradeIdea:
    strategy: str
    direction: str   # "long" | "short" | "neutral"
    instrument: str
    rationale: str
    confidence: float
    sizing_pct: float  # percentage of risk budget


def recommend_trades(
    signal: CompositeSignal,
    risk_budget: float = 100_000.0,
    openai_api_key: Optional[str] = None,
) -> list[TradeIdea]:
    """Convert a CompositeSignal into actionable trade ideas.

    Uses rule-based logic; optionally enriched with GPT narrative.
    """
    ideas: list[TradeIdea] = []
    rec = signal.recommendation

    if rec == TradeRecommendation.BUY_VOL:
        ideas.append(TradeIdea(
            strategy="delta_hedged_straddle",
            direction="long",
            instrument="ATM straddle",
            rationale="Vol is below historical average; buy vol via straddle.",
            confidence=signal.confidence,
            sizing_pct=min(signal.confidence * 100, 20.0),
        ))
    elif rec == TradeRecommendation.SELL_VOL:
        ideas.append(TradeIdea(
            strategy="short_straddle",
            direction="short",
            instrument="ATM straddle",
            rationale="Vol is elevated; collect premium via short straddle.",
            confidence=signal.confidence,
            sizing_pct=min(signal.confidence * 100, 10.0),
        ))
    elif rec == TradeRecommendation.BUY_SKEW:
        ideas.append(TradeIdea(
            strategy="risk_reversal",
            direction="long",
            instrument="25d risk reversal (long call, short put)",
            rationale="Skew is flat; buy skew via risk reversal.",
            confidence=signal.confidence,
            sizing_pct=min(signal.confidence * 100, 15.0),
        ))
    elif rec == TradeRecommendation.SELL_SKEW:
        ideas.append(TradeIdea(
            strategy="risk_reversal",
            direction="short",
            instrument="25d risk reversal (short call, long put)",
            rationale="Skew is steep; sell skew via risk reversal.",
            confidence=signal.confidence,
            sizing_pct=min(signal.confidence * 100, 15.0),
        ))
    elif rec in (TradeRecommendation.STEEPEN_TS, TradeRecommendation.FLATTEN_TS):
        direction = "long" if rec == TradeRecommendation.STEEPEN_TS else "short"
        ideas.append(TradeIdea(
            strategy="calendar_spread",
            direction=direction,
            instrument="Front-to-back calendar spread",
            rationale=f"Term structure signal: {rec.value}.",
            confidence=signal.confidence,
            sizing_pct=min(signal.confidence * 100, 10.0),
        ))

    if openai_api_key and ideas:
        try:
            ideas = _enrich_with_llm(ideas, signal, openai_api_key)
        except Exception as exc:
            logger.warning("LLM enrichment failed: %s", exc)

    return ideas


def _enrich_with_llm(
    ideas: list[TradeIdea],
    signal: CompositeSignal,
    api_key: str,
) -> list[TradeIdea]:
    """Enrich trade ideas with GPT-generated rationale."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        for idea in ideas:
            prompt = (
                f"You are a vol trader. Provide a 1-sentence rationale for: "
                f"{idea.strategy} ({idea.direction}). Context: {signal.rationale}"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            idea.rationale = resp.choices[0].message.content.strip()
    except ImportError:
        pass
    return ideas
