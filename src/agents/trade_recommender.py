"""
Phase 3 placeholder: LLM agent for suggesting specific vol-arb trades
with quantitative rationale.

TODO: Implement trade recommendation logic with LLM.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TradeRecommender:
    """LLM agent that suggests vol-arb trades with rationale.

    TODO (Phase 3): Implement full trade recommendation pipeline.
    """

    def recommend(
        self,
        signals: dict,
        regime: str,
        surface_summary: dict,
    ) -> str:
        """Generate trade recommendation.

        Parameters
        ----------
        signals:
            Composite signal dict from :mod:`src.signals.composite`.
        regime:
            Current vol regime.
        surface_summary:
            Summary statistics of the current surface.

        Returns
        -------
        str
            Natural-language trade recommendation.
        """
        # TODO: Implement Phase 3 LLM-based trade recommendation.
        action = signals.get("action", "flat")
        confidence = signals.get("confidence", 0.0)
        logger.warning("TradeRecommender.recommend() is a Phase 3 stub.")
        return (
            f"[Phase 3 Placeholder] Action: {action} (confidence: {confidence:.2f}). "
            "Full LLM trade recommendation will be available in Phase 3."
        )
