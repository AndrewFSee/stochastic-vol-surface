"""
Phase 3 placeholder: LLM agent for narrating the current vol regime
in historical context.

TODO: Implement regime narration with LLM and historical parallel retrieval.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RegimeNarrator:
    """LLM agent that contextualises the current vol regime historically.

    TODO (Phase 3): Retrieve historical parallels and generate narrative.
    """

    def narrate(self, regime: str, vix_level: float) -> str:
        """Generate a regime narration.

        Parameters
        ----------
        regime:
            Current regime label.
        vix_level:
            Current VIX level.

        Returns
        -------
        str
            Natural-language regime context.
        """
        # TODO: Phase 3 — query vector DB of historical vol regimes
        # and generate narrative with LLM.
        logger.warning("RegimeNarrator.narrate() is a Phase 3 stub.")
        return (
            f"[Phase 3 Placeholder] Current regime: {regime} (VIX={vix_level:.1f}). "
            "Full historical narration will be available in Phase 3."
        )
