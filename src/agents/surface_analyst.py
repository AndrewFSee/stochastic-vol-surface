"""
Phase 3 placeholder: LLM agent for interpreting vol surface anomalies
in natural language using OpenAI GPT-4.

TODO:
- Implement surface anomaly detection feature extraction.
- Pass features to OpenAI chat completion API.
- Return natural-language interpretation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SurfaceAnalyst:
    """LLM agent that interprets vol surface anomalies in natural language.

    Requires ``OPENAI_API_KEY`` to be set in the environment.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client: Optional[object] = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError as exc:
                raise ImportError("openai is required: pip install openai") from exc
        return self._client

    def analyse(
        self,
        surface_summary: dict,
        regime: str = "Normal",
    ) -> str:
        """Generate a natural-language interpretation of a vol surface snapshot.

        Parameters
        ----------
        surface_summary:
            Dict with keys: atm_vol, skew_25d, term_slope, bf_spread, etc.
        regime:
            Current vol regime label.

        Returns
        -------
        str
            Natural-language interpretation.
        """
        # TODO: Implement full LLM-based analysis.
        logger.warning("SurfaceAnalyst.analyse() is a Phase 3 stub.")
        return (
            f"[Phase 3 Placeholder] Vol regime: {regime}. "
            f"ATM vol: {surface_summary.get('atm_vol', 'N/A'):.2%}. "
            "Full LLM analysis will be available in Phase 3."
        )
