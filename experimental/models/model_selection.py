"""AIC/BIC model comparison across parametric vol models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelFitResult:
    name: str
    n_params: int
    n_obs: int
    sse: float          # sum of squared errors
    params: dict[str, float]

    @property
    def log_likelihood(self) -> float:
        """Gaussian log-likelihood assuming i.i.d. normal errors."""
        n = self.n_obs
        sigma2 = self.sse / n
        if sigma2 <= 0:
            return -1e15
        return -n / 2 * (math.log(2 * math.pi * sigma2) + 1)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.n_params * math.log(self.n_obs) - 2 * self.log_likelihood

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sse / self.n_obs)


def compare_models(results: list[ModelFitResult]) -> "import pandas; pandas.DataFrame":
    """Return a DataFrame ranked by AIC."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append(
            dict(
                model=r.name,
                n_params=r.n_params,
                n_obs=r.n_obs,
                rmse=r.rmse,
                log_lik=r.log_likelihood,
                aic=r.aic,
                bic=r.bic,
            )
        )
    df = pd.DataFrame(rows)
    df["delta_aic"] = df["aic"] - df["aic"].min()
    df["delta_bic"] = df["bic"] - df["bic"].min()
    return df.sort_values("aic").reset_index(drop=True)


def akaike_weights(aics: np.ndarray) -> np.ndarray:
    """Compute Akaike weights from AIC values for model averaging."""
    delta = aics - aics.min()
    w = np.exp(-0.5 * delta)
    return w / w.sum()
