from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MertonDavisNormanModel:
    """Optimal portfolio allocation with no-transaction bands.

    All methods accept dicts (mu, costs) and DataFrames (cov).
    Ticker ordering is derived from the covariance DataFrame.
    """

    gamma: float
    r: float

    def __post_init__(self):
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")

    def optimal_weights(
        self,
        mu: dict[str, float],
        cov: pd.DataFrame,
    ) -> dict[str, float]:
        """Merton optimal weights.

        w* = (1 / (1 - gamma)) * Sigma^{-1} * (mu - r)
        """
        tickers = list(cov.columns)
        mu_arr = np.array([mu[t] for t in tickers])
        cov_arr = cov.values

        excess = mu_arr - self.r
        weights = np.linalg.solve(cov_arr, excess) / (1 - self.gamma)
        return dict(zip(tickers, weights))

    def no_trade_bands(
        self,
        mu: dict[str, float],
        cov: pd.DataFrame,
        buy_costs: dict[str, float],
        sell_costs: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Optimal weights + Davis-Norman no-transaction bands.

        Returns:
            Tuple (weights, lower, upper) as dicts ticker -> value.
        """
        tickers = list(cov.columns)
        weights = self.optimal_weights(mu, cov)

        w_arr = np.array([weights[t] for t in tickers])
        sigmas = np.sqrt(np.diag(cov.values))
        bc = np.array([buy_costs.get(t, 0.0) for t in tickers])
        sc = np.array([sell_costs.get(t, 0.0) for t in tickers])

        factor = (
            3.0 * w_arr ** 2 * (1 - w_arr) ** 2 * sigmas ** 2
            / (4.0 * (1.0 - self.gamma))
        )

        delta_lower = np.cbrt(factor * bc)
        delta_upper = np.cbrt(factor * sc)

        lower = dict(zip(tickers, w_arr - delta_lower))
        upper = dict(zip(tickers, w_arr + delta_upper))
        return weights, lower, upper
