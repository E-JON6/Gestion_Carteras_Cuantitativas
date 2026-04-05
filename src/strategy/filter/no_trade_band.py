from dataclasses import dataclass

import pandas as pd

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.merton_davisnorman import MertonDavisNormanModel
from src.strategy.transformers import normalise_weights


@dataclass
class NoTradeBandFilter:
    """Filters out signals whose current weight is inside the DN no-trade band.

    Expects precalculated mu, cov, and costs as dicts/DataFrame.
    Computes Davis-Norman bands and drops signals for tickers that
    are already within their band. Bands are clipped to portfolio
    constraints.

    Args:
        gamma: Risk aversion parameter.
        r: Annualised risk-free rate.
        mu: Expected returns dict (ticker -> float).
        cov: Covariance DataFrame (tickers as index/columns).
        costs: Transaction costs dict (ticker -> float).
        band_scale: Multiplicative factor to widen/narrow bands.
    """

    gamma: float
    r: float
    mu: dict[str, float]
    cov: pd.DataFrame
    costs: dict[str, float]
    band_scale: float = 1.0

    def filter(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        prices: PriceSnapshot,
    ) -> list[Signal]:
        model = MertonDavisNormanModel(gamma=self.gamma, r=self.r)
        optimal, lower, upper = model.no_trade_bands(self.mu, self.cov, self.costs, self.costs)

        # Widen bands by scale factor around optimal
        lower = {t: optimal[t] - (optimal[t] - lower[t]) * self.band_scale for t in optimal}
        upper = {t: optimal[t] + (upper[t] - optimal[t]) * self.band_scale for t in optimal}

        clipped_lo = normalise_weights(lower, portfolio)
        clipped_hi = normalise_weights(upper, portfolio)

        current_weights = portfolio.positions_weights(prices)

        filtered = []
        for signal in signals:
            if signal.ticker not in clipped_lo:
                filtered.append(signal)
                continue
            w_cur = current_weights.get(signal.ticker, 0.0)
            if w_cur < clipped_lo[signal.ticker] or w_cur > clipped_hi[signal.ticker]:
                filtered.append(signal)

        return filtered
