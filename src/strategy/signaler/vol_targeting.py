"""Scales signal weights to target a specific portfolio volatility."""

from dataclasses import dataclass, field

from src.domain.asset import PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.sigma import SigmaEstimator, HistoricalVolatility

import numpy as np


@dataclass
class VolTargetingSignaler:
    """Scales incoming signals so the portfolio targets a given volatility.

    Computes the current realised volatility of each ticker, estimates
    the portfolio volatility from the signal weights, and rescales all
    weights by ``sigma_target / sigma_portfolio``.

    If ``max_exposure`` is set, the scale factor is capped so total
    exposure does not exceed that value.

    Args:
        sigma_target: Target annualised portfolio volatility (e.g. 0.10 = 10%).
        sigma_estimator: Estimator for per-asset volatility.
        max_exposure: Maximum total weight after scaling (e.g. 1.0).
    """

    sigma_target: float = 0.10
    sigma_estimator: SigmaEstimator = field(default_factory=HistoricalVolatility)
    max_exposure: float = 1.0

    def scale(
        self,
        signals: list[Signal],
        history: PriceHistory,
    ) -> list[Signal]:
        """Scale signal weights to target volatility."""
        if not signals:
            return signals

        tickers = [s.ticker for s in signals]
        available = [t for t in tickers if t in history.tickers]
        if len(available) < 2:
            return signals

        try:
            sub = history.select(available)
            sigma_dict = self.sigma_estimator.estimate(sub)
        except (ValueError, Exception):
            return signals

        weights = {s.ticker: s.target_weight for s in signals}

        # Estimate portfolio volatility as weighted average of asset vols
        # (simplified — ignores correlations for speed)
        port_vol = sum(
            abs(weights.get(t, 0.0)) * sigma_dict[t]
            for t in available
            if t in sigma_dict
        )

        if port_vol <= 0:
            return signals

        scale = self.sigma_target / port_vol
        scale = min(scale, self.max_exposure / max(sum(abs(w) for w in weights.values()), 1e-9))

        if abs(scale - 1.0) < 0.01:
            return signals

        return [
            Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=s.target_weight * scale,
                reason=s.reason,
                info={**(s.info or {}), "vol_scale": scale, "port_vol": port_vol},
            )
            for s in signals
        ]
