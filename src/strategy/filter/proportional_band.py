"""Davis-Norman approximation with bands proportional to the target weight."""

from dataclasses import dataclass

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal


@dataclass
class ProportionalBandFilter:
    """Bang-bang DN approximation: rebalance fully when any ticker breaches.

    For each signal, the band is::

        width = max(min_band, base_band * max(target, eps))
        [target - width, target + width]

    If *any* signal is outside its band, the full list is returned (bang-bang
    to target). Otherwise, the empty list is returned (no rebalance).

    Mirrors Jaime's ``run_davis_norman_fake_v0``: a single breach triggers
    a full rebalance to the targets — not to the band edge.
    """

    base_band: float = 0.05
    min_band: float = 0.02

    def filter(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        prices: PriceSnapshot,
    ) -> list[Signal]:
        if not signals:
            return signals

        current_weights = portfolio.positions_weights(prices)

        for s in signals:
            target = s.target_weight
            width = max(self.min_band, self.base_band * max(target, 1e-6))
            lo = max(0.0, target - width)
            hi = target + width
            cur = current_weights.get(s.ticker, 0.0)
            if cur < lo or cur > hi:
                return signals

        return []
