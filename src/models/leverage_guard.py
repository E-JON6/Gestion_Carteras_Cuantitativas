"""Detects when portfolio leverage drifts beyond a safe threshold."""

from dataclasses import dataclass

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio


@dataclass
class LeverageGuard:
    """Detects when current portfolio leverage exceeds a threshold.

    Use this to force a rebalance even when ``rebalance_every``
    hasn't triggered yet.  The strategy checks ``is_triggered()``
    each step and skips the rebalance counter if True.

    Args:
        max_leverage: Absolute leverage limit (e.g. 2.0).
        tolerance: Fraction above max_leverage that triggers (default 0.05 = 5%).
    """

    max_leverage: float
    tolerance: float = 0.05

    @property
    def threshold(self) -> float:
        return self.max_leverage * (1 + self.tolerance)

    def is_triggered(self, portfolio: Portfolio, prices: PriceSnapshot) -> bool:
        """True if current gross leverage exceeds threshold."""
        nav = portfolio.total_value(prices)
        if nav <= 0:
            return False

        gross = portfolio.total_positions_value(prices)
        leverage = abs(gross) / nav
        return leverage > self.threshold
