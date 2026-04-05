from dataclasses import dataclass

from src.domain.trade import Signal


@dataclass(frozen=True)
class MinWeightFilter:
    """Discard signals with target_weight below a minimum threshold.

    Signals with target_weight == 0.0 (liquidation) are always kept.
    Required by project rules: no position below 1% of portfolio.
    """

    min_weight: float = 0.01

    def filter(self, signals: list[Signal]) -> list[Signal]:
        return [
            s for s in signals
            if s.target_weight >= self.min_weight or s.target_weight == 0.0
        ]
