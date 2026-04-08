"""Caps individual signal weights to a maximum."""

from dataclasses import dataclass

from src.domain.trade import Signal


@dataclass
class MaxWeightFilter:
    """Clips each signal weight to ``max_weight``.

    Iterates a few times so that any spillage from earlier clips is
    re-absorbed (matches Jaime's iterative cap).
    """

    max_weight: float = 0.40
    max_iterations: int = 10

    def filter(self, signals: list[Signal]) -> list[Signal]:
        if not signals:
            return signals

        weights = {s.ticker: s.target_weight for s in signals}
        for _ in range(self.max_iterations):
            if not any(w > self.max_weight for w in weights.values()):
                break
            weights = {t: min(w, self.max_weight) for t, w in weights.items()}

        return [
            Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=weights[s.ticker],
                reason=s.reason,
                info=s.info,
            )
            for s in signals
        ]
