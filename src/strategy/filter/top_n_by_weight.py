"""Filter that keeps only the top-N signals by absolute target weight."""

from dataclasses import dataclass

from src.domain.trade import Signal


@dataclass
class TopNByWeightFilter:
    """Keeps only the top N signals ranked by absolute target weight.

    Merton assigns higher weight to tickers with better risk/return
    profile, so ranking by weight is a natural selection criterion
    coherent with the model.

    Args:
        n: Number of top tickers to keep.
    """

    n: int = 5

    def filter(self, signals: list[Signal]) -> list[Signal]:
        if len(signals) <= self.n:
            return signals

        ranked = sorted(signals, key=lambda s: abs(s.target_weight), reverse=True)
        return ranked[:self.n]
