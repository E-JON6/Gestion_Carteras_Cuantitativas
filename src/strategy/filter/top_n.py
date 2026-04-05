"""Filter that keeps only the top-N signals by Omega ranking."""

from dataclasses import dataclass

from src.domain.asset import PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.omega_ranker import OmegaRanker


@dataclass
class TopNFilter:
    """Keeps only the top N signals ranked by Omega ratio.

    Signals for tickers not in the top N are dropped.
    An optional ``keep`` set allows extra tickers to pass through
    regardless of ranking (useful for frozen positions).

    Args:
        n: Number of top tickers to keep.
        omega_threshold: Minimum Omega threshold for ranking.
    """

    n: int = 5
    omega_threshold: float = 0.0

    def filter(
        self,
        signals: list[Signal],
        history: PriceHistory,
        keep: set[str] | None = None,
    ) -> list[Signal]:
        if not signals:
            return signals

        tickers = [s.ticker for s in signals]
        ranker = OmegaRanker(threshold=self.omega_threshold)
        top = ranker.top(history, self.n, tickers)
        top_set = set(top)

        if keep:
            top_set |= keep

        return [s for s in signals if s.ticker in top_set]
