"""Manages frozen assets: tickers that exited top-N but stay until their weight drops."""

from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio


@dataclass
class FrozenAssetManager:
    """Tracks tickers that left the top-N but remain in portfolio.

    A frozen ticker keeps its current weight (no new signal) until
    its weight drops below ``thaw_threshold``, at which point it
    gets liquidated.

    Args:
        thaw_threshold: Weight below which a frozen ticker is liquidated (default 0.02 = 2%).
    """

    thaw_threshold: float = 0.02
    _frozen: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def frozen(self) -> set[str]:
        return set(self._frozen)

    def update(
        self,
        top_tickers: set[str],
        portfolio: Portfolio,
        prices: PriceSnapshot,
    ) -> set[str]:
        """Update frozen set and return tickers to liquidate.

        Args:
            top_tickers: Current top-N tickers from the filter.
            portfolio: Current portfolio state.
            prices: Current prices.

        Returns:
            Set of tickers that should be liquidated (weight < threshold).
        """
        current_weights = portfolio.positions_weights(prices)

        # Tickers that were in portfolio but dropped out of top-N → freeze
        held_tickers = {t for t, w in current_weights.items() if w > 1e-9}
        new_frozen = (held_tickers - top_tickers) - self._frozen
        self._frozen |= new_frozen

        # Tickers that came back to top-N → unfreeze
        self._frozen -= top_tickers

        # Check which frozen tickers should be liquidated
        to_liquidate = set()
        for ticker in list(self._frozen):
            w = current_weights.get(ticker, 0.0)
            if w < self.thaw_threshold:
                to_liquidate.add(ticker)
                self._frozen.discard(ticker)

        return to_liquidate

    def reset(self) -> None:
        self._frozen = set()
