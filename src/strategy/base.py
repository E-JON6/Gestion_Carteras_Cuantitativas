
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.domain.asset import Universe, PriceSnapshot, PriceHistory
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal


@dataclass
class Strategy(ABC):
    """Base class for all trading strategies."""

    universe: Universe
    name: str = ""

    _initialized: bool = field(default=False, init=False, repr=False)
    _history: PriceHistory = field(init=False, repr=False)

    @property
    def initialized(self):
        return self._initialized

    @property
    def history(self):
        return self._history

    def initialize(self, history: PriceHistory, portfolio: Portfolio) -> None:
        """Prepare the strategy with warmup data and portfolio reference."""
        self.reset()
        self._history = history
        self._warmup(portfolio)
        self._initialized = True

    def reset(self) -> None:
        """Reset strategy to pre-initialised state.

        Subclasses that hold internal state (caches, models, etc.)
        should override this and call ``super().reset()``.
        """
        self._initialized = False
        self._history = None

    @abstractmethod
    def _warmup(self, portfolio: Portfolio) -> None:
        """Process warmup data. Called automatically by initialize()."""

    def on_step(
        self,
        prices: PriceSnapshot,
        portfolio: Portfolio,
    ) -> list[Signal]:
        """Generate trade signals for this step."""
        self._history.append(prices)
        return self._generate_signals(prices, portfolio)

    def trim_history(self, max_size: int) -> None:
        """Keep only the last ``max_size`` rows of internal history."""
        self._history.trim(max_size)

    @abstractmethod
    def _generate_signals(
        self,
        prices: PriceSnapshot,
        portfolio: Portfolio,
    ) -> list[Signal]:
        """Produce raw trade signals. Subclasses implement this."""
