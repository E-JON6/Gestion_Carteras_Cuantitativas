from dataclasses import dataclass

from src.domain.asset import PriceHistory
from src.strategy.base import Strategy
from src.backtesting.backtest import Backtest


@dataclass
class WalkForwardBacktest(Backtest):
    """Walk-forward backtest with a fixed rolling window.

    Same warmup split as ``Backtest``, but after each simulation step
    the strategy's history is trimmed to ``window_size`` rows so it
    always sees a fixed-length window that slides forward.
    """

    window_size: int = 252

    def _on_after_step(self, strategy: Strategy) -> None:
        strategy.trim_history(self.window_size)
