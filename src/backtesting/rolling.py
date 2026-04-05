from dataclasses import dataclass

from src.domain.asset import PriceHistory
from src.backtesting.engine import BacktestEngine


@dataclass
class RollingBacktest(BacktestEngine):
    """Backtest that re-initialises the strategy on rolling windows.

    Splits the history into (train, test) window pairs.  For each window
    the strategy is re-initialised with the train data, then simulated on
    the test data.  The portfolio carries over between windows so the
    equity curve is continuous.

    Parameters:
        train_size: days of history for each re-initialisation.
        test_size:  days to simulate per window.
        step_size:  how far the window slides (default = test_size,
                    giving non-overlapping test periods).
    """

    train_size: int = 504
    test_size: int = 63
    step_size: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.step_size is None:
            self.step_size = self.test_size

    def _split_history(self) -> list[tuple[PriceHistory, PriceHistory]]:
        total = len(self.history)
        min_required = self.train_size + self.test_size

        if total < min_required:
            raise ValueError(
                f"History too short: {total} days, "
                f"need at least {min_required} (train={self.train_size} + test={self.test_size})"
            )

        df = self.history.df

        # Build windows backwards from the end so the last test window
        # always finishes on the last available date.
        raw = []
        end = total
        while end - min_required >= 0:
            test_start = end - self.test_size
            train_start = test_start - self.train_size
            raw.append((train_start, test_start, end))
            end -= self.step_size

        # Reverse to chronological order
        windows = []
        for train_start, test_start, test_end in reversed(raw):
            warmup = PriceHistory(self.universe, df.iloc[train_start:test_start])
            sim = PriceHistory(self.universe, df.iloc[test_start:test_end])
            windows.append((warmup, sim))

        return windows
