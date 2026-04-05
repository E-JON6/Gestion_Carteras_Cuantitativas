from dataclasses import dataclass

import pandas as pd

from src.domain.asset import PriceHistory
from src.backtesting.engine import BacktestEngine


@dataclass
class Backtest(BacktestEngine):
    """Standard backtest with configurable warmup.

    The first ``warmup_size`` days are used for strategy warmup.
    The remaining days are simulated.

    ``warmup_size`` accepts:
      - int: number of rows (e.g. 252). Use 0 for no warmup.
      - str or Timestamp: first simulation date (warmup = everything before it)
    """

    warmup_size: int | str | pd.Timestamp = 252

    def _split_history(self) -> list[tuple[PriceHistory, PriceHistory]]:
        df = self.history.df
        split = self._resolve_split(df)
        warmup = PriceHistory(self.universe, df.iloc[:split])
        sim = PriceHistory(self.universe, df.iloc[split:])
        return [(warmup, sim)]

    def _resolve_split(self, df: pd.DataFrame) -> int:
        if isinstance(self.warmup_size, int):
            return self.warmup_size

        date = pd.Timestamp(self.warmup_size)
        mask = df.index < date
        return int(mask.sum())
