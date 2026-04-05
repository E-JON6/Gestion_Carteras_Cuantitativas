from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataProviderError(Exception):
    ...


@dataclass(frozen=True)
class DataProvider(ABC):
    """Interface for market data providers."""

    def get_prices(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        """Return a DataFrame with adjusted close prices.

        Columns = tickers, Index = dates.
        """
        prices = self._get_prices(tickers, start, end)
        self._validate(prices, tickers)
        return prices

    @abstractmethod
    def _get_prices(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        ...

    def _validate(self, prices: pd.DataFrame, requested: list[str]) -> None:
        """Check that all requested tickers have data. Warn about gaps."""
        # Missing tickers
        missing = set(requested) - set(prices.columns)
        if missing:
            raise DataProviderError(f"No data for tickers: {missing}")

        # Tickers with all NaN
        empty = [t for t in requested if prices[t].isna().all()]
        if empty:
            raise DataProviderError(f"All-NaN data for tickers: {empty}")

        # Warn about partial gaps
        nan_counts = prices.isna().sum()
        for ticker, count in nan_counts[nan_counts > 0].items():
            pct = count / len(prices) * 100
            logger.warning(f"{ticker}: {count} missing days ({pct:.1f}%)")

        # No valid rows
        if prices.dropna().empty:
            raise DataProviderError("No valid rows after dropping NaN")
