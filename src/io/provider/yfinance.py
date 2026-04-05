import pandas as pd
import yfinance as yf

from dataclasses import dataclass

from .base import DataProvider


@dataclass(frozen=True)
class YFinanceProvider(DataProvider):
    """Fetches market data from Yahoo Finance."""

    def _get_prices(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers

        return prices
