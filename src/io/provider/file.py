
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from .base import DataProvider


@dataclass(frozen=True)
class CSVProvider(DataProvider):
    """Reads prices from a CSV file.

    The CSV must have a 'Date' column and one column per ticker.
    """

    filepath: str | Path

    def _get_prices(
        self, tickers: list[str], start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        df = df.loc[start:end, tickers]
        df = df.dropna()
        return df
