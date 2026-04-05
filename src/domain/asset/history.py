from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import Universe
from .snapshot import PriceSnapshot


class PriceHistoryError(Exception):
    ...


class PriceHistory:
    """Time series of prices for all assets in a universe."""

    def __init__(self, universe: Universe, df: pd.DataFrame, fill_na: bool = False):
        """
        Args:
            universe: Asset universe.
            df: Price DataFrame (DatetimeIndex, one column per ticker).
            fill_na: How to handle NaN after the first valid row.
                ``"ffill"`` forward-fills gaps (default).
                ``"drop"`` drops any row with NaN.
        """
        missing = set(universe.tickers) - set(df.columns)
        if missing:
            raise PriceHistoryError(f"Missing tickers in DataFrame: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise PriceHistoryError(f"df index must be pd.DatetimeIndex. Current: {type(df.index)}")

        self._universe = universe
        data = df[universe.tickers].copy()

        if not data.empty:
            first_valid = data.apply(lambda col: col.first_valid_index()).max()
            if first_valid is not None:
                data = data.loc[first_valid:]

            if fill_na:
                data = data.ffill()
            else:
                data = data.dropna()

        self._df = data

    # Properties

    @property
    def universe(self) -> Universe:
        return self._universe

    @property
    def df(self) -> pd.DataFrame:
        """Copy of the internal DataFrame."""
        return self._df.copy()

    @property
    def tickers(self) -> list[str]:
        return self._universe.tickers

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._df.index

    def __len__(self) -> int:
        return len(self._df)

    def copy(self) -> PriceHistory:
        """Independent copy (new DataFrame, same Universe)."""
        return PriceHistory(self._universe, self._df.copy())

    def select(self, tickers: list[str]) -> PriceHistory:
        """Return a sub-history with only the given tickers."""
        sub_universe = self._universe.select(tickers)
        return PriceHistory(sub_universe, self._df[sub_universe.tickers].copy())

    # Slicing

    def last(self, n: int) -> pd.DataFrame:
        """Last n rows of price history."""
        return self._df.iloc[-n:]

    def between(self, start, end) -> pd.DataFrame:
        """Prices between two dates (inclusive)."""
        return self._df.loc[start:end]

    def at(self, date) -> PriceSnapshot:
        """Get prices at a specific date as a snapshot."""
        return PriceSnapshot(
            universe=self._universe,
            date=date,
            prices=self._df.loc[date].to_dict(),
        )

    # Mutation

    def append(self, snapshot: PriceSnapshot) -> None:
        """Append a single day's prices."""
        row = pd.DataFrame([snapshot.prices], index=[snapshot.date])
        self._df = pd.concat([self._df, row])

    def trim(self, max_size: int) -> None:
        """Keep only the last max_size rows, discarding older data."""
        if len(self._df) > max_size:
            self._df = self._df.iloc[-max_size:]

    # Technical helpers

    def log_returns(self, dropna: bool = True) -> pd.DataFrame:
        """Daily log returns for all assets."""
        ret = np.log(self._df / self._df.shift(1))
        return ret.dropna() if dropna else ret

    def sma(self, window: int) -> pd.DataFrame:
        """Simple moving average per asset."""
        return self._df.rolling(window=window).mean()

    def sigma(self, window: int) -> pd.DataFrame:
        """Rolling standard deviation of prices per asset."""
        return self._df.rolling(window=window).std()

    def rolling_return_mean(self, window: int, trading_days: int = 252) -> pd.DataFrame:
        """Rolling annualized mean of log returns per asset."""
        log_ret = np.log(self._df / self._df.shift(1))
        return log_ret.rolling(window=window).mean() * trading_days

    def rolling_volatility(self, window: int, trading_days: int = 252) -> pd.DataFrame:
        """Rolling annualized volatility per asset."""
        log_ret = np.log(self._df / self._df.shift(1))
        return log_ret.rolling(window=window).std() * np.sqrt(trading_days)

    def covariance(self, trading_days: int = 252) -> np.ndarray:
        """Annualized covariance matrix from full history."""
        log_ret = np.log(self._df / self._df.shift(1)).dropna()
        return log_ret.cov().values * trading_days
