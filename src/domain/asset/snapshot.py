from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .universe import Universe


class PriceSnapshotError(Exception):
    ...


@dataclass(frozen=True)
class PriceSnapshot:
    """Prices of all assets at a single point in time."""

    universe: Universe
    date: pd.Timestamp
    prices: dict[str, float]

    def __post_init__(self):
        missing = set(self.universe.tickers) - set(self.prices.keys())
        if missing:
            raise PriceSnapshotError(f"Missing tickers: {missing}")

        leftovers = set(self.prices.keys()) - set(self.universe.tickers)
        if leftovers:
            object.__setattr__(self, "prices", {t: p for t, p in self.prices.items() if t in self.universe.tickers})

        nans = {t for t, p in self.prices.items() if p is None or (isinstance(p, float) and math.isnan(p))}
        if nans:
            raise PriceSnapshotError(f"NaN prices for tickers: {nans}")

        negative = {t: p for t, p in self.prices.items() if p < 0}
        if negative:
            raise PriceSnapshotError(f"Negative prices: {negative}")

    def get_price(self, ticker: str) -> float:
        price = self.prices.get(ticker, None)
        if price is None:
            raise PriceSnapshotError(f"Ticker '{ticker}' not in snapshot")
        return price
    
