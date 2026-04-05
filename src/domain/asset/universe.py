from dataclasses import dataclass

from .base import Asset


class UniverseError(Exception):
    ...

@dataclass(frozen=True)
class Universe:
    """Single source of truth for the tradeable asset universe."""

    assets: tuple[Asset, ...]


    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def transaction_costs(self) -> dict[str, float]:
        """Transaction cost per ticker (0.0 if not defined)."""
        return {a.ticker: (a.transaction_cost or 0.0) for a in self.assets}


    def asset(self, ticker: str) -> Asset:
        """Get an Asset by ticker."""
        for a in self.assets:
            if a.ticker == ticker:
                return a
        raise UniverseError(f"Ticker '{ticker}' not in universe")

    def transaction_cost(self, ticker: str) -> float:
        """Get transaction cost for a specific ticker."""
        return self.asset(ticker).transaction_cost or 0.0


    def select(self, tickers: list[str]) -> "Universe":
        """Return a sub-universe with only the given tickers."""
        return Universe(assets=tuple(a for a in self.assets if a.ticker in tickers))

    def __contains__(self, ticker: str) -> bool:
        return ticker in self.tickers

    def __getitem__(self, ticker: str) -> Asset:
        return self.asset(ticker)

    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self):
        return iter(self.assets)
