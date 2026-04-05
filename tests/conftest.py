"""Shared fixtures for broker, portfolio and backtest tests."""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.domain.asset import Asset, Universe, PriceSnapshot, PriceHistory
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal, Broker
from src.strategy.base import Strategy


DATE = pd.Timestamp("2024-06-01")


# ---------------------------------------------------------------------------
# Dummy strategy for backtest tests
# ---------------------------------------------------------------------------

@dataclass
class FixedWeightStrategy(Strategy):
    """Always signals the same target weights. Useful for testing."""

    weights: dict[str, float] = field(default_factory=dict)
    rebalance_every: int = 1
    _step_count: int = field(default=0, init=False, repr=False)

    def _warmup(self, portfolio: Portfolio) -> None:
        pass

    def reset(self) -> None:
        super().reset()
        self._step_count = 0

    def _generate_signals(self, prices: PriceSnapshot, portfolio: Portfolio) -> list[Signal]:
        self._step_count += 1
        if self._step_count % self.rebalance_every != 0:
            return []
        return [
            Signal(date=prices.date, ticker=t, target_weight=w, reason=None)
            for t, w in self.weights.items()
        ]


def make_price_history(universe: Universe, n_days: int, start_date="2020-01-01", seed=42) -> PriceHistory:
    """Generate synthetic price history with random walk."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    data = {}
    for ticker in universe.tickers:
        returns = rng.normal(0.0005, 0.02, size=n_days)
        prices = 100.0 * np.cumprod(1 + returns)
        data[ticker] = prices
    return PriceHistory(universe, pd.DataFrame(data, index=dates))


def make_universe(*tickers_with_cost: tuple[str, float]) -> Universe:
    assets = tuple(
        Asset(ticker=t, transaction_cost=c) for t, c in tickers_with_cost
    )
    return Universe(assets=assets)


def make_prices(universe: Universe, price_map: dict[str, float], date=DATE) -> PriceSnapshot:
    return PriceSnapshot(universe=universe, date=date, prices=price_map)


def make_signal(ticker: str, target_weight: float, date=DATE, reason=None) -> Signal:
    return Signal(date=date, ticker=ticker, target_weight=target_weight, reason=reason)


# ---------------------------------------------------------------------------
# Standard 2-asset universe: AAPL @ $100, MSFT @ $200
# ---------------------------------------------------------------------------

@pytest.fixture
def universe_2():
    return make_universe(("AAPL", 0.001), ("MSFT", 0.001))


@pytest.fixture
def prices_2(universe_2):
    return make_prices(universe_2, {"AAPL": 100.0, "MSFT": 200.0})


@pytest.fixture
def broker_2(universe_2):
    return Broker(transaction_costs=universe_2.transaction_costs)


# ---------------------------------------------------------------------------
# Standard 3-asset universe: AAPL @ $100, MSFT @ $200, GOOG @ $150
# ---------------------------------------------------------------------------

@pytest.fixture
def universe_3():
    return make_universe(("AAPL", 0.001), ("MSFT", 0.002), ("GOOG", 0.0))


@pytest.fixture
def prices_3(universe_3):
    return make_prices(universe_3, {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 150.0})


@pytest.fixture
def broker_3(universe_3):
    return Broker(transaction_costs=universe_3.transaction_costs)
