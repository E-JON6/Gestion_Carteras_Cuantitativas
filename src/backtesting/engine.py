from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from src.domain.asset import Universe, PriceHistory
from src.domain.portfolio import Portfolio, PortfolioSnapshot
from src.domain.trade import Trade, Order, Signal, Broker
from src.strategy.base import Strategy

from .result import BacktestResult, StrategyResult, MultipleResult


@dataclass
class BacktestEngine(ABC):
    """Abstract backtesting engine.

    Orchestrates: Strategy -> Broker -> Portfolio.

    Subclasses only implement ``_split_history`` which returns one or
    more (warmup, simulation) window pairs.  The engine re-initialises
    the strategy at the start of each window and produces a
    ``BacktestResult`` per window, wrapped in a ``StrategyResult``.
    """

    portfolio: Portfolio
    universe: Universe
    strategies: list[Strategy]
    history: PriceHistory
    broker: Broker | None = None
    risk_free_rate: float = 0.0

    def __post_init__(self):
        self._validate_strategies()

    def _validate_strategies(self) -> None:
        names = []
        for strat in self.strategies:
            if strat.name in names:
                raise ValueError(f"multiple strategies with name {strat.name!r}")
            names.append(strat.name)

            if set(strat.universe.tickers) != set(self.universe.tickers):
                raise ValueError(
                    f"strategy {strat.name!r} universe {strat.universe.tickers} "
                    f"does not match engine universe {self.universe.tickers}"
                )

    # ── Public ──────────────────────────────────────────────────────────

    def run(self) -> MultipleResult:
        """Run the full backtest (each strategy from the same initial state)."""
        windows = self._split_history()
        broker = self.broker or Broker(transaction_costs=self.universe.transaction_costs)

        results: dict[str, StrategyResult] = {}
        for strat in self.strategies:
            self.portfolio.reset()

            window_results: list[BacktestResult] = []
            for warmup, sim in windows:
                initial_snapshot = self.portfolio.snapshot(sim.at(sim.dates[0]))

                strat.initialize(warmup.copy(), self.portfolio)
                snapshots, signals, orders, trades = self._simulate(
                    strat, sim, broker,
                )

                window_results.append(BacktestResult(
                    initial_snapshot=initial_snapshot,
                    snapshots=snapshots,
                    prices=sim,
                    universe=self.universe,
                    signals=signals,
                    orders=orders,
                    trades=trades,
                    risk_free_rate=self.risk_free_rate,
                ))

            results[strat.name] = StrategyResult(windows=window_results)

        return MultipleResult(results)

    # ── Abstract ────────────────────────────────────────────────────────

    @abstractmethod
    def _split_history(self) -> list[tuple[PriceHistory, PriceHistory]]:
        """Return (warmup, simulation) window pairs.

        Single-window backtests return a one-element list.
        Rolling backtests return many pairs.
        """

    # ── Simulation ──────────────────────────────────────────────────────

    def _simulate(
        self,
        strategy: Strategy,
        sim_history: PriceHistory,
        broker: Broker,
    ) -> tuple[list[PortfolioSnapshot], list[list[Signal]], list[list[Order]], list[list[Trade]]]:
        """Day-by-day simulation loop."""
        snapshots = []
        all_signals = []
        all_orders = []
        all_trades = []

        for date in sim_history.dates:
            price_snapshot = sim_history.at(date)

            signals = strategy.on_step(price_snapshot, self.portfolio)
            orders, trades = broker.execute(self.portfolio, signals, price_snapshot)

            all_signals.append(signals)
            all_orders.append(orders)
            all_trades.append(trades)
            snapshots.append(self.portfolio.snapshot(price_snapshot))

            self._on_after_step(strategy)

        return snapshots, all_signals, all_orders, all_trades

    def _on_after_step(self, strategy: Strategy) -> None:
        """Hook called after each simulation step. Override in subclasses."""
