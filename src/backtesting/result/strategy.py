from dataclasses import dataclass, field

import pandas as pd

from src.domain.asset import PriceHistory
from .base import BacktestResult


@dataclass(frozen=True)
class StrategyResult:
    """Result for a single strategy, potentially across multiple windows.

    For a standard backtest (one window), ``windows`` has one element
    and ``combined`` is that same result.  For a rolling backtest,
    ``windows`` has many elements and ``combined`` concatenates them
    into a single continuous BacktestResult.
    """

    windows: list[BacktestResult]

    _combined: BacktestResult = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_combined", self._build_combined())

    @property
    def combined(self) -> BacktestResult:
        return self._combined

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def window(self, index: int) -> BacktestResult:
        return self.windows[index]

    # ── Delegate common access to combined ──────────────────────────────

    @property
    def metrics(self) -> dict:
        return self._combined.metrics

    @property
    def summary(self) -> pd.Series:
        return self._combined.summary

    # ── Plots (delegate to combined) ─────────────────────────────────────

    def plot_value(self, **kwargs):
        return self._combined.plot_value(**kwargs)

    def plot_returns(self, **kwargs):
        return self._combined.plot_returns(**kwargs)

    def plot_drawdown(self, **kwargs):
        return self._combined.plot_drawdown(**kwargs)

    def plot_weights(self, **kwargs):
        return self._combined.plot_weights(**kwargs)

    def plot_target_vs_actual(self, ticker: str, **kwargs):
        return self._combined.plot_target_vs_actual(ticker, **kwargs)

    def plot_asset_values(self, **kwargs):
        return self._combined.plot_asset_values(**kwargs)

    def plot_rolling_sharpe(self, **kwargs):
        return self._combined.plot_rolling_sharpe(**kwargs)

    def plot_rolling_volatility(self, **kwargs):
        return self._combined.plot_rolling_volatility(**kwargs)

    def plot_return_distribution(self, **kwargs):
        return self._combined.plot_return_distribution(**kwargs)

    def plot_monthly_heatmap(self, **kwargs):
        return self._combined.plot_monthly_heatmap(**kwargs)

    def plot_annual_returns(self, **kwargs):
        return self._combined.plot_annual_returns(**kwargs)

    def print_summary(self) -> None:
        self._combined.print_summary()

    # ── Build combined ──────────────────────────────────────────────────

    def _build_combined(self) -> BacktestResult:
        if len(self.windows) == 1:
            return self.windows[0]

        first = self.windows[0]

        all_snapshots = []
        all_signals = []
        all_orders = []
        all_trades = []
        price_dfs = []

        for w in self.windows:
            all_snapshots.extend(w.snapshots)
            all_signals.extend(w.signals)
            all_orders.extend(w.orders)
            all_trades.extend(w.trades)
            price_dfs.append(w.prices.df)

        combined_df = pd.concat(price_dfs)
        combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
        combined_prices = PriceHistory(first.universe, combined_df)

        return BacktestResult(
            initial_snapshot=first.initial_snapshot,
            snapshots=all_snapshots,
            prices=combined_prices,
            universe=first.universe,
            signals=all_signals,
            orders=all_orders,
            trades=all_trades,
            risk_free_rate=first.risk_free_rate,
        )
