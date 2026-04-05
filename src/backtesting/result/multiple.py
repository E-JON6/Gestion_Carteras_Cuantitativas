
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .strategy import StrategyResult


@dataclass(frozen=True)
class MultipleResult:
    """Results for all strategies in a backtest run."""

    results: dict[str, StrategyResult]

    # ── Access ──────────────────────────────────────────────────────────

    @property
    def strategies(self) -> list[str]:
        return list(self.results.keys())

    def result(self, strategy: str) -> StrategyResult:
        r = self.results.get(strategy)
        if r is None:
            raise ValueError(f"No result for strategy {strategy!r}")
        return r

    # ── Comparison DataFrames ───────────────────────────────────────────

    @property
    def summary_df(self) -> pd.DataFrame:
        """One row per strategy with all metrics."""
        return pd.DataFrame(
            {name: r.combined.summary for name, r in self.results.items()}
        ).T

    @property
    def value_df(self) -> pd.DataFrame:
        """Portfolio value over time, one column per strategy."""
        return pd.DataFrame(
            {name: r.combined.value_over_time for name, r in self.results.items()}
        )

    @property
    def returns_df(self) -> pd.DataFrame:
        """Daily returns, one column per strategy."""
        return pd.DataFrame(
            {name: r.combined.returns_over_time for name, r in self.results.items()}
        )

    @property
    def cumulative_returns_df(self) -> pd.DataFrame:
        """Cumulative returns, one column per strategy."""
        return pd.DataFrame(
            {name: r.combined.cumulative_returns for name, r in self.results.items()}
        )

    @property
    def drawdown_df(self) -> pd.DataFrame:
        """Drawdown over time, one column per strategy."""
        return pd.DataFrame(
            {name: r.combined.drawdown for name, r in self.results.items()}
        )

    # ── Plots ───────────────────────────────────────────────────────────

    @staticmethod
    def _style_ax(ax, title: str, ylabel: str):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    def plot_value(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))
        for i, name in enumerate(self.strategies):
            self.value_df[name].plot(ax=ax, label=name, color=cmap(i), linewidth=1.2)
        self._style_ax(ax, "Portfolio Value", "Value ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_returns(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))
        for i, name in enumerate(self.strategies):
            self.cumulative_returns_df[name].plot(ax=ax, label=name, color=cmap(i), linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
        self._style_ax(ax, "Cumulative Returns", "Return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_drawdown(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))
        for i, name in enumerate(self.strategies):
            dd = self.drawdown_df[name]
            ax.fill_between(dd.index, dd.values, 0, color=cmap(i), alpha=0.15)
            dd.plot(ax=ax, label=name, color=cmap(i), linewidth=0.8)
        self._style_ax(ax, "Drawdown", "Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_rolling_sharpe(self, window: int = 252, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))
        for i, name in enumerate(self.strategies):
            ret = self.returns_df[name].dropna()
            rs = (ret.rolling(window).mean() / ret.rolling(window).std()) * np.sqrt(252)
            rs.plot(ax=ax, label=name, color=cmap(i), linewidth=1, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        self._style_ax(ax, f"Rolling Sharpe Ratio ({window}d)", "Sharpe")
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_rolling_volatility(self, window: int = 63, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))
        for i, name in enumerate(self.strategies):
            ret = self.returns_df[name].dropna()
            rv = ret.rolling(window).std() * np.sqrt(252)
            rv.plot(ax=ax, label=name, color=cmap(i), linewidth=1, alpha=0.8)
        self._style_ax(ax, f"Rolling Volatility ({window}d, annualised)", "Volatility")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_annual_returns(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        cmap = plt.cm.get_cmap("tab10", len(self.strategies))

        annual = {}
        for name in self.strategies:
            ret = self.returns_df[name].dropna()
            annual[name] = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        annual_df = pd.DataFrame(annual)
        annual_df.index = annual_df.index.year

        x = np.arange(len(annual_df))
        n = len(self.strategies)
        width = 0.8 / n

        for i, name in enumerate(self.strategies):
            ax.bar(x + i * width, annual_df[name], width, label=name, color=cmap(i), alpha=0.85)

        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels(annual_df.index, fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.5)
        self._style_ax(ax, "Annual Returns", "Return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.legend(title="Strategy", fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def print_summary(self) -> None:
        print(self.summary_df.to_string())
