from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.domain.asset import Universe, PriceHistory
from src.domain.portfolio import PortfolioSnapshot
from src.domain.trade import Trade, Signal, Order, AcceptedOrder, RejectedOrder, InvalidOrder
from src.metrics.performance import PerformanceReport


@dataclass(frozen=True)
class BacktestResult:
    """Holds all backtest outputs and provides quick access to metrics and plots."""

    universe: Universe
    prices: PriceHistory
    initial_snapshot: PortfolioSnapshot
    snapshots: list[PortfolioSnapshot]
    signals: list[list[Signal]]
    orders: list[list[Order]]
    trades: list[list[Trade]]

    risk_free_rate: float = 0.0

    # Computed
    metrics: dict = field(init=False, repr=False)

    def __post_init__(self):
        report = PerformanceReport(values=self.value_over_time, risk_free_rate=self.risk_free_rate)
        metrics = report.compute()
        object.__setattr__(self, "metrics", metrics)

    # Trades

    @property
    def total_trades(self) -> int:
        return sum(len(t) for t in self.trades)

    @property
    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            t.to_dict() for day in self.trades for t in day
        )

    # Orders

    @property
    def total_orders(self) -> int:
        return sum(len(o) for o in self.orders)

    @property
    def orders_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            o.to_dict() for day in self.orders for o in day
        )

    @property
    def accepted_orders(self) -> int:
        return sum(1 for day in self.orders for o in day if isinstance(o, AcceptedOrder))

    @property
    def rejected_orders(self) -> int:
        return sum(1 for day in self.orders for o in day if isinstance(o, RejectedOrder))

    @property
    def rejected_orders_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            o.to_dict() for day in self.orders for o in day if isinstance(o, RejectedOrder)
        )

    @property
    def invalid_orders(self) -> int:
        return sum(1 for day in self.orders for o in day if isinstance(o, InvalidOrder))

    @property
    def invalid_orders_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            o.to_dict() for day in self.orders for o in day if isinstance(o, InvalidOrder)
        )

    # Signals

    @property
    def total_signals(self) -> int:
        return sum(len(s) for s in self.signals)

    @property
    def signals_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            s.to_dict() for day in self.signals for s in day
        )

    # Portfolio

    @property
    def tickers(self) -> list[str]:
        return self.universe.tickers

    @property
    def snapshots_df(self) -> pd.DataFrame:
        """All snapshot data as a flat DataFrame."""
        return pd.DataFrame(
            [s.full_dict for s in self.snapshots],
        ).set_index("date")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([s.date for s in self.snapshots])

    @property
    def value_over_time(self) -> pd.Series:
        return pd.Series(
            [s.total_value for s in self.snapshots],
            index=self.dates,
            name="total_value",
        )

    @property
    def returns_over_time(self) -> pd.Series:
        return self.value_over_time.pct_change().dropna()

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.returns_over_time).cumprod() - 1

    @property
    def drawdown(self) -> pd.Series:
        cummax = self.value_over_time.cummax()
        return (self.value_over_time - cummax) / cummax

    @property
    def initial_value(self) -> float:
        return self.initial_snapshot.total_value

    @property
    def final_value(self) -> float:
        return self.value_over_time.iloc[-1]

    @property
    def final_return(self) -> float:
        return self.final_value / self.initial_value - 1

    @property
    def total_costs(self) -> float:
        return sum(t.cost for day in self.trades for t in day)

    @property
    def weights_df(self) -> pd.DataFrame:
        """Actual portfolio weights per ticker over time, including cash."""
        records = []
        for snapshot in self.snapshots:
            record = {"date": snapshot.date}
            record.update(snapshot.positions_weights)
            cash_weight = snapshot.cash / snapshot.total_value if snapshot.total_value != 0 else 0.0
            record["cash"] = cash_weight
            records.append(record)
        return pd.DataFrame(records).set_index("date")

    @property
    def values_df(self) -> pd.DataFrame:
        """Portfolio values breakdown over time."""
        return pd.DataFrame(
            [s.values_dict for s in self.snapshots],
            index=self.dates,
        )

    @property
    def target_weights_df(self) -> pd.DataFrame:
        """Target weights from signals over time."""
        records = []
        for day_signals, snapshot in zip(self.signals, self.snapshots):
            record = {"date": snapshot.date}
            for s in day_signals:
                record[s.ticker] = s.target_weight
            records.append(record)
        return pd.DataFrame(records).set_index("date")

    @property
    def summary(self) -> pd.Series:
        extra = {
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "final_return": self.final_return,
            "total_signals": self.total_signals,
            "total_orders": self.total_orders,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "invalid_orders": self.invalid_orders,
            "total_trades": self.total_trades,
            "total_costs": self.total_costs,
        }
        return pd.Series({**self.metrics, **extra})

    def print_summary(self) -> None:
        print(f"{'Period:':<22} {self.initial_snapshot.date.date()} -> {self.dates[-1].date()} ({len(self.dates)} days)")
        print(f"{'Initial value:':<22} ${self.initial_value:,.2f} (cash=${self.initial_snapshot.cash:,.2f})")
        print(f"{'Final value:':<22} ${self.final_value:,.2f}")
        print(f"{'Total return:':<22} {self.final_return:.2%}")
        print(f"{'Annualized return:':<22} {self.metrics['annualized_return']:.2%}")
        print(f"{'Annualized vol:':<22} {self.metrics['annualized_volatility']:.2%}")
        print(f"{'Sharpe ratio:':<22} {self.metrics['sharpe_ratio']:.4f}")
        print(f"{'Max drawdown:':<22} {self.metrics['max_drawdown']:.2%}")
        print(f"{'Calmar ratio:':<22} {self.metrics['calmar_ratio']:.4f}")
        print(f"{'Total costs:':<22} ${self.total_costs:,.2f}")
        print(f"{'Signals:':<22} {self.total_signals}")
        print(f"{'Orders:':<22} {self.total_orders} (accepted={self.accepted_orders}, rejected={self.rejected_orders}, invalid={self.invalid_orders})")
        print(f"{'Trades:':<22} {self.total_trades}")

    # Plots

    @staticmethod
    def _style_ax(ax, title: str, ylabel: str, grid: bool = True):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("")
        if grid:
            ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    def plot_value(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        self.value_over_time.plot(ax=ax, color="#2563eb", linewidth=1.2, **kwargs)
        self._style_ax(ax, "Portfolio Value", "Value ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_returns(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        self.cumulative_returns.plot(ax=ax, color="#2563eb", linewidth=1.2, **kwargs)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
        self._style_ax(ax, "Cumulative Returns", "Return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_drawdown(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        dd = self.drawdown
        ax.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.25)
        dd.plot(ax=ax, color="#dc2626", linewidth=0.8, **kwargs)
        self._style_ax(ax, "Drawdown", "Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_weights(self, return_ax=False, **kwargs):
        figsize = kwargs.pop("figsize", (12, 5))
        wdf = self.weights_df
        cmap = plt.cm.get_cmap("tab10", len(wdf.columns))
        colors = {col: cmap(i) for i, col in enumerate(wdf.columns)}

        has_negative = (wdf < -1e-9).any().any()

        if has_negative:
            fig, (ax_long, ax_short) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.4),
                                                     sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        else:
            fig, ax_long = plt.subplots(figsize=figsize)
            ax_short = None

        # Long positions
        long_df = wdf.clip(lower=0)
        long_cols = [c for c in long_df.columns if (long_df[c] > 1e-9).any()]
        if long_cols:
            long_df[long_cols].plot.area(
                ax=ax_long, stacked=True, alpha=0.8, linewidth=0.3,
                color=[colors[c] for c in long_cols], **kwargs,
            )
        self._style_ax(ax_long, "Long Positions", "Weight", grid=False)
        ax_long.set_ylim(0, None)
        ax_long.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_long.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

        # Short positions
        if ax_short is not None:
            short_df = wdf.clip(upper=0).abs()
            short_cols = [c for c in short_df.columns if (short_df[c] > 1e-9).any()]
            if short_cols:
                short_df[short_cols].plot.area(
                    ax=ax_short, stacked=True, alpha=0.8, linewidth=0.3,
                    color=[colors[c] for c in short_cols], **kwargs,
                )
            self._style_ax(ax_short, "Short Positions", "Weight", grid=False)
            ax_short.set_ylim(0, None)
            ax_short.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax_short.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

        plt.tight_layout()
        if return_ax:
            return (ax_long, ax_short) if ax_short else ax_long
        plt.show()

    def plot_target_vs_actual(self, ticker: str, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        tw = self.target_weights_df
        aw = self.weights_df
        if ticker in tw.columns:
            tw[ticker].plot(ax=ax, label="Target", linestyle="--", alpha=0.6, color="#9ca3af", linewidth=1)
        if ticker in aw.columns:
            aw[ticker].plot(ax=ax, label="Actual", color="#2563eb", linewidth=1.2)
        self._style_ax(ax, f"Target vs Actual: {ticker}", "Weight")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_asset_values(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        vdf = self.values_df
        value_cols = [c for c in vdf.columns if c.endswith("_value") and c != "total_positions_value"]
        asset_data = vdf[value_cols]
        asset_data.columns = [c.replace("_value", "") for c in value_cols]
        cmap = plt.cm.get_cmap("tab10", len(asset_data.columns))
        asset_data.plot.area(
            ax=ax, stacked=True, alpha=0.8, linewidth=0.3,
            color=[cmap(i) for i in range(len(asset_data.columns))],
            **kwargs,
        )
        self._style_ax(ax, "Asset Allocation (Value)", "Value ($)", grid=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_rolling_sharpe(self, window: int = 252, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        ret = self.returns_over_time
        rolling_sharpe = (ret.rolling(window).mean() / ret.rolling(window).std()) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax, color="#2563eb", linewidth=1, **kwargs)
        ax.axhline(0, color="grey", linewidth=0.5)
        self._style_ax(ax, f"Rolling Sharpe Ratio ({window}d)", "Sharpe")
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_rolling_volatility(self, window: int = 63, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 4)))
        rolling_vol = self.returns_over_time.rolling(window).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax, color="#f59e0b", linewidth=1, **kwargs)
        self._style_ax(ax, f"Rolling Volatility ({window}d, annualised)", "Volatility")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_return_distribution(self, bins: int = 80, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 4)))
        ret = self.returns_over_time
        ax.hist(ret, bins=bins, alpha=0.7, color="#2563eb", edgecolor="white", linewidth=0.3, **kwargs)
        ax.axvline(ret.mean(), color="#dc2626", linestyle="--", linewidth=1, label=f"mean={ret.mean():.4f}")
        ax.axvline(0, color="grey", linewidth=0.5)
        self._style_ax(ax, "Daily Return Distribution", "Frequency")
        ax.legend(fontsize=9, frameon=False)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_monthly_heatmap(self, return_ax=False, **kwargs):
        ret = self.returns_over_time
        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        pivot = monthly.to_frame("r")
        pivot["year"] = pivot.index.year
        pivot["month"] = pivot.index.month
        hm = pivot.pivot_table(values="r", index="year", columns="month")
        hm.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, max(3, len(hm) * 0.35))))
        im = ax.imshow(hm.values, cmap="RdYlGn", aspect="auto", vmin=-0.08, vmax=0.08)
        ax.set_xticks(range(len(hm.columns)))
        ax.set_xticklabels(hm.columns, fontsize=9)
        ax.set_yticks(range(len(hm.index)))
        ax.set_yticklabels(hm.index, fontsize=9)

        for i in range(len(hm.index)):
            for j in range(len(hm.columns)):
                val = hm.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7,
                            color="black" if abs(val) < 0.05 else "white")

        plt.colorbar(im, ax=ax, label="Return", format=plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title("Monthly Returns", fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()

    def plot_annual_returns(self, return_ax=False, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 5)))
        ret = self.returns_over_time
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in annual.values]
        ax.bar(annual.index.year, annual.values, color=colors, alpha=0.85)
        ax.axhline(0, color="grey", linewidth=0.5)
        self._style_ax(ax, "Annual Returns", "Return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        plt.tight_layout()
        if return_ax:
            return ax
        plt.show()
