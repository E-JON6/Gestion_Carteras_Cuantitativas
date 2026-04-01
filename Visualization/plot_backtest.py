"""Plot backtest wrapper."""

from __future__ import annotations

from Visualization.plots import plot_backtest_v0, plot_drawdown_v0, plot_strategy_vs_sp500_v0


if __name__ == "__main__":
    print("Grafico estrategia:", plot_strategy_vs_sp500_v0())
    print("Grafico backtest:", plot_backtest_v0())
    print("Grafico drawdown:", plot_drawdown_v0())
