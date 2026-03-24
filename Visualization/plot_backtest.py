"""
Plot backtest version 0.
Recibe: los CSVs del backtest.
Devuelve: graficos guardados en results.
"""

from Visualization.plots import plot_backtest_v0, plot_strategy_vs_sp500_v0


if __name__ == "__main__":
    strategy_plot = plot_strategy_vs_sp500_v0()
    backtest_plot = plot_backtest_v0()
    print("Grafico estrategia vs SP500:", strategy_plot)
    print("Grafico backtest:", backtest_plot)
