"""
Entry point para ejecutar el backtest completo con graficos.

Uso:
    python main_backtest.py
"""

from backtest.engine import run_backtest
from Visualization.plots import generate_all_plots


if __name__ == "__main__":
    result = run_backtest(
        start_date="2020-01-01",
        end_date="2026-04-01",
    )
    generate_all_plots(output_dir=result["output_dir"])
    print("\nBacktest completado.")
    print(f"Resultados en: {result['output_dir']}")
