"""Entrada principal de compatibilidad para el backtest BL-Omega."""

from backtest.engine import run_backtest_v0


if __name__ == "__main__":
    result = run_backtest_v0(
        start_date="2024-01-01",
        end_date="2024-06-30",
        metric_name="omega",
    )
    print("Backtest guardado en:", result["output_path"])
    print(result["metrics"])
