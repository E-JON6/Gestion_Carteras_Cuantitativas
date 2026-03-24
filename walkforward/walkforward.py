"""
Walkforward version 0.
Recibe: un rango temporal y ventanas simples.
Devuelve: un resumen por ventana guardado en results.
"""

from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest_v0


def run_walkforward_v0(
    start_date="2024-01-01",
    end_date="2024-12-31",
    metric_name="omega",
    train_months=3,
    test_months=1,
    output_path="results/walkforward_resumen.csv",
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []

    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        file_name = f"walkforward_{test_end.strftime('%Y%m%d')}.csv"
        backtest_result = run_backtest_v0(
            start_date=test_start.strftime("%Y-%m-%d"),
            end_date=test_end.strftime("%Y-%m-%d"),
            metric_name=metric_name,
            output_path=f"results/{file_name}",
        )

        last_row = backtest_result["backtest"].iloc[-1]
        rows.append(
            {
                "Train Start": train_start.strftime("%Y-%m-%d"),
                "Train End": train_end.strftime("%Y-%m-%d"),
                "Test Start": test_start.strftime("%Y-%m-%d"),
                "Test End": test_end.strftime("%Y-%m-%d"),
                "Metric": metric_name,
                "Selected ETFs": last_row["Selected ETFs"],
                "Weight XEON": last_row["Weight XEON"],
                "Backtest File": backtest_result["output_path"],
            }
        )

        train_start = train_start + pd.DateOffset(months=test_months)

    walkforward_df = pd.DataFrame(rows)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    walkforward_df.to_csv(output_file, index=False)
    walkforward_excel_file = output_file.with_suffix(".xlsx")
    walkforward_df.to_excel(walkforward_excel_file, index=False)

    return {
        "walkforward": walkforward_df,
        "output_path": str(output_file),
        "excel_output_path": str(walkforward_excel_file),
    }
