"""Wrapper de compatibilidad sobre `main_black_litterman.run_backtest`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from main_black_litterman import run_backtest



def _build_backtest_rows(history: pd.DataFrame, xeon_ticker: str) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Date", "Selected ETFs", "Rebalance", "Weight XEON", "Trade Cost"])

    rows = []
    weight_xeon_col = f"weight_{xeon_ticker}"
    weight_cols = [column for column in history.columns if column.startswith("weight_") and column != weight_xeon_col]
    for date, row in history.iterrows():
        selected = []
        for column in weight_cols:
            weight = float(row.get(column, 0.0))
            if weight > 1e-6:
                selected.append(column.removeprefix("weight_"))
        rows.append(
            {
                "Date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "Selected ETFs": ", ".join(selected),
                "Rebalance": bool(row.get("rebalance", False)),
                "Weight XEON": float(row.get(weight_xeon_col, 0.0)),
                "Trade Cost": float(row.get("trade_cost", 0.0)),
            }
        )
    return pd.DataFrame(rows)



def _build_wealth_rows(history: pd.DataFrame, benchmark_wealth) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Date", "Strategy", "SP500"])

    wealth = history[["wealth"]].rename(columns={"wealth": "Strategy"}).copy()
    if benchmark_wealth is None:
        wealth["SP500"] = pd.NA
    else:
        wealth["SP500"] = benchmark_wealth.reindex(wealth.index).ffill()
    wealth = wealth.reset_index().rename(columns={wealth.index.name or "index": "Date"})
    wealth["Date"] = pd.to_datetime(wealth["Date"]).dt.strftime("%Y-%m-%d")
    return wealth



def run_backtest_v0(
    start_date="2024-01-01",
    end_date="2024-06-30",
    metric_name="omega",
    freq="ME",
    output_path="results/backtest_resumen.csv",
):
    del metric_name, freq
    result = run_backtest(start=start_date, end=end_date, save_results=False)
    history = result["history"]
    xeon_ticker = result["xeon_ticker"]

    backtest_df = _build_backtest_rows(history, xeon_ticker)
    wealth_df = _build_wealth_rows(history, result.get("benchmark_wealth"))
    metrics_df = pd.DataFrame(
        [
            {"Metric": "Sharpe Ratio", "Value": result["sharpe"]},
            {"Metric": "CAGR", "Value": result["cagr"]},
            {"Metric": "Max Drawdown", "Value": result["max_dd"]},
            {"Metric": "Average RF", "Value": result["avg_rf"]},
            {"Metric": "Number of Rebalances", "Value": int(backtest_df["Rebalance"].sum()) if not backtest_df.empty else 0},
            {"Metric": "Average Weight XEON", "Value": backtest_df["Weight XEON"].mean() if not backtest_df.empty else 0.0},
        ]
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output_file, index=False)
    wealth_file = output_file.with_name("wealth_history.csv") if output_file.stem == "backtest_resumen" else output_file.with_name(f"{output_file.stem}_wealth_history.csv")
    metrics_file = output_file.with_name("Metrics.xlsx") if output_file.stem == "backtest_resumen" else output_file.with_name(f"{output_file.stem}_Metrics.xlsx")
    wealth_df.to_csv(wealth_file, index=False)
    backtest_excel_file = output_file.with_suffix(".xlsx")
    backtest_df.to_excel(backtest_excel_file, index=False)
    with pd.ExcelWriter(metrics_file) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        backtest_df.to_excel(writer, sheet_name="Backtest", index=False)
        wealth_df.to_excel(writer, sheet_name="Wealth", index=False)

    return {
        "backtest": backtest_df,
        "wealth_history": wealth_df,
        "metrics": metrics_df,
        "output_path": str(output_file),
        "wealth_output_path": str(wealth_file),
        "excel_output_path": str(backtest_excel_file),
        "metrics_output_path": str(metrics_file),
    }
