"""
Engine de backtest version 0.
Recibe: un rango temporal y una metrica.
Devuelve: un resumen simple del backtest.
"""

from pathlib import Path

import pandas as pd

from Data.universe import get_defensive_ticker
from main import run_v0



def _last_price_on_or_before(prices, review_date, ticker):
    series = prices.loc[:review_date, ticker].dropna()
    if series.empty:
        raise ValueError(
            f"No hay precio historico disponible para {ticker} en o antes de {review_date}."
        )
    return series.iloc[-1]



def run_backtest_v0(
    start_date="2024-01-01",
    end_date="2024-06-30",
    metric_name="omega",
    freq="ME",
    output_path="results/backtest_resumen.csv",
):
    review_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    rows = []
    wealth_rows = []
    strategy_value = 100.0
    sp500_value = 100.0
    previous_review_date = None
    xeon_ticker = get_defensive_ticker()

    for review_date in review_dates:
        file_name = f"operaciones_{review_date.strftime('%Y%m%d')}.xlsx"
        result = run_v0(
            start_date=start_date,
            end_date=review_date.strftime("%Y-%m-%d"),
            metric_name=metric_name,
            output_path=f"results/{file_name}",
        )
        prices = result["market_data"]["prices"]
        selected_etfs = [ticker for ticker in result["selected_etfs"] if ticker != xeon_ticker]
        final_weights = result["dn_result"]["final_weights"]
        weight_xeon = result["dn_result"]["weight_xeon"]

        if previous_review_date is None:
            wealth_rows.append(
                {
                    "Date": review_date.strftime("%Y-%m-%d"),
                    "Strategy": strategy_value,
                    "SP500": sp500_value,
                }
            )
            previous_review_date = review_date
        else:
            strategy_return = 0.0

            for ticker in selected_etfs:
                start_price = _last_price_on_or_before(prices, previous_review_date, ticker)
                end_price = _last_price_on_or_before(prices, review_date, ticker)
                ticker_return = end_price / start_price - 1
                strategy_return += final_weights[ticker] * ticker_return

            xeon_start = _last_price_on_or_before(prices, previous_review_date, xeon_ticker)
            xeon_end = _last_price_on_or_before(prices, review_date, xeon_ticker)
            xeon_return = xeon_end / xeon_start - 1
            strategy_return += weight_xeon * xeon_return

            spy_start = _last_price_on_or_before(prices, previous_review_date, "SPY")
            spy_end = _last_price_on_or_before(prices, review_date, "SPY")
            sp500_return = spy_end / spy_start - 1

            strategy_value = strategy_value * (1 + strategy_return)
            sp500_value = sp500_value * (1 + sp500_return)

            wealth_rows.append(
                {
                    "Date": review_date.strftime("%Y-%m-%d"),
                    "Strategy": strategy_value,
                    "SP500": sp500_value,
                }
            )
            previous_review_date = review_date

        rows.append(
            {
                "Date": review_date.strftime("%Y-%m-%d"),
                "Metric": result["metric_name"],
                "Selected ETFs": ", ".join(selected_etfs),
                "Rebalance": result["dn_result"]["rebalance"],
                "Weight XEON": result["dn_result"]["weight_xeon"],
                "Orders File": result["registrador_result"]["output_path"],
            }
        )

    backtest_df = pd.DataFrame(rows)
    wealth_df = pd.DataFrame(wealth_rows)
    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Strategy Final Value",
                "Value": wealth_df["Strategy"].iloc[-1] if not wealth_df.empty else None,
            },
            {
                "Metric": "SP500 Final Value",
                "Value": wealth_df["SP500"].iloc[-1] if not wealth_df.empty else None,
            },
            {
                "Metric": "Strategy Total Return",
                "Value": (wealth_df["Strategy"].iloc[-1] / wealth_df["Strategy"].iloc[0] - 1)
                if len(wealth_df) > 1
                else 0.0,
            },
            {
                "Metric": "SP500 Total Return",
                "Value": (wealth_df["SP500"].iloc[-1] / wealth_df["SP500"].iloc[0] - 1)
                if len(wealth_df) > 1
                else 0.0,
            },
            {
                "Metric": "Number of Rebalances",
                "Value": int(backtest_df["Rebalance"].sum()) if not backtest_df.empty else 0,
            },
            {
                "Metric": "Average Weight XEON",
                "Value": backtest_df["Weight XEON"].mean() if not backtest_df.empty else 0.0,
            },
        ]
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output_file, index=False)
    if output_file.stem == "backtest_resumen":
        wealth_file = output_file.with_name("wealth_history.csv")
        metrics_file = output_file.with_name("Metrics.xlsx")
    else:
        wealth_file = output_file.with_name(f"{output_file.stem}_wealth_history.csv")
        metrics_file = output_file.with_name(f"{output_file.stem}_Metrics.xlsx")

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
