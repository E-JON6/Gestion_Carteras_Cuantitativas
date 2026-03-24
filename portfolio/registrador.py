"""
Registrador version 0.
Recibe: la decision final, la cartera actual, los precios y las comisiones.
Devuelve: la cartera actualizada y un Excel con las operaciones.
"""

from pathlib import Path

import pandas as pd

from Data.data_loader import get_execution_data_v0
from Data.universe import get_defensive_ticker


def _resolve_price(ticker, execution_prices, market_prices):
    price = execution_prices.get(ticker)
    if pd.isna(price):
        if ticker not in market_prices.columns:
            raise ValueError(f"No hay precio disponible para el ticker {ticker}.")
        price = market_prices[ticker].dropna().iloc[-1]
    return price



def run_registrador_v0(
    dn_result,
    market_data,
    current_positions=None,
    total_value=100000,
    output_path="results/operaciones_rebalanceo.xlsx",
):
    current_positions = current_positions or {}
    target_weights = dn_result["final_weights"]
    execution_data = get_execution_data_v0(market_data["tickers"], market_data["prices"])
    execution_prices = execution_data["prices"]
    transaction_costs = market_data["transaction_costs"]
    xeon_ticker = get_defensive_ticker()

    orders = []
    target_positions = {}
    used_value = 0.0

    for ticker, target_weight in target_weights.items():
        if ticker == xeon_ticker:
            continue

        price = _resolve_price(ticker, execution_prices, market_data["prices"])
        target_quantity = int(total_value * target_weight / price)
        target_positions[ticker] = target_quantity
        used_value += target_quantity * price

    xeon_price = _resolve_price(xeon_ticker, execution_prices, market_data["prices"])
    xeon_target_value = max(total_value - used_value, 0.0)
    target_positions[xeon_ticker] = xeon_target_value / xeon_price

    for ticker, target_quantity in target_positions.items():
        price = _resolve_price(ticker, execution_prices, market_data["prices"])
        current_quantity = current_positions.get(ticker, 0.0)
        rebalance_quantity = target_quantity - current_quantity

        if rebalance_quantity == 0:
            continue

        ct = transaction_costs.get(ticker, 0.0)

        if rebalance_quantity > 0:
            executed_price = price * (1 + ct)
        else:
            executed_price = price * (1 - ct)

        orders.append(
            {
                "ID": ticker,
                "Cantidad": rebalance_quantity,
                "Precio": price,
                "CT": ct,
                "Precio Ejecutado": executed_price,
            }
        )

    orders_df = pd.DataFrame(orders, columns=["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"])

    updated_positions = dict(current_positions)
    for row in orders:
        ticker = row["ID"]
        updated_positions[ticker] = updated_positions.get(ticker, 0.0) + row["Cantidad"]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        orders_df.to_excel(output_file, index=False)
    except PermissionError:
        output_file = output_file.with_name("operaciones_rebalanceo_v0.xlsx")
        orders_df.to_excel(output_file, index=False)

    return {
        "trade_date": execution_data["date"],
        "used_next_day": execution_data["used_next_day"],
        "orders": orders_df,
        "output_path": str(output_file),
        "updated_positions": updated_positions,
    }
