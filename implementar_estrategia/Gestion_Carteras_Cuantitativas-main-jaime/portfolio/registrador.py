"""

Registrador version 0.

Recibe: la decision final, la cartera actual, los precios y las comisiones.

Devuelve: la cartera actualizada y un Excel con las operaciones.



Usa la misma regla y pesos que backtest/engine.py (portfolio.rebalance_policy).

"""



from pathlib import Path



import pandas as pd



from Data.data_loader import get_execution_data_v0

from Data.universe import get_defensive_ticker

from portfolio.rebalance_policy import (

    positions_from_weights_full,

    should_apply_rebalance_after_dn,

)





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

    only_when_engine_would_trade: bool = True,

):

    """

    Genera Excel de ordenes (deltas) hacia la cartera que aplicaria el engine.



    - Pesos objetivo: siempre dn_result['final_weights_full'] (igual que engine).

    - Si only_when_engine_would_trade=True: no genera ordenes cuando el backtest

      tampoco operaria: con posiciones y dn['rebalance']==False.

      Con cartera vacia, siempre genera (como el engine en la primera fecha).

    """

    current_positions = current_positions or {}

    execution_data = get_execution_data_v0(market_data["tickers"], market_data["prices"])

    execution_prices = execution_data["prices"]

    transaction_costs = market_data["transaction_costs"]

    xeon_ticker = get_defensive_ticker()



    skip_orders = only_when_engine_would_trade and not should_apply_rebalance_after_dn(

        dn_result, current_positions if current_positions else None

    )



    if skip_orders:

        orders_df = pd.DataFrame(

            columns=["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"]

        )

        note = pd.DataFrame(

            {

                "Concepto": [

                    "Politica (igual que engine)",

                    "Davis-Norman",

                    "Motivo",

                ],

                "Detalle": [

                    "No operar si hay posiciones y rebalance=False (misma regla que backtest).",

                    f"rebalance = {dn_result.get('rebalance')}",

                    str(dn_result.get("reason", "")),

                ],

            }

        )

        output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:

            with pd.ExcelWriter(output_file) as writer:

                orders_df.to_excel(writer, sheet_name="Ordenes", index=False)

                note.to_excel(writer, sheet_name="Nota", index=False)

        except PermissionError:

            output_file = output_file.with_name("operaciones_rebalanceo_v0.xlsx")

            with pd.ExcelWriter(output_file) as writer:

                orders_df.to_excel(writer, sheet_name="Ordenes", index=False)

                note.to_excel(writer, sheet_name="Nota", index=False)



        return {

            "trade_date": execution_data["date"],

            "used_next_day": execution_data["used_next_day"],

            "orders": orders_df,

            "output_path": str(output_file),

            "updated_positions": dict(current_positions),

            "skipped_due_to_dn": True,

        }



    fw = dict(dn_result["final_weights_full"])

    prices_series = pd.Series(

        {t: _resolve_price(t, execution_prices, market_data["prices"]) for t in fw}

    )

    target_positions = positions_from_weights_full(

        fw, total_value, prices_series, xeon_ticker

    )



    orders = []

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

    meta = pd.DataFrame(

        {

            "Campo": ["Politica", "Davis-Norman rebalance", "Motivo"],

            "Valor": [

                "final_weights_full + misma regla que engine",

                str(dn_result.get("rebalance")),

                str(dn_result.get("reason", "")),

            ],

        }

    )

    try:

        with pd.ExcelWriter(output_file) as writer:

            orders_df.to_excel(writer, sheet_name="Ordenes", index=False)

            meta.to_excel(writer, sheet_name="Nota", index=False)

    except PermissionError:

        output_file = output_file.with_name("operaciones_rebalanceo_v0.xlsx")

        with pd.ExcelWriter(output_file) as writer:

            orders_df.to_excel(writer, sheet_name="Ordenes", index=False)

            meta.to_excel(writer, sheet_name="Nota", index=False)



    return {

        "trade_date": execution_data["date"],

        "used_next_day": execution_data["used_next_day"],

        "orders": orders_df,

        "output_path": str(output_file),

        "updated_positions": updated_positions,

        "skipped_due_to_dn": False,

    }


