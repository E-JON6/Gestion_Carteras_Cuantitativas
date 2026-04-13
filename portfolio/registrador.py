"""

Registrador version 0.

Recibe: la decision final, la cartera actual, los precios y las comisiones.

Devuelve: la cartera actualizada y un Excel con las operaciones.



Usa la misma regla y pesos que backtest/engine.py (portfolio.rebalance_policy).

"""



from pathlib import Path



import pandas as pd

from Data.data_loader import get_execution_data_v0, resolve_transaction_cost_for_ticker

from Data.universe import get_defensive_ticker

from portfolio.rebalance_policy import (

    positions_from_weights_full,

    should_apply_rebalance_after_dn,

)


def _ordenes_sheet_name() -> str:
    try:
        import config as _cfg

        return str(getattr(_cfg, "REGISTRADOR_ORDENES_SHEET_NAME", "Operativa"))
    except Exception:
        return "Operativa"


def _resolve_price(ticker, execution_prices, market_prices):

    price = execution_prices.get(ticker)

    if pd.isna(price):

        if ticker not in market_prices.columns:

            raise ValueError(f"No hay precio disponible para el ticker {ticker}.")

        price = market_prices[ticker].dropna().iloc[-1]

    return price


def _xeon_delta_for_notional_target(diff_eur: float, price: float, ct: float) -> tuple[float, float]:
    """
    diff_eur = NAV_real - sum(otras ordenes: Cantidad * Precio Ejecutado).
    Devuelve (delta_titulos_xeon, precio_ejecutado) para cuadrar el nominal.
    """
    if abs(diff_eur) < 1e-9:
        return 0.0, 0.0
    p = float(price)
    c = float(ct)
    if diff_eur > 0.0:
        p_ex = p * (1.0 + c)
        return diff_eur / p_ex, p_ex
    p_ex = p * (1.0 - c)
    return diff_eur / p_ex, p_ex


def run_registrador_v0(

    dn_result,

    market_data,

    current_positions=None,

    total_value=100000,

    output_path="results/Operativa_Grupo4.xlsx",

    only_when_engine_would_trade: bool = True,

):

    """

    Genera Excel de ordenes (deltas) hacia la cartera que aplicaria el engine.



    - Pesos objetivo: siempre dn_result['final_weights_full'] (igual que engine).

    - Si only_when_engine_would_trade=True: no genera ordenes cuando el backtest

      tampoco operaria: con posiciones y dn['rebalance']==False.

      Con cartera vacia, siempre genera (como el engine en la primera fecha).

    - Tras las ordenes del resto de activos, ajusta solo XEON.DE para que

      sum(Cantidad * Precio Ejecutado) == total_value (NAV a dia de ejecucion).

      INITIAL_CAPITAL en config solo aplica al arranque (main fija NAV si no hay posiciones).

    - Ventas: delta = objetivo_titulos - actual; si objetivo es 0 (ETF fuera del

      modelo), Cantidad negativa. Compra: Precio Ejecutado = Precio*(1+ct); venta: *(1-ct).

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

                orders_df.to_excel(writer, sheet_name=_ordenes_sheet_name(), index=False)

                note.to_excel(writer, sheet_name="Nota", index=False)

        except PermissionError:

            output_file = output_file.with_name(
                f"{output_file.stem}_v0{output_file.suffix}"
            )

            with pd.ExcelWriter(output_file) as writer:

                orders_df.to_excel(writer, sheet_name=_ordenes_sheet_name(), index=False)

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

    # Union actual U objetivo: si Merton deja de asignar peso a un ETF (no esta en
    # target_positions), el objetivo implicito es 0 -> delta negativa (venta total).
    non_xeon = (set(target_positions) | set(current_positions)) - {xeon_ticker}

    for ticker in sorted(non_xeon):

        target_quantity = float(target_positions.get(ticker, 0.0))

        current_quantity = float(current_positions.get(ticker, 0.0))

        rebalance_quantity = target_quantity - current_quantity

        if abs(rebalance_quantity) < 1e-12:

            continue

        price = _resolve_price(ticker, execution_prices, market_data["prices"])

        ct = resolve_transaction_cost_for_ticker(ticker, transaction_costs)

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

    notional_target = float(total_value)

    sum_others = sum(float(r["Cantidad"]) * float(r["Precio Ejecutado"]) for r in orders)

    price_x = _resolve_price(xeon_ticker, execution_prices, market_data["prices"])

    ct_x = resolve_transaction_cost_for_ticker(xeon_ticker, transaction_costs)

    if orders:

        diff = notional_target - sum_others

        q_x, p_ex_x = _xeon_delta_for_notional_target(diff, price_x, ct_x)

        if abs(q_x) > 1e-12:

            orders.append(

                {

                    "ID": xeon_ticker,

                    "Cantidad": q_x,

                    "Precio": price_x,

                    "CT": ct_x,

                    "Precio Ejecutado": p_ex_x,

                }

            )

    else:

        tq_x = float(target_positions.get(xeon_ticker, 0.0))

        cq_x = float(current_positions.get(xeon_ticker, 0.0))

        rebalance_x = tq_x - cq_x

        if abs(rebalance_x) > 1e-12:

            if rebalance_x > 0:

                p_ex_x = price_x * (1 + ct_x)

            else:

                p_ex_x = price_x * (1 - ct_x)

            orders.append(

                {

                    "ID": xeon_ticker,

                    "Cantidad": rebalance_x,

                    "Precio": price_x,

                    "CT": ct_x,

                    "Precio Ejecutado": p_ex_x,

                }

            )

    orders_df = pd.DataFrame(orders, columns=["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"])



    updated_positions = dict(current_positions)

    for row in orders:

        ticker = row["ID"]

        updated_positions[ticker] = updated_positions.get(ticker, 0.0) + row["Cantidad"]



    output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    sum_sheet = (

        float(

            (

                orders_df["Cantidad"].astype(float)

                * orders_df["Precio Ejecutado"].astype(float)

            ).sum()

        )

        if not orders_df.empty

        else 0.0

    )

    meta = pd.DataFrame(

        {

            "Campo": [

                "Politica",

                "Davis-Norman rebalance",

                "Motivo",

                "Columna CT",

                "Cuadre nominal",

                "NAV (total_value, patrimonio hoy)",

                "Suma Cantidad*Precio Ejecutado",

            ],

            "Valor": [

                "final_weights_full + misma regla que engine",

                str(dn_result.get("rebalance")),

                str(dn_result.get("reason", "")),

                "Fraccion por lado: fila en Data/comisiones_etfs.xlsx (ticker); si no, TX_COST_PER_SIDE config",

                "Con ordenes en otros activos: XEON cuadra sum(Cant*PxEj) al NAV de esta ejecucion",

                str(notional_target),

                str(round(sum_sheet, 8)),

            ],

        }

    )

    try:

        with pd.ExcelWriter(output_file) as writer:

            orders_df.to_excel(writer, sheet_name=_ordenes_sheet_name(), index=False)

            meta.to_excel(writer, sheet_name="Nota", index=False)

    except PermissionError:

        output_file = output_file.with_name(f"{output_file.stem}_v0{output_file.suffix}")

        with pd.ExcelWriter(output_file) as writer:

            orders_df.to_excel(writer, sheet_name=_ordenes_sheet_name(), index=False)

            meta.to_excel(writer, sheet_name="Nota", index=False)



    return {

        "trade_date": execution_data["date"],

        "used_next_day": execution_data["used_next_day"],

        "orders": orders_df,

        "output_path": str(output_file),

        "updated_positions": updated_positions,

        "skipped_due_to_dn": False,

    }


