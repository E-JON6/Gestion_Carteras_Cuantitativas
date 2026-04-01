"""
Registrador de órdenes con sizing autofinanciado y costes explícitos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from Data.data_loader import get_execution_data_v0
from Data.universe import get_defensive_ticker

EPS = 1e-12


def _resolve_price(ticker, execution_prices, market_prices):
    price = execution_prices.get(ticker)
    if pd.isna(price):
        if ticker not in market_prices.columns:
            raise ValueError(f"No hay precio disponible para el ticker {ticker}.")
        series = market_prices[ticker].dropna()
        if series.empty:
            raise ValueError(f"No hay histórico utilizable para el ticker {ticker}.")
        price = series.iloc[-1]
    return float(price)


def _normalize_target_weights(dn_result, tickers, xeon_ticker):
    if "final_weights_full" in dn_result:
        raw = dn_result["final_weights_full"]
        if isinstance(raw, Mapping):
            weights = {ticker: float(raw.get(ticker, 0.0)) for ticker in tickers}
        else:
            weights = {
                ticker: float(weight)
                for ticker, weight in zip(tickers, raw, strict=False)
            }
    elif "target_weights_full" in dn_result:
        raw = dn_result["target_weights_full"]
        if isinstance(raw, Mapping):
            weights = {ticker: float(raw.get(ticker, 0.0)) for ticker in tickers}
        else:
            weights = {
                ticker: float(weight)
                for ticker, weight in zip(tickers, raw, strict=False)
            }
    elif "target_weights" in dn_result and not isinstance(dn_result["target_weights"], Mapping):
        weights = {
            ticker: float(weight)
            for ticker, weight in zip(tickers, dn_result["target_weights"], strict=False)
        }
    else:
        risk_weights = {ticker: float(weight) for ticker, weight in dn_result.get("final_weights", {}).items()}
        weight_xeon = float(dn_result.get("weight_xeon", max(0.0, 1.0 - sum(risk_weights.values()))))
        weights = {ticker: risk_weights.get(ticker, 0.0) for ticker in tickers}
        weights[xeon_ticker] = weight_xeon

    for ticker in tickers:
        weights.setdefault(ticker, 0.0)

    weights = {ticker: max(float(weight), 0.0) for ticker, weight in weights.items()}
    total = float(sum(weights.values()))
    if total <= EPS:
        return {ticker: 1.0 if ticker == xeon_ticker else 0.0 for ticker in tickers}
    return {ticker: float(weight) / total for ticker, weight in weights.items()}


def _solve_post_trade_wealth(
    budget_before: float,
    current_clean_values: dict[str, float],
    target_weights: Mapping[str, float],
    transaction_costs: Mapping[str, float],
    tickers: list[str],
) -> float:
    wealth_after = float(budget_before)
    for _ in range(200):
        total_cost = 0.0
        for ticker in tickers:
            target_value = float(target_weights.get(ticker, 0.0)) * wealth_after
            current_value = float(current_clean_values.get(ticker, 0.0))
            lam = float(transaction_costs.get(ticker, 0.0))
            total_cost += lam * abs(target_value - current_value)
        new_wealth_after = float(budget_before) - total_cost
        if new_wealth_after < -1e-8:
            raise ValueError("Los costes de transacción superan el patrimonio disponible.")
        if abs(new_wealth_after - wealth_after) <= 1e-10 * max(float(budget_before), 1.0):
            wealth_after = max(new_wealth_after, 0.0)
            break
        wealth_after = max(new_wealth_after, 0.0)
    return float(wealth_after)


def _build_target_positions(
    tickers,
    xeon_ticker,
    current_positions,
    execution_prices,
    market_prices,
    transaction_costs,
    target_weights,
    total_value,
):
    clean_prices = {
        ticker: _resolve_price(ticker, execution_prices, market_prices)
        for ticker in tickers
    }
    current_positions = {ticker: float(current_positions.get(ticker, 0.0)) for ticker in tickers}
    current_clean_values = {
        ticker: current_positions[ticker] * clean_prices[ticker]
        for ticker in tickers
    }

    clean_holdings_value = float(sum(current_clean_values.values()))
    if clean_holdings_value > EPS:
        budget_before = clean_holdings_value
    else:
        budget_before = float(total_value)
    initial_cash = max(float(budget_before) - clean_holdings_value, 0.0)

    wealth_after = _solve_post_trade_wealth(
        budget_before=budget_before,
        current_clean_values=current_clean_values,
        target_weights=target_weights,
        transaction_costs=transaction_costs,
        tickers=list(tickers),
    )

    target_positions = {ticker: current_positions.get(ticker, 0.0) for ticker in tickers}
    cash_balance = float(initial_cash)
    risk_tickers = [ticker for ticker in tickers if ticker != xeon_ticker]

    for ticker in risk_tickers:
        price = clean_prices[ticker]
        target_clean_value = float(target_weights.get(ticker, 0.0)) * wealth_after
        target_quantity = np.floor(target_clean_value / price) if price > 0 else 0.0
        target_quantity = float(max(target_quantity, 0.0))
        current_quantity = float(current_positions.get(ticker, 0.0))
        delta = target_quantity - current_quantity
        lam = float(transaction_costs.get(ticker, 0.0))
        if delta > 0:
            cash_balance -= delta * price * (1.0 + lam)
        elif delta < 0:
            cash_balance += (-delta) * price * (1.0 - lam)
        target_positions[ticker] = target_quantity

    xeon_price = clean_prices[xeon_ticker]
    xeon_lam = float(transaction_costs.get(xeon_ticker, 0.0))
    current_xeon = float(current_positions.get(xeon_ticker, 0.0))

    if cash_balance >= -EPS:
        xeon_target = current_xeon + cash_balance / (xeon_price * (1.0 + xeon_lam)) if xeon_price > 0 else current_xeon
        cash_balance = 0.0
    else:
        affordable_sale_qty = current_xeon
        max_sellable_cash = affordable_sale_qty * xeon_price * (1.0 - xeon_lam)
        if -cash_balance > max_sellable_cash + 1e-8:
            buy_candidates = [ticker for ticker in risk_tickers if target_positions[ticker] > current_positions.get(ticker, 0.0) + EPS]
            buy_candidates.sort(key=lambda ticker: clean_prices[ticker] * (1.0 + float(transaction_costs.get(ticker, 0.0))), reverse=True)
            for ticker in buy_candidates:
                if cash_balance >= -max_sellable_cash - 1e-8:
                    break
                target_positions[ticker] -= 1.0
                cash_balance += clean_prices[ticker] * (1.0 + float(transaction_costs.get(ticker, 0.0)))
            if -cash_balance > max_sellable_cash + 1e-8:
                raise ValueError("No se puede financiar la operativa ni vendiendo toda la posición en XEON.DE.")

        xeon_target = current_xeon + cash_balance / (xeon_price * (1.0 - xeon_lam)) if xeon_price > 0 else current_xeon
        xeon_target = max(xeon_target, 0.0)
        cash_balance = 0.0

    target_positions[xeon_ticker] = float(xeon_target)
    return target_positions, clean_prices, float(budget_before), float(initial_cash)


def run_registrador_v0(
    dn_result,
    market_data,
    current_positions=None,
    total_value=100000,
    output_path="results/operaciones_rebalanceo.xlsx",
):
    current_positions = current_positions or {}
    xeon_ticker = get_defensive_ticker()
    tickers = list(market_data["tickers"])
    target_weights = _normalize_target_weights(dn_result, tickers, xeon_ticker)

    execution_data = get_execution_data_v0(tickers, market_data["prices"])
    execution_prices = execution_data["prices"]
    transaction_costs = market_data["transaction_costs"]

    target_positions, clean_prices, budget_before, initial_cash = _build_target_positions(
        tickers=tickers,
        xeon_ticker=xeon_ticker,
        current_positions=current_positions,
        execution_prices=execution_prices,
        market_prices=market_data["prices"],
        transaction_costs=transaction_costs,
        target_weights=target_weights,
        total_value=float(total_value),
    )

    orders = []
    updated_positions = dict(current_positions)
    buy_executed_notional = 0.0
    sell_executed_notional = 0.0
    total_costs = 0.0

    all_tickers = list(dict.fromkeys(list(target_positions.keys()) + list(current_positions.keys())))
    for ticker in all_tickers:
        target_quantity = float(target_positions.get(ticker, 0.0))
        current_quantity = float(current_positions.get(ticker, 0.0))
        rebalance_quantity = float(target_quantity - current_quantity)
        if abs(rebalance_quantity) < EPS:
            continue

        price = clean_prices.get(ticker)
        if price is None:
            price = _resolve_price(ticker, execution_prices, market_data["prices"])
        ct = float(transaction_costs.get(ticker, 0.0))
        if rebalance_quantity > 0:
            executed_price = price * (1.0 + ct)
            executed_notional = rebalance_quantity * executed_price
            buy_executed_notional += executed_notional
            trade_cost = rebalance_quantity * price * ct
        else:
            executed_price = price * (1.0 - ct)
            executed_notional = abs(rebalance_quantity) * executed_price
            sell_executed_notional += executed_notional
            trade_cost = abs(rebalance_quantity) * price * ct

        total_costs += trade_cost
        orders.append(
            {
                "ID": ticker,
                "Cantidad": rebalance_quantity,
                "Precio": price,
                "CT": ct,
                "Precio Ejecutado": executed_price,
                "Notional Ejecutado": executed_notional,
                "Coste Total": trade_cost,
            }
        )
        updated_positions[ticker] = current_quantity + rebalance_quantity

    cash_slack = float(initial_cash + sell_executed_notional - buy_executed_notional)
    if cash_slack < -1e-6:
        raise ValueError(
            f"La operativa no es autofinanciada: falta caja por {-cash_slack:.2f} EUR."
        )

    orders_df = pd.DataFrame(
        orders,
        columns=[
            "ID",
            "Cantidad",
            "Precio",
            "CT",
            "Precio Ejecutado",
            "Notional Ejecutado",
            "Coste Total",
        ],
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        orders_df.to_excel(output_file, index=False)
    except PermissionError:
        output_file = output_file.with_name(f"{output_file.stem}_v0.xlsx")
        orders_df.to_excel(output_file, index=False)

    return {
        "trade_date": execution_data["date"],
        "used_next_day": execution_data["used_next_day"],
        "orders": orders_df,
        "output_path": str(output_file),
        "updated_positions": updated_positions,
        "buy_executed_notional": float(buy_executed_notional),
        "sell_executed_notional": float(sell_executed_notional),
        "total_costs": float(total_costs),
        "budget": float(budget_before),
        "cash_slack": float(cash_slack),
    }
