"""Wrapper de compatibilidad para generar una operativa puntual con el stack BL-Omega."""

from __future__ import annotations

import inspect

from config import (
    ETF_UNIVERSE,
    XEON_TICKER,
    BL_OMEGA_WINDOW_FAST,
    BL_RECALIB_FREQ,
)
from Data.data_loader import download_market_data
from portfolio.registrador import run_registrador_v0
from strategies.bl_omega_strategy import BLOmegaStrategy

DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_GAMMA = -2.0



def _build_strategy(tickers):
    signature = inspect.signature(BLOmegaStrategy)
    kwargs = {
        "tickers": tickers,
        "xeon_ticker": XEON_TICKER,
    }
    optional = {
        "categoria_por_ticker": dict(ETF_UNIVERSE),
        "gamma": DEFAULT_GAMMA,
        "recalib_freq": BL_RECALIB_FREQ,
        "omega_window_fast": BL_OMEGA_WINDOW_FAST,
    }
    for key, value in optional.items():
        if key in signature.parameters:
            kwargs[key] = value
    return BLOmegaStrategy(**kwargs)



def run_v0(
    start_date="2024-01-01",
    end_date="2024-03-01",
    metric_name="omega",
    output_path="results/operaciones_rebalanceo.xlsx",
):
    if metric_name != "omega":
        raise ValueError("La versión BL-Omega solo soporta la métrica omega.")

    market_data = download_market_data(start_date=start_date, end_date=end_date)
    strategy = _build_strategy(market_data["tickers"])
    target_weights = strategy.get_initial_weights(
        market_data["returns"],
        DEFAULT_RISK_FREE_RATE,
        market_data["transaction_costs"],
    )
    target_weights_full = {
        ticker: float(weight)
        for ticker, weight in zip(market_data["tickers"], target_weights, strict=False)
    }
    dn_result = {
        "rebalance": True,
        "reason": "initial_allocation",
        "target_weights": target_weights,
        "target_weights_full": target_weights_full,
        "final_weights_full": target_weights_full,
        "final_weights": {ticker: weight for ticker, weight in target_weights_full.items() if ticker != XEON_TICKER and weight > 0},
        "weight_xeon": target_weights_full.get(XEON_TICKER, 0.0),
    }
    registrador_result = run_registrador_v0(
        dn_result,
        market_data,
        current_positions={},
        output_path=output_path,
    )

    return {
        "metric_name": metric_name,
        "market_data": market_data,
        "scores": getattr(strategy, "_last_bl_result", {}).get("omega_scores") if getattr(strategy, "_last_bl_result", None) else None,
        "selected_etfs": getattr(strategy, "_last_merton", {}).get("selected_tickers") if getattr(strategy, "_last_merton", None) else [],
        "bl_result": getattr(strategy, "_last_bl_result", None),
        "merton_result": getattr(strategy, "_last_merton", None),
        "dn_result": dn_result,
        "registrador_result": registrador_result,
    }


if __name__ == "__main__":
    result = run_v0(metric_name="omega")
    print("Métrica usada:", result["metric_name"])
    print("Órdenes guardadas en:", result["registrador_result"]["output_path"])
