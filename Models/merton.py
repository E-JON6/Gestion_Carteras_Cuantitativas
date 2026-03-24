"""
Merton version 0.
Recibe: el resultado de Black-Litterman.
Devuelve: pesos iguales para los ETFs seleccionados.
"""


def run_merton_v0(bl_result):
    selected_etfs = bl_result["top_tickers"]

    if len(selected_etfs) == 0:
        return {
            "weights": {},
            "selected_etfs": [],
            "weight_sum": 0.0,
            "weight_xeon": 1.0,
        }

    equal_weight = 1 / len(selected_etfs)
    weights = {ticker: equal_weight for ticker in selected_etfs}

    return {
        "weights": weights,
        "selected_etfs": selected_etfs,
        "weight_sum": sum(weights.values()),
        "weight_xeon": 1 - sum(weights.values()),
    }
