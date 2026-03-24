"""
Davis-Norman fake version 0.
Recibe: pesos actuales y pesos objetivo.
Devuelve: si se rebalancea o no.
"""


def run_davis_norman_fake_v0(current_weights, target_weights, band=0.05, risk_budget=0.8):
    scaled_target_weights = {ticker: weight * risk_budget for ticker, weight in target_weights.items()}
    rebalance = False

    for ticker, target_weight in scaled_target_weights.items():
        current_weight = current_weights.get(ticker, 0.0)
        if abs(target_weight - current_weight) > band:
            rebalance = True
            break

    if rebalance:
        final_weights = scaled_target_weights
        reason = "rebalance"
    else:
        final_weights = current_weights
        reason = "no rebalance"

    return {
        "rebalance": rebalance,
        "target_weights": scaled_target_weights,
        "final_weights": final_weights,
        "weight_xeon": 1 - sum(final_weights.values()),
        "reason": reason,
    }
