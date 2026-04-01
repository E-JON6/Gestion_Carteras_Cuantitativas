"""
Wrapper de compatibilidad sobre la implementación de bandas Davis-Norman.
"""

from __future__ import annotations

from Models.davis_norman import dn_bands_asymptotic


def _clean_risk_weights(weights, xeon_ticker):
    clean_weights = {}
    for ticker, weight in (weights or {}).items():
        if ticker == xeon_ticker:
            continue
        clean_weights[ticker] = max(float(weight), 0.0)
    total_risk = sum(clean_weights.values())
    if total_risk > 1.0 and total_risk > 0:
        clean_weights = {ticker: weight / total_risk for ticker, weight in clean_weights.items()}
    return clean_weights


def _weight_xeon_from_risk(risk_weights):
    return max(0.0, 1.0 - sum(risk_weights.values()))


def run_davis_norman_fake_v0(
    current_weights,
    target_weights,
    band=0.05,
    min_band=0.02,
    xeon_ticker="XEON.DE",
):
    target_risk_weights = _clean_risk_weights(target_weights, xeon_ticker)
    current_risk_weights = _clean_risk_weights(current_weights, xeon_ticker)

    target_weight_xeon = _weight_xeon_from_risk(target_risk_weights)
    current_weight_xeon = current_weights.get(xeon_ticker, _weight_xeon_from_risk(current_risk_weights))

    lower_bands = {}
    upper_bands = {}
    rebalance = False
    reason = "no rebalance"

    for ticker in sorted(set(target_risk_weights) | set(current_risk_weights)):
        target_weight = target_risk_weights.get(ticker, 0.0)
        current_weight = current_risk_weights.get(ticker, 0.0)
        lower, upper = dn_bands_asymptotic(target_weight, sigma=max(target_weight, 0.05), lambda_L=band, lambda_M=band, gamma=-2)
        lower = max(0.0, min(lower, target_weight - min_band)) if target_weight > min_band else 0.0
        upper = min(1.0, max(upper, target_weight + min_band))
        lower_bands[ticker] = lower
        upper_bands[ticker] = upper
        if current_weight < lower or current_weight > upper:
            rebalance = True
            reason = f"rebalance: {ticker} peso={current_weight:.4f} fuera de [{lower:.4f}, {upper:.4f}]"
            break

    xeon_lower, xeon_upper = dn_bands_asymptotic(target_weight_xeon, sigma=max(target_weight_xeon, 0.05), lambda_L=band, lambda_M=band, gamma=-2)
    lower_bands[xeon_ticker] = max(0.0, xeon_lower)
    upper_bands[xeon_ticker] = min(1.0, xeon_upper)
    if not rebalance and (current_weight_xeon < lower_bands[xeon_ticker] or current_weight_xeon > upper_bands[xeon_ticker]):
        rebalance = True
        reason = f"rebalance: {xeon_ticker} peso={current_weight_xeon:.4f} fuera de [{lower_bands[xeon_ticker]:.4f}, {upper_bands[xeon_ticker]:.4f}]"

    final_risk_weights = target_risk_weights if rebalance else current_risk_weights
    weight_xeon = target_weight_xeon if rebalance else _weight_xeon_from_risk(final_risk_weights)

    target_weights_full = dict(target_risk_weights)
    target_weights_full[xeon_ticker] = target_weight_xeon
    final_weights_full = dict(final_risk_weights)
    final_weights_full[xeon_ticker] = weight_xeon

    return {
        "rebalance": rebalance,
        "target_weights": target_risk_weights,
        "final_weights": final_risk_weights,
        "weight_xeon": weight_xeon,
        "target_weights_full": target_weights_full,
        "final_weights_full": final_weights_full,
        "lower_bands": lower_bands,
        "upper_bands": upper_bands,
        "reason": reason,
    }
