"""
Approximation version 0.5 de Davis-Norman.

Idea:
- Merton propone pesos objetivo para activos de riesgo.
- Davis-Norman decide si se rebalancea o no usando bandas de inaccion.
- XEON.DE absorbe el complemento hasta sumar exactamente 1.

Esta version sigue siendo sencilla, pero ya respeta mejor la logica de una estrategia
cuantitativa real y no rompe el flujo con Merton, backtesting ni registrador.
"""


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


def _band_width(target_weight, base_band, min_band):
    return max(min_band, base_band * max(target_weight, 1e-6))


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

    all_risk_tickers = sorted(set(target_risk_weights) | set(current_risk_weights))
    lower_bands = {}
    upper_bands = {}
    rebalance = False
    reason = "no rebalance"
    first_breach_reason = None

    for ticker in all_risk_tickers:
        target_weight = target_risk_weights.get(ticker, 0.0)
        current_weight = current_risk_weights.get(ticker, 0.0)
        width = _band_width(target_weight, band, min_band)
        lower = max(0.0, target_weight - width)
        upper = min(1.0, target_weight + width)

        lower_bands[ticker] = lower
        upper_bands[ticker] = upper

        if (current_weight < lower or current_weight > upper) and first_breach_reason is None:
            rebalance = True
            first_breach_reason = (
                f"rebalance: {ticker} peso={current_weight:.4f} "
                f"fuera de [{lower:.4f}, {upper:.4f}]"
            )

    xeon_width = _band_width(target_weight_xeon, band, min_band)
    xeon_lower = max(0.0, target_weight_xeon - xeon_width)
    xeon_upper = min(1.0, target_weight_xeon + xeon_width)
    lower_bands[xeon_ticker] = xeon_lower
    upper_bands[xeon_ticker] = xeon_upper

    if first_breach_reason is None:
        if current_weight_xeon < xeon_lower or current_weight_xeon > xeon_upper:
            rebalance = True
            first_breach_reason = (
                f"rebalance: {xeon_ticker} peso={current_weight_xeon:.4f} "
                f"fuera de [{xeon_lower:.4f}, {xeon_upper:.4f}]"
            )

    if first_breach_reason is not None:
        reason = first_breach_reason

    if rebalance:
        final_risk_weights = target_risk_weights
        weight_xeon = target_weight_xeon
    else:
        final_risk_weights = current_risk_weights
        weight_xeon = _weight_xeon_from_risk(final_risk_weights)

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
