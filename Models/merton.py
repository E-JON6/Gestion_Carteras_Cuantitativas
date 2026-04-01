"""
Merton multivariante con restricciones long-only y complemento en XEON.DE.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

GAMMA = -2
N_TOP_ASSETS = 5
MAX_WEIGHT = 0.40
MIN_WEIGHT = 0.01
MAX_SECTOR_WEIGHT = 0.25
FREEZE_EXIT_THR = 0.02
VOL_CAUTION_THR = 0.20
VOL_CRISIS_THR = 0.30


def detect_regime(sigma_mercado: float) -> tuple[str, float]:
    if sigma_mercado > VOL_CRISIS_THR:
        return "crisis", 0.40
    if sigma_mercado > VOL_CAUTION_THR:
        return "caution", 0.70
    return "normal", 1.00


def merton_weights_risk_only(
    mu_bl: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float,
    tickers: list[str],
    xeon_ticker: str = "XEON.DE",
    gamma: float = GAMMA,
) -> tuple[np.ndarray, list[str], list[int]]:
    risk_indices = [index for index, ticker in enumerate(tickers) if ticker != xeon_ticker]
    risk_tickers = [tickers[index] for index in risk_indices]
    if not risk_indices:
        return np.array([], dtype=float), [], []

    mu_risk = np.asarray(mu_bl, dtype=float)[risk_indices]
    sigma_risk = np.asarray(sigma, dtype=float)[np.ix_(risk_indices, risk_indices)]
    excess_return = mu_risk - float(risk_free_rate)
    gamma_eff = max(1e-6, 1.0 - gamma)

    try:
        sigma_inv = np.linalg.inv(sigma_risk)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma_risk)

    w_raw = (1.0 / gamma_eff) * sigma_inv @ excess_return
    return np.asarray(w_raw, dtype=float), risk_tickers, risk_indices


def apply_constraints(
    w_raw: np.ndarray,
    risk_tickers: list[str],
    categoria_por_ticker: dict[str, str] | None = None,
    n_top: int = N_TOP_ASSETS,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    max_sector: float = MAX_SECTOR_WEIGHT,
    max_risk_total: float = 1.0,
) -> np.ndarray:
    if len(w_raw) == 0:
        return np.array([], dtype=float)

    w = np.maximum(np.asarray(w_raw, dtype=float), 0.0)
    if w.sum() <= 1e-12:
        return np.zeros_like(w)

    sorted_idx = np.argsort(w)[::-1]
    keep = sorted_idx[: min(n_top, len(sorted_idx))]
    filtered = np.zeros_like(w)
    filtered[keep] = w[keep]
    w = np.minimum(filtered, max_weight)

    if categoria_por_ticker:
        category_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        for idx, ticker in enumerate(risk_tickers):
            category_to_indices[categoria_por_ticker.get(ticker, f"unknown_{ticker}")].append(idx)
        for indices in category_to_indices.values():
            sector_total = float(w[indices].sum())
            if sector_total > max_sector + 1e-12:
                scale = max_sector / sector_total
                w[indices] *= scale

    total = float(w.sum())
    if total > max_risk_total + 1e-12:
        w *= max_risk_total / total

    w[w < min_weight] = 0.0
    return w


def check_frozen_exits(
    current_weights: np.ndarray,
    tickers: list[str],
    frozen_tickers: list[str],
    xeon_ticker: str = "XEON.DE",
    threshold: float = FREEZE_EXIT_THR,
) -> list[str]:
    liquidar: list[str] = []
    current_weights = np.asarray(current_weights, dtype=float)
    for ticker in frozen_tickers or []:
        if ticker == xeon_ticker or ticker not in tickers:
            continue
        idx = tickers.index(ticker)
        if idx < len(current_weights) and current_weights[idx] < threshold:
            liquidar.append(ticker)
    return liquidar


def run_merton_v0(
    bl_result: dict,
    risk_free_rate: float,
    xeon_ticker: str = "XEON.DE",
    sigma_mercado: float = 0.15,
    categoria_por_ticker: dict[str, str] | None = None,
    current_weights: np.ndarray | None = None,
    frozen_tickers: list[str] | None = None,
    gamma: float = GAMMA,
) -> dict:
    mu_bl = np.asarray(bl_result["mu_BL"], dtype=float)
    sigma = np.asarray(bl_result["Sigma"], dtype=float)
    tickers = list(bl_result["tickers"])
    n_assets = len(tickers)

    regime, max_risk_total = detect_regime(float(sigma_mercado))
    w_raw_risk, risk_tickers, risk_indices = merton_weights_risk_only(
        mu_bl,
        sigma,
        risk_free_rate,
        tickers,
        xeon_ticker=xeon_ticker,
        gamma=gamma,
    )

    w_risk_final = apply_constraints(
        w_raw_risk,
        risk_tickers,
        categoria_por_ticker=categoria_por_ticker,
        max_risk_total=max_risk_total,
    )

    weights_array = np.zeros(n_assets, dtype=float)
    for local_idx, original_idx in enumerate(risk_indices):
        weights_array[original_idx] = w_risk_final[local_idx]

    risk_sum = float(weights_array.sum())
    weight_xeon = max(0.0, 1.0 - risk_sum)
    if xeon_ticker in tickers:
        weights_array[tickers.index(xeon_ticker)] = weight_xeon

    liquidar = []
    if current_weights is not None and frozen_tickers:
        liquidar = check_frozen_exits(
            np.asarray(current_weights, dtype=float),
            tickers,
            frozen_tickers,
            xeon_ticker=xeon_ticker,
        )

    selected_tickers = [
        risk_tickers[idx]
        for idx, weight in enumerate(w_risk_final)
        if weight > MIN_WEIGHT
    ]
    weights_dict = {
        risk_tickers[idx]: float(weight)
        for idx, weight in enumerate(w_risk_final)
        if weight > MIN_WEIGHT
    }

    w_raw = np.zeros(n_assets, dtype=float)
    for local_idx, original_idx in enumerate(risk_indices):
        w_raw[original_idx] = w_raw_risk[local_idx]

    return {
        "weights_array": weights_array,
        "weights_dict": weights_dict,
        "selected_tickers": selected_tickers,
        "weight_xeon": weight_xeon,
        "w_raw": w_raw,
        "regime": regime,
        "liquidar": liquidar,
        "weights": weights_dict,
        "selected_etfs": selected_tickers,
    }


def compute_optimal_weights(*args, **kwargs):
    return run_merton_v0(*args, **kwargs)


def run_merton_v0_legacy(*args, **kwargs):
    result = run_merton_v0(*args, **kwargs)
    legacy = dict(result)
    legacy["weights"] = result["weights_dict"]
    legacy["selected_etfs"] = result["selected_tickers"]
    return legacy
