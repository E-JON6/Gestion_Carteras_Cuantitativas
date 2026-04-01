"""
Black-Litterman con Omega Ratio como generador de views.

Implementación v1 orientada a corrección matemática del pipeline:
- XEON.DE participa en tickers, Sigma y prior
- XEON.DE NO genera views
- Views top-3 confirmadas por omega fast/slow
- Covarianza EWMA con shrinkage diagonal robusto
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OMEGA_WINDOW_FAST = 42
OMEGA_WINDOW_SLOW = 84
OMEGA_THRESHOLD = 0.0
N_TOP_VIEWS = 3
TAU = 0.05
DELTA = 2.5
EWMA_LAMBDA = 0.94
CONF_MIN = 0.30
CONF_MAX = 0.90
ANNUALIZATION = 252


def compute_omega_ratio(
    returns_series: pd.Series,
    window: int,
    threshold: float = OMEGA_THRESHOLD,
) -> float:
    recent = returns_series.dropna().iloc[-window:]
    if len(recent) < max(window // 2, 10):
        return np.nan

    gains = np.maximum(recent - threshold, 0.0).sum()
    losses = np.maximum(threshold - recent, 0.0).sum()

    if losses < 1e-12:
        return 10.0 if gains > 1e-12 else 1.0

    return float(min(gains / losses, 10.0))


def rank_etfs_by_omega(
    returns_df: pd.DataFrame,
    window: int,
    xeon_ticker: str = "XEON.DE",
) -> pd.Series:
    scores: dict[str, float] = {}
    for ticker in returns_df.columns:
        if ticker == xeon_ticker:
            continue
        scores[ticker] = compute_omega_ratio(returns_df[ticker], window)
    return pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)


def get_confirmed_top(
    returns_df: pd.DataFrame,
    xeon_ticker: str = "XEON.DE",
    n_top: int = N_TOP_VIEWS,
    omega_window_fast: int = OMEGA_WINDOW_FAST,
    omega_window_slow: int = OMEGA_WINDOW_SLOW,
) -> tuple[list[str], pd.Series, pd.Series]:
    omega_fast = rank_etfs_by_omega(returns_df, omega_window_fast, xeon_ticker)
    omega_slow = rank_etfs_by_omega(returns_df, omega_window_slow, xeon_ticker)

    top_fast = set(omega_fast.head(n_top).index)
    top_slow = set(omega_slow.head(n_top).index)
    confirmed = top_fast & top_slow

    if len(confirmed) >= max(1, n_top // 2):
        top_tickers = [ticker for ticker in omega_fast.index if ticker in confirmed][:n_top]
    else:
        top_tickers = list(omega_fast.head(n_top).index)

    if len(top_tickers) < n_top:
        for ticker in omega_fast.index:
            if ticker not in top_tickers:
                top_tickers.append(ticker)
            if len(top_tickers) >= n_top:
                break

    return top_tickers[:n_top], omega_fast, omega_slow


def _nearest_positive_semidefinite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.clip(eigvals, floor, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    return (repaired + repaired.T) / 2.0


def compute_covariance_matrix(
    returns_df: pd.DataFrame,
    ewma_lambda: float = EWMA_LAMBDA,
    annualization: int = ANNUALIZATION,
) -> np.ndarray:
    clean = returns_df.dropna(how="all").ffill(limit=1).dropna(axis=0, how="all")
    if clean.empty:
        raise ValueError("No hay retornos suficientes para estimar Sigma.")

    values = clean.fillna(0.0).to_numpy(dtype=float)
    t_obs, n_assets = values.shape
    if t_obs == 1:
        return np.eye(n_assets) * 1e-4

    alpha = 1.0 - ewma_lambda
    weights = alpha * (1.0 - alpha) ** np.arange(t_obs - 1, -1, -1)
    weights /= weights.sum()

    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    sigma_daily = np.einsum("t,ti,tj->ij", weights, centered, centered)
    sigma = sigma_daily * annualization
    sigma = (sigma + sigma.T) / 2.0

    target = np.diag(np.diag(sigma))
    shrink_intensity = min(0.35, max(0.05, n_assets / max(t_obs, 2) * 0.25))
    sigma = (1.0 - shrink_intensity) * sigma + shrink_intensity * target
    return _nearest_positive_semidefinite(sigma)


def _inverse_vol_weights(sigma: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.clip(np.diag(sigma), 1e-10, None))
    inv_vol = 1.0 / np.clip(vols, 1e-6, None)
    return inv_vol / inv_vol.sum()


def build_P_matrix(tickers: list[str], selected_tickers: list[str]) -> np.ndarray:
    p = np.zeros((len(selected_tickers), len(tickers)))
    for row, ticker in enumerate(selected_tickers):
        p[row, tickers.index(ticker)] = 1.0
    return p


def build_Q_vector(
    returns_df: pd.DataFrame,
    selected_tickers: list[str],
    window: int = OMEGA_WINDOW_SLOW,
    annualization: int = ANNUALIZATION,
) -> np.ndarray:
    if not selected_tickers:
        return np.array([], dtype=float)
    recent = returns_df[selected_tickers].tail(window)
    if recent.empty:
        return np.zeros(len(selected_tickers), dtype=float)
    return recent.mean().to_numpy(dtype=float) * annualization


def build_confidence_vector(omega_scores: pd.Series, selected_tickers: list[str]) -> np.ndarray:
    if not selected_tickers:
        return np.array([], dtype=float)
    selected = omega_scores.reindex(selected_tickers).replace([np.inf, -np.inf], np.nan)
    if selected.isna().all() or float(selected.max()) == float(selected.min()):
        return np.full(len(selected_tickers), 0.5, dtype=float)
    filled = selected.fillna(selected.median())
    scaled = (filled - filled.min()) / (filled.max() - filled.min())
    confidence = CONF_MIN + (CONF_MAX - CONF_MIN) * scaled
    return confidence.to_numpy(dtype=float)


def build_view_uncertainty(
    p_matrix: np.ndarray,
    sigma: np.ndarray,
    confidence: np.ndarray,
    tau: float = TAU,
) -> np.ndarray:
    if len(confidence) == 0:
        return np.zeros((0, 0), dtype=float)
    base_view_var = np.diag(p_matrix @ (tau * sigma) @ p_matrix.T)
    omega_diag = base_view_var / np.clip(confidence, 1e-6, None)
    return np.diag(np.clip(omega_diag, 1e-8, None))


def run_black_litterman(
    returns_df: pd.DataFrame,
    risk_free_rate: float | object,
    xeon_ticker: str | object = "XEON.DE",
    omega_window_fast: int = OMEGA_WINDOW_FAST,
    omega_window_slow: int = OMEGA_WINDOW_SLOW,
    n_top_views: int = N_TOP_VIEWS,
    tau: float = TAU,
    delta: float = DELTA,
    ewma_lambda: float = EWMA_LAMBDA,
    annualization: int = ANNUALIZATION,
) -> dict:
    if not isinstance(risk_free_rate, (int, float, np.floating)):
        risk_free_rate = 0.02
    if not isinstance(xeon_ticker, str):
        xeon_ticker = "XEON.DE"

    clean_returns = returns_df.copy()
    clean_returns = clean_returns.sort_index().dropna(axis=1, how="all")
    if xeon_ticker not in clean_returns.columns:
        xeon_ticker = "__MISSING_XEON__"
    tickers = list(clean_returns.columns)

    sigma = compute_covariance_matrix(clean_returns, ewma_lambda=ewma_lambda, annualization=annualization)
    market_weights = _inverse_vol_weights(sigma)
    pi = risk_free_rate + delta * (sigma @ market_weights)

    top_tickers, omega_fast, omega_slow = get_confirmed_top(
        clean_returns,
        xeon_ticker=xeon_ticker,
        n_top=n_top_views,
        omega_window_fast=omega_window_fast,
        omega_window_slow=omega_window_slow,
    )
    omega_scores = omega_fast.copy()

    if not top_tickers:
        mu_bl = np.asarray(pi, dtype=float)
        q = np.array([], dtype=float)
        confidence = np.array([], dtype=float)
        p_matrix = np.zeros((0, len(tickers)), dtype=float)
        omega_views = np.zeros((0, 0), dtype=float)
    else:
        p_matrix = build_P_matrix(tickers, top_tickers)
        q = build_Q_vector(clean_returns, top_tickers, window=omega_window_slow, annualization=annualization)
        confidence = build_confidence_vector(omega_fast, top_tickers)
        omega_views = build_view_uncertainty(p_matrix, sigma, confidence, tau=tau)

        tau_sigma_inv = np.linalg.pinv(tau * sigma)
        omega_inv = np.linalg.pinv(omega_views)
        middle = tau_sigma_inv + p_matrix.T @ omega_inv @ p_matrix
        rhs = tau_sigma_inv @ pi + p_matrix.T @ omega_inv @ q
        mu_bl = np.linalg.pinv(middle) @ rhs

    return {
        "mu_BL": np.asarray(mu_bl, dtype=float).flatten(),
        "Sigma": sigma,
        "tickers": tickers,
        "top_tickers": top_tickers,
        "omega_scores": omega_scores,
        "omega_scores_slow": omega_slow,
        "Q": np.asarray(q, dtype=float).flatten(),
        "confidence": np.asarray(confidence, dtype=float).flatten(),
        "P": p_matrix,
        "Omega_views": omega_views,
        "pi": np.asarray(pi, dtype=float).flatten(),
    }
