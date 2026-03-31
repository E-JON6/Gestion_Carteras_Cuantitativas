"""
Black-Litterman version 0.
Recibe: retornos, ETFs seleccionados y scores de omega.
Devuelve: mu y Sigma inventadas para conectar con Merton.
"""


import numpy as np
import pandas as pd


def build_P_matrix(tickers, selected_etfs):
    """
    Matriz P de views absolutas.
    Una fila por view, una columna por activo del universo.
    """
    n_assets = len(tickers)
    k_views = len(selected_etfs)

    P = np.zeros((k_views, n_assets))

    for i, etf in enumerate(selected_etfs):
        j = tickers.index(etf)
        P[i, j] = 1.0

    return P


def build_Q_vector(returns_df, selected_etfs, window=42, annualization=252):
    """
    Vector Q: retorno reciente anualizado de los ETFs seleccionados.
    """
    recent = returns_df.tail(window)
    q_series = recent[selected_etfs].mean() * annualization
    return q_series.values


def build_confidence_vector(omega_scores, selected_etfs):
    """
    Convierte los scores Omega de los ETFs seleccionados
    en un vector de confianza entre 0.3 y 0.9.
    """
    omega_sel = omega_scores.loc[selected_etfs].replace([np.inf, -np.inf], np.nan)

    if omega_sel.isna().all():
        return np.ones(len(selected_etfs)) * 0.5

    x = omega_sel.fillna(omega_sel.median())

    if x.max() == x.min():
        return np.ones(len(selected_etfs)) * 0.5

    conf = 0.3 + 0.6 * (x - x.min()) / (x.max() - x.min())
    return conf.values


def build_view_uncertainty(P, Sigma, confidence, tau=0.05):
    """
    Matriz Omega_views de incertidumbre de las views.
    Ojo: esto NO es el ratio Omega, sino la matriz de error de views.
    """
    base_view_var = np.diag(P @ (tau * Sigma) @ P.T)
    omega_diag = base_view_var / np.clip(confidence, 1e-6, None)
    return np.diag(omega_diag)


def run_black_litterman(
    returns_df: pd.DataFrame,
    selected_etfs: list,
    omega_scores: pd.Series,
    rf: float = 0.02,
    delta: float = 2.5,
    tau: float = 0.05,
    window_q: int = 42,
    annualization: int = 252,
    ridge: float = 1e-6,
):
    """
    Black-Litterman v1:
    - Sigma histórica anualizada
    - prior con pesos iguales
    - views absolutas sobre selected_etfs
    - Q con retornos recientes anualizados
    - confianza derivada de omega_scores
    """

    returns_df = returns_df.copy()
    tickers = list(returns_df.columns)
    n_assets = len(tickers)

    if len(selected_etfs) == 0:
        raise ValueError("selected_etfs está vacío")

    # 1) Sigma anualizada
    sigma_df = returns_df.cov() * annualization
    sigma_df = sigma_df + np.eye(n_assets) * ridge
    Sigma = sigma_df.values

    # 2) Prior pi = r + delta * Sigma * w_mkt
    w_mkt = np.ones(n_assets) / n_assets
    pi = rf + delta * (Sigma @ w_mkt)

    # 3) Views
    P = build_P_matrix(tickers, selected_etfs)
    Q = build_Q_vector(returns_df, selected_etfs, window=window_q, annualization=annualization)
    confidence = build_confidence_vector(omega_scores, selected_etfs)
    Omega_views = build_view_uncertainty(P, Sigma, confidence, tau=tau)

    # 4) Posterior BL
    tauSigma_inv = np.linalg.pinv(tau * Sigma)
    Omega_inv = np.linalg.pinv(Omega_views)

    middle = tauSigma_inv + P.T @ Omega_inv @ P
    rhs = tauSigma_inv @ pi + P.T @ Omega_inv @ Q
    mu_bl = np.linalg.pinv(middle) @ rhs

    return {
        "mu_BL": mu_bl,
        "Sigma": Sigma,
        "tickers": tickers,
        "top_tickers": selected_etfs,
        "omega_scores": omega_scores,
        "Q": Q,
        "confidence": confidence,
        "P": P,
        "Omega_views": Omega_views,
        "pi": pi,
    }