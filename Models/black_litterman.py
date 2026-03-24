"""
Black-Litterman version 0.
Recibe: retornos, ETFs seleccionados y scores de omega.
Devuelve: mu y Sigma inventadas para conectar con Merton.
"""

import numpy as np


def run_black_litterman_v0(returns_df, selected_etfs, omega_scores):
    tickers = list(returns_df.columns)
    n_assets = len(tickers)

    mu_bl = np.array([0.05 + 0.01 * i for i in range(n_assets)])

    sigma = np.full((n_assets, n_assets), 0.02)
    np.fill_diagonal(sigma, 0.10)

    top_tickers = selected_etfs
    q = np.array([0.06 + 0.01 * i for i in range(len(top_tickers))]) # Son los retornos esperados por el modelo de Black-Litterman
    confidence = np.array([0.5 for _ in top_tickers]) #Cuanta confianza le damos a los retornos

    return {
        "mu_BL": mu_bl,
        "Sigma": sigma,
        "tickers": tickers,
        "top_tickers": top_tickers,
        "omega_scores": omega_scores,
        "Q": q,
        "confidence": confidence,
    }