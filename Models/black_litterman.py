"""
Black-Litterman con covarianza EWMA y views desde senal compuesta.

Pipeline:
  1. Covarianza robusta: blend corto/largo + EWMA en diagonal (RiskMetrics)
  2. Prior de equilibrio: pi = rf + delta * Sigma * w_mkt
  3. Views absolutas para TODOS los ETFs de riesgo (senal compuesta)
  4. Posterior BL: combina prior + views segun confianza
  5. Log de decisiones a Decisiones/{fecha}/

Referencia: Black & Litterman (1992), Idzorek (2007)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# COVARIANZA ROBUSTA
# ============================================================

def compute_covariance_blended(
    returns_df: pd.DataFrame,
    short_window: int = 63,
    long_window: int = 252,
    blend_alpha: float = 0.6,
    ewma_lambda: float = 0.94,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """
    Matriz de covarianza anualizada para inputs de BL y Merton.

    Por que asi:
      - Mezcla corto/largo: la correlacion reacciona a regimen reciente pero no
        ignora estructura estable (blend_alpha pesa la ventana corta).
      - Volatilidades en diagonal con EWMA (estilo RiskMetrics): captura clusters
        de volatilidad mejor que una sola ventana fija.
      - Shrinkage hacia la identidad: estabilidad numerica con muchos activos.
      - Proyeccion PSD: evita que Sigma tenga autovalores negativos al invertir.

    Pasos:
      1. Covarianzas muestrales corta y larga (x252), blend lineal.
      2. Extraer matriz de correlacion del blend.
      3. Diagonal EWMA de vols → Sigma = D_ewma @ Corr @ D_ewma.
    """
    avail_short = min(short_window, len(returns_df))
    avail_long = min(long_window, len(returns_df))

    sigma_short = returns_df.tail(avail_short).cov().values * 252
    sigma_long = returns_df.tail(avail_long).cov().values * 252
    sigma_blend = blend_alpha * sigma_short + (1 - blend_alpha) * sigma_long

    # Correlaciones del blend
    d = np.sqrt(np.maximum(np.diag(sigma_blend), 1e-10))
    d_inv = np.diag(1.0 / d)
    corr = d_inv @ sigma_blend @ d_inv
    np.fill_diagonal(corr, 1.0)

    # EWMA volatilidades (diagonal)
    if 0 < ewma_lambda < 1:
        halflife = np.log(0.5) / np.log(ewma_lambda)
    else:
        halflife = 60.0
    ewma_var = returns_df.ewm(halflife=halflife, min_periods=20).var().iloc[-1].values
    ewma_vol = np.sqrt(np.maximum(ewma_var, 1e-10) * 252)

    # Reconstruir: Sigma = D_ewma @ Corr @ D_ewma
    d_ewma = np.diag(ewma_vol)
    sigma_final = d_ewma @ corr @ d_ewma

    # Shrinkage hacia diagonal (estabiliza matrices grandes, N>20)
    if shrinkage > 0:
        n = sigma_final.shape[0]
        mu_cov = np.trace(sigma_final) / n
        sigma_final = (1 - shrinkage) * sigma_final + shrinkage * mu_cov * np.eye(n)

    # Garantizar PSD
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_final)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    sigma_final = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    sigma_final = (sigma_final + sigma_final.T) / 2

    return sigma_final


# ============================================================
# PRIOR DE EQUILIBRIO
# ============================================================

def compute_market_weights_prior(returns_df: pd.DataFrame, mode: str = "equal") -> np.ndarray:
    """
    Pesos del 'mercado' implicito en pi = rf + delta * Sigma @ w_mkt.

    - equal: 1/N (neutral).
    - inv_vol: 1/sigma_i anualizada, normalizado — aproxima menor peso en activos mas volatiles.
    """
    n = returns_df.shape[1]
    if n == 0:
        return np.array([])

    mode = (mode or "equal").lower().strip()
    if mode == "inv_vol":
        vol_d = returns_df.std().values * np.sqrt(252)
        vol_d = np.maximum(vol_d, 1e-8)
        w = 1.0 / vol_d
        w = w / w.sum()
        return w

    return np.ones(n) / n


def compute_prior(
    Sigma: np.ndarray,
    delta: float,
    rf: float,
    w_mkt: np.ndarray,
) -> np.ndarray:
    """pi = rf + delta * Sigma @ w_mkt (CAPM implicito en equilibrio si w_mkt es el mercado)."""
    return rf + delta * (Sigma @ w_mkt)


# ============================================================
# VIEWS DESDE SENAL COMPUESTA
# ============================================================

def build_views_from_signal(
    composite_scores: pd.Series,
    tickers: list[str],
    view_scale: float = 0.05,
    pi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Views absolutas sobre retorno esperado, ancladas al prior BL.

    - P = I_N: una view por activo, linealmente independientes (vista i solo afecta activo i).
      Es el caso estandar cuando las opiniones son "creo que el retorno del activo i es ...".
      Alternativas (no implementadas aqui): views relativas P con filas tipo (0,...,1,...,-1,...).

    Q_i = pi_i + view_scale * score_i (score ya en z-score cross-section).

    confidence_i = 0.3 + 0.6 * |score_i|/max|score|: mas confianza si la senal es extrema
    frente al resto de ETFs ese dia (Idzorek-style continuo).

    Returns (P, Q, confidence)
    """
    n_assets = len(tickers)
    aligned = composite_scores.reindex(tickers, fill_value=0.0)

    P = np.eye(n_assets)
    base = pi if pi is not None else np.zeros(n_assets)
    Q = base + view_scale * aligned.values

    abs_scores = np.abs(aligned.values)
    max_abs = abs_scores.max() if abs_scores.max() > 1e-8 else 1.0
    confidence = 0.3 + 0.6 * np.minimum(abs_scores / max_abs, 1.0)

    return P, Q, confidence


def build_view_uncertainty(
    P: np.ndarray,
    Sigma: np.ndarray,
    confidence: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Omega_views: incertidumbre de cada view, inversamente proporcional a la confianza."""
    base_var = np.diag(P @ (tau * Sigma) @ P.T)
    omega_diag = base_var / np.clip(confidence, 0.1, None)
    return np.diag(omega_diag)


# ============================================================
# POSTERIOR
# ============================================================

def bl_posterior(
    pi: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega_views: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """mu_BL = inv(inv(tau*Sigma) + P'*inv(Omega)*P) @ (inv(tau*Sigma)*pi + P'*inv(Omega)*Q)"""
    tau_Sigma_inv = np.linalg.pinv(tau * Sigma)
    Omega_inv = np.linalg.pinv(Omega_views)

    middle = tau_Sigma_inv + P.T @ Omega_inv @ P
    rhs = tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q
    return np.linalg.solve(middle, rhs)


# ============================================================
# LOGGING DE DECISIONES
# ============================================================

def log_decisions(
    review_date,
    tickers: list[str],
    composite_scores: pd.Series,
    pi: np.ndarray,
    Q: np.ndarray,
    confidence: np.ndarray,
    mu_bl: np.ndarray,
    Sigma: np.ndarray,
    base_dir: str = "Decisiones",
) -> None:
    """Guarda decisiones de BL en Decisiones/{fecha}/ como JSON y CSV."""
    if review_date is None:
        return

    date_str = pd.Timestamp(review_date).strftime("%Y-%m-%d")
    folder = Path(base_dir) / date_str
    folder.mkdir(parents=True, exist_ok=True)

    # JSON detallado
    detail = {
        "fecha": date_str,
        "tickers": tickers,
        "composite_scores": {t: round(float(composite_scores.get(t, 0.0)), 6) for t in tickers},
        "prior_pi": {t: round(float(pi[i]), 6) for i, t in enumerate(tickers)},
        "views_Q": {t: round(float(Q[i]), 6) for i, t in enumerate(tickers)},
        "confidence": {t: round(float(confidence[i]), 4) for i, t in enumerate(tickers)},
        "mu_BL_posterior": {t: round(float(mu_bl[i]), 6) for i, t in enumerate(tickers)},
    }
    with open(folder / "bl_decisions.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    # CSV para revision rapida
    pd.DataFrame({
        "Ticker": tickers,
        "Composite_Score": [round(float(composite_scores.get(t, 0.0)), 6) for t in tickers],
        "Prior_pi": [round(float(v), 6) for v in pi],
        "View_Q": [round(float(v), 6) for v in Q],
        "Confidence": [round(float(v), 4) for v in confidence],
        "mu_BL": [round(float(v), 6) for v in mu_bl],
    }).to_csv(folder / "bl_decisions.csv", index=False)

    # Matriz de covarianza
    pd.DataFrame(Sigma, index=tickers, columns=tickers).to_csv(folder / "covariance.csv")


# ============================================================
# FUNCION PRINCIPAL
# ============================================================

def run_black_litterman(
    returns_df: pd.DataFrame,
    composite_scores: pd.Series,
    rf: float = 0.03,
    delta: float = 2.5,
    tau: float = 0.05,
    view_scale: float = 0.05,
    short_window: int = 63,
    long_window: int = 252,
    blend_alpha: float = 0.6,
    ewma_lambda: float = 0.94,
    shrinkage: float = 0.0,
    review_date=None,
    log_dir: str = "Decisiones",
    prior_weights_mode: str = "equal",
) -> dict:
    """
    Pipeline completo de Black-Litterman.

    Recibe:
      - returns_df: retornos diarios de ETFs de riesgo (sin XEON.DE)
      - composite_scores: pd.Series con score compuesto por ETF

    Devuelve:
      - mu_BL, Sigma, tickers + detalle de views para debugging
    """
    tickers = list(returns_df.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        raise ValueError("No hay activos de riesgo para Black-Litterman.")

    # 1) Covarianza robusta (blend + EWMA + shrinkage)
    Sigma = compute_covariance_blended(
        returns_df, short_window, long_window, blend_alpha, ewma_lambda, shrinkage,
    )

    # 2) Prior de equilibrio
    w_mkt = compute_market_weights_prior(returns_df, mode=prior_weights_mode)
    pi = compute_prior(Sigma, delta, rf, w_mkt)

    # 3) Views desde senal compuesta
    P, Q, confidence = build_views_from_signal(composite_scores, tickers, view_scale, pi)

    # 4) Incertidumbre de views
    Omega_views = build_view_uncertainty(P, Sigma, confidence, tau)

    # 5) Posterior
    mu_bl = bl_posterior(pi, Sigma, P, Q, Omega_views, tau)

    # 6) Logging
    log_decisions(
        review_date, tickers, composite_scores,
        pi, Q, confidence, mu_bl, Sigma,
        base_dir=log_dir,
    )

    return {
        "mu_BL": mu_bl,
        "Sigma": Sigma,
        "tickers": tickers,
        "pi": pi,
        "w_mkt_prior": w_mkt,
        "P": P,
        "Q": Q,
        "confidence": confidence,
        "Omega_views": Omega_views,
        "composite_scores": composite_scores,
    }
