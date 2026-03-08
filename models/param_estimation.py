"""
Estimación de parámetros del modelo: μ, σ, r.
Métodos: Rolling, EWMA, GARCH.
"""

import numpy as np
import pandas as pd
from gestion_cuantitativa.config import (
    EWMA_LAMBDA, ROLLING_WINDOW_MU, ROLLING_WINDOW_SIGMA,
    EWMA_WINDOW_SIGMA, TRADING_DAYS_PER_YEAR,
    MU_SHRINKAGE, MU_PRIOR_ERP
)


def compute_log_returns(prices):
    """Calcula retornos logarítmicos diarios."""
    return np.log(prices / prices.shift(1)).dropna()


def rolling_mu(returns, risk_free_series=None, window=None,
               shrinkage=None, prior_erp=None):
    """
    Estima μ (drift anualizado) con media rolling + shrinkage bayesiano.

    El shrinkage hacia una prima de riesgo histórica (Jorion 1986) reduce
    el error de estimación de μ, que es el parámetro más difícil de estimar
    en finanzas (Merton 1980).

    μ_shrunk = w · μ_rolling + (1-w) · (r + ERP_prior)

    Args:
        returns: pd.Series de retornos diarios
        risk_free_series: pd.Series con tasa libre de riesgo (para shrinkage prior)
        window: ventana en días (default: ROLLING_WINDOW_MU)
        shrinkage: factor de shrinkage (0=solo prior, 1=solo rolling)
        prior_erp: prima de riesgo equity del prior (default: MU_PRIOR_ERP)

    Returns:
        pd.Series con μ anualizado (capped a [-0.20, +0.20])
    """
    window = window or ROLLING_WINDOW_MU
    shrinkage = shrinkage if shrinkage is not None else MU_SHRINKAGE
    prior_erp = prior_erp if prior_erp is not None else MU_PRIOR_ERP

    mu_daily = returns.rolling(window).mean()
    # Anualizar: μ_annual = μ_daily * 252 + 0.5 * σ² (corrección log-normal)
    sigma_daily = returns.rolling(window).std()
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR + 0.5 * (sigma_daily ** 2) * TRADING_DAYS_PER_YEAR

    # ETFs de acumulación (total return): los retornos ya incluyen dividendos.
    # No se necesita ajuste por DIVIDEND_YIELD.

    # Shrinkage bayesiano hacia prior: μ_prior = r + ERP
    if risk_free_series is not None and shrinkage < 1.0:
        # Alinear risk-free con los retornos
        rf_aligned = risk_free_series.reindex(returns.index, method='ffill')
        rf_aligned = rf_aligned.fillna(method='bfill').fillna(0.02)
        mu_prior = rf_aligned + prior_erp
        mu_annual = shrinkage * mu_annual + (1.0 - shrinkage) * mu_prior
    elif shrinkage < 1.0:
        # Sin datos de rf, usar prior fijo
        mu_prior = 0.02 + prior_erp  # r ≈ 2% + ERP
        mu_annual = shrinkage * mu_annual + (1.0 - shrinkage) * mu_prior

    # Cap μ para evitar fracciones de Merton extremas
    # ±20% anualizado es un rango conservador para índices europeos
    mu_annual = mu_annual.clip(lower=-0.20, upper=0.20)

    return mu_annual


def rolling_sigma(returns, window=None):
    """
    Estima σ (volatilidad anualizada) con desviación estándar rolling.

    Args:
        returns: pd.Series de retornos diarios
        window: ventana en días (default: ROLLING_WINDOW_SIGMA)

    Returns:
        pd.Series con σ anualizado
    """
    window = window or ROLLING_WINDOW_SIGMA
    sigma_daily = returns.rolling(window).std()
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


def ewma_sigma(returns, ewma_lambda=None, window=None):
    """
    Estima σ con EWMA (Exponentially Weighted Moving Average).
    Método RiskMetrics: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t

    Args:
        returns: pd.Series de retornos diarios
        ewma_lambda: factor de decaimiento (default: 0.94)
        window: ventana mínima para inicializar (default: EWMA_WINDOW_SIGMA)

    Returns:
        pd.Series con σ anualizado
    """
    ewma_lambda = ewma_lambda or EWMA_LAMBDA
    window = window or EWMA_WINDOW_SIGMA

    # Usar pandas ewm con span correspondiente a λ
    # span = 2/(1-λ) - 1, pero es más directo usar alpha = 1-λ
    ewma_var = returns.pow(2).ewm(alpha=1 - ewma_lambda, min_periods=window).mean()
    sigma_daily = np.sqrt(ewma_var)
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


def get_daily_risk_free_rate(risk_free_series, dates):
    """
    Interpola la tasa libre de riesgo diaria para las fechas dadas.
    FRED proporciona datos no diarios; se forward-fill.

    Args:
        risk_free_series: pd.Series con tasa anualizada (decimal)
        dates: DatetimeIndex de fechas de trading

    Returns:
        pd.Series con tasa anualizada alineada con dates
    """
    # Forward-fill y reindex
    rf = risk_free_series.reindex(dates, method='ffill')
    rf = rf.fillna(method='bfill')  # Por si las primeras fechas no tienen dato
    return rf


class ParameterEstimator:
    """
    Clase que encapsula la estimación de μ, σ, r para el modelo.
    """

    def __init__(self, prices, risk_free_series,
                 sigma_method='ewma',
                 mu_window=None, sigma_window=None,
                 ewma_lambda=None):
        """
        Args:
            prices: pd.Series de precios de cierre del activo de riesgo
            risk_free_series: pd.Series con tasa libre de riesgo anualizada (decimal)
            sigma_method: 'rolling', 'ewma'
            mu_window: ventana para μ
            sigma_window: ventana para σ
            ewma_lambda: factor EWMA
        """
        self.prices = prices
        self.returns = compute_log_returns(prices)
        self.risk_free_raw = risk_free_series
        self.sigma_method = sigma_method
        self.mu_window = mu_window or ROLLING_WINDOW_MU
        self.sigma_window = sigma_window or ROLLING_WINDOW_SIGMA
        self.ewma_lambda = ewma_lambda or EWMA_LAMBDA

        # Pre-computar series completas (μ con shrinkage bayesiano)
        self._mu = rolling_mu(self.returns, risk_free_series=risk_free_series,
                              window=self.mu_window)

        if sigma_method == 'ewma':
            self._sigma = ewma_sigma(self.returns, self.ewma_lambda)
        elif sigma_method == 'rolling':
            self._sigma = rolling_sigma(self.returns, window=self.sigma_window)
        else:
            raise ValueError(f"sigma_method desconocido: {sigma_method}")

        self._r = get_daily_risk_free_rate(risk_free_series, self.returns.index)

    def get_params_at(self, date):
        """
        Devuelve (μ, σ, r) estimados para una fecha dada.

        Returns:
            tuple (mu, sigma, r) todos anualizados
        """
        mu = self._mu.loc[:date].iloc[-1] if date in self._mu.index or date >= self._mu.index[0] else np.nan
        sigma = self._sigma.loc[:date].iloc[-1] if date in self._sigma.index or date >= self._sigma.index[0] else np.nan
        r = self._r.loc[:date].iloc[-1] if date in self._r.index or date >= self._r.index[0] else np.nan

        # Buscar el valor más reciente disponible
        try:
            mu = self._mu.loc[:date].dropna().iloc[-1]
        except IndexError:
            mu = np.nan
        try:
            sigma = self._sigma.loc[:date].dropna().iloc[-1]
        except IndexError:
            sigma = np.nan
        try:
            r = self._r.loc[:date].dropna().iloc[-1]
        except IndexError:
            r = np.nan

        return float(mu), float(sigma), float(r)

    @property
    def mu_series(self):
        return self._mu

    @property
    def sigma_series(self):
        return self._sigma

    @property
    def r_series(self):
        return self._r
