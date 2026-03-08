"""
Modelo de Merton (1971).
Fracción óptima en el activo de riesgo bajo utilidad potencial.
"""

import numpy as np
from gestion_cuantitativa.config import GAMMA, MAX_LEVERAGE, MIN_POSITION


def merton_fraction(mu, sigma, r, gamma=None):
    """
    Calcula la fracción óptima de Merton en el activo de riesgo.

    Para utilidad potencial U(w) = w^γ / γ:
        α* = (μ - r) / (σ² · (1 - γ))

    Para utilidad logarítmica (γ → 0):
        α* = (μ - r) / σ²

    Args:
        mu: drift anualizado del activo de riesgo
        sigma: volatilidad anualizada
        r: tasa libre de riesgo anualizada
        gamma: parámetro de aversión al riesgo (default: GAMMA)

    Returns:
        float: fracción óptima α*
    """
    gamma = gamma if gamma is not None else GAMMA

    if sigma <= 0 or np.isnan(sigma):
        return 0.0

    if np.isnan(mu) or np.isnan(r):
        return 0.0

    excess_return = mu - r

    if abs(gamma) < 1e-10:
        # Utilidad logarítmica
        alpha_star = excess_return / (sigma ** 2)
    else:
        alpha_star = excess_return / (sigma ** 2 * (1 - gamma))

    return alpha_star


def clip_allocation(alpha, max_leverage=None, min_position=None):
    """
    Aplica restricciones de apalancamiento y posición mínima.

    Args:
        alpha: fracción deseada
        max_leverage: máximo apalancamiento (default: MAX_LEVERAGE = 2.0)
        min_position: posición mínima (default: MIN_POSITION = 0.01)

    Returns:
        float: fracción ajustada
    """
    max_leverage = max_leverage if max_leverage is not None else MAX_LEVERAGE
    min_position = min_position if min_position is not None else MIN_POSITION

    # Limitar al máximo apalancamiento
    alpha = min(alpha, max_leverage)

    # Si es muy pequeño pero positivo, forzar a mínimo o a 0
    if 0 < alpha < min_position:
        alpha = 0.0  # No vale la pena tener una posición tan pequeña

    # Si es negativo (short), no lo permitimos en este contexto
    alpha = max(alpha, 0.0)

    return alpha
