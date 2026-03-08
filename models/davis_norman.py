"""
Modelo de Davis-Norman (1990).
Bandas de no-transacción óptimas con costes proporcionales.

Implementación: Fórmula asintótica estacionaria de Janecek & Shreve (2004),
basada en la expansión de Shreve & Soner (1994).

Referencias:
  - Shreve & Soner (1994), "Optimal Investment and Consumption
    with Transaction Costs", Annals of Applied Probability.
  - Janecek & Shreve (2004), "Asymptotic Analysis for Optimal Investment
    and Consumption with Transaction Costs", Finance & Stochastics.
"""

import numpy as np
from gestion_cuantitativa.config import GAMMA, MAX_LEVERAGE


# ============================================================
# FÓRMULA ASINTÓTICA (Janecek-Shreve / Shreve-Soner)
# ============================================================

def dn_bands_asymptotic(alpha_star, sigma, lambda_L, lambda_M, gamma=None):
    """
    Calcula las bandas de no-transacción usando la fórmula asintótica
    de Janecek & Shreve (2004, Theorem 4.2).

    Para costes proporcionales pequeños λ, el ancho de la zona de
    no-transacción en espacio de proporciones es:

        Δα ≈ ( 3·λ·α*²·(1-α*)²·σ² / (4·(1-γ)) )^(1/3)

    El factor (1-α*)² refleja la geometría del espacio de proporciones:
    cuando α* → 1, pequeños movimientos de precio causan grandes cambios
    en la proporción, por lo que la banda se estrecha.

    Con λ_L ≠ λ_M las bandas son asimétricas:
        α_L = α* - ( 3·λ_L·α*²·(1-α*)²·σ² / (4·(1-γ)) )^(1/3)
        α_U = α* + ( 3·λ_M·α*²·(1-α*)²·σ² / (4·(1-γ)) )^(1/3)

    Args:
        alpha_star: fracción óptima de Merton (α*)
        sigma: volatilidad anualizada (σ)
        lambda_L: coste proporcional de compra (λ_L)
        lambda_M: coste proporcional de venta (λ_M)
        gamma: parámetro de utilidad potencial (γ < 1)

    Returns:
        tuple (alpha_lower, alpha_upper)
    """
    gamma = gamma if gamma is not None else GAMMA

    if alpha_star <= 0 or sigma <= 0 or np.isnan(alpha_star) or np.isnan(sigma):
        return (0.0, 0.0)

    # Factor: 3 / (4·(1-γ))  para utilidad potencial U(w) = w^γ / γ
    factor = 3.0 / (4.0 * (1.0 - gamma)) if abs(1.0 - gamma) > 1e-10 else 3.0 / 4.0

    # (1-α*)² — geometría del espacio de proporciones
    # Para α* > 1 (apalancamiento), usamos (α*-1)² que es equivalente
    one_minus_alpha_sq = (1.0 - alpha_star) ** 2

    delta_lower = (factor * lambda_L * alpha_star ** 2 * one_minus_alpha_sq * sigma ** 2) ** (1.0 / 3.0)
    delta_upper = (factor * lambda_M * alpha_star ** 2 * one_minus_alpha_sq * sigma ** 2) ** (1.0 / 3.0)

    alpha_lower = alpha_star - delta_lower
    alpha_upper = alpha_star + delta_upper

    # Clamp a rango razonable
    alpha_lower = max(alpha_lower, 0.0)
    alpha_upper = min(alpha_upper, MAX_LEVERAGE)

    return (alpha_lower, alpha_upper)
