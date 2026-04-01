"""
Aproximación asintótica simple para bandas de no-transacción tipo Davis-Norman.
"""

from __future__ import annotations

import numpy as np


def dn_bands_asymptotic(
    alpha_star: float,
    sigma: float,
    lambda_L: float,
    lambda_M: float,
    gamma: float,
) -> tuple[float, float]:
    alpha = float(np.clip(alpha_star, 1e-6, 1 - 1e-6))
    sigma_eff = float(max(abs(sigma), 1e-4))
    gamma_eff = float(max(1e-6, 1.0 - gamma))
    lambda_eff = float(max(1e-6, 0.5 * (abs(lambda_L) + abs(lambda_M))))

    half_width = ((3.0 / (2.0 * gamma_eff)) * (alpha**2) * ((1.0 - alpha) ** 2) * lambda_eff) ** (1.0 / 3.0)
    half_width *= sigma_eff ** (2.0 / 3.0)
    half_width = float(np.clip(half_width, 0.01, 0.35))

    lower = max(0.0, alpha - half_width)
    upper = min(1.0, alpha + half_width)
    return lower, upper
