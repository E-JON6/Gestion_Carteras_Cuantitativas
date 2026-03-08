"""
Enfoque 3: Davis-Norman con Volatility Targeting.
La exposición al activo de riesgo se modula por la volatilidad realizada:
    α_adj = min(σ_target / σ_realizada, max_leverage)
Davis-Norman recalcula fronteras con σ actual y aplica bang-bang.
"""

import numpy as np
from gestion_cuantitativa.strategies.base import Strategy
from gestion_cuantitativa.models.merton import clip_allocation
from gestion_cuantitativa.models.davis_norman import dn_bands_asymptotic
from gestion_cuantitativa.config import (
    GAMMA, RECALIBRATION_FREQ, SIGMA_TARGET, MAX_LEVERAGE
)


class E3_DN_VolTarget(Strategy):
    """
    Volatility Targeting + Davis-Norman.
    - La posición se ajusta inversamente a la volatilidad realizada
    - α_adj = min(σ_target / σ_realizada, max_leverage)
    - Bandas DN alrededor de α_adj, recalculadas con σ actual
    - Regla bang-bang: solo operar cuando α sale de la banda
    """

    def __init__(self, gamma=None, recalib_freq=None,
                 sigma_target=None, max_leverage=None):
        super().__init__(name="E3: DN + Vol Targeting")
        self.gamma = gamma if gamma is not None else GAMMA
        self.recalib_freq = recalib_freq or RECALIBRATION_FREQ
        self.sigma_target = sigma_target or SIGMA_TARGET
        self.max_leverage = max_leverage or MAX_LEVERAGE

        # Estado interno
        self._alpha_adj = None
        self._alpha_lower = None
        self._alpha_upper = None
        self._last_recalib = -999
        self._sigma_current = None

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)

        if np.isnan(sigma) or sigma <= 0:
            return 0.5  # Default conservador

        # Vol targeting: α_adj = σ_target / σ_realizada
        alpha_adj = min(self.sigma_target / sigma, self.max_leverage)
        alpha_adj = clip_allocation(alpha_adj)

        self._alpha_adj = alpha_adj
        self._sigma_current = sigma

        lower, upper = dn_bands_asymptotic(
            alpha_adj, sigma, lambda_L, lambda_M, self.gamma
        )
        self._alpha_lower = lower
        self._alpha_upper = upper

        return alpha_adj

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        1. Recalibrar σ y α_adj periódicamente
        2. Recalcular bandas DN con σ actual alrededor de α_adj
        3. Filtro bang-bang: solo operar si α sale de la banda
        """
        # --- Recalibrar periódicamente ---
        if day_index - self._last_recalib >= self.recalib_freq:
            mu, sigma, r = estimator.get_params_at(date)

            if np.isnan(sigma) or sigma <= 0:
                return None

            self._sigma_current = sigma

            # Vol targeting: escalar exposición inversamente a la volatilidad
            alpha_adj = self.sigma_target / sigma
            alpha_adj = min(alpha_adj, self.max_leverage)
            alpha_adj = clip_allocation(alpha_adj)
            self._alpha_adj = alpha_adj

            # Bandas DN alrededor de α_adj
            if alpha_adj > 0.01:
                self._alpha_lower, self._alpha_upper = dn_bands_asymptotic(
                    alpha_adj, sigma, lambda_L, lambda_M, self.gamma
                )
            else:
                self._alpha_lower = 0.0
                self._alpha_upper = 0.05

            self._last_recalib = day_index

        # --- Verificar fronteras ---
        if self._alpha_lower is None or self._alpha_upper is None:
            return None

        # --- Filtro DN: solo operar si fuera de la banda ---
        current_alpha = portfolio.alpha

        if current_alpha < self._alpha_lower:
            return {
                'trade': True,
                'target_alpha': self._alpha_lower,
                'reason': (f'COMPRA (vol target): σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} < lower={self._alpha_lower:.3f}')
            }

        elif current_alpha > self._alpha_upper:
            return {
                'trade': True,
                'target_alpha': self._alpha_upper,
                'reason': (f'VENTA (vol target): σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} > upper={self._alpha_upper:.3f}')
            }

        else:
            return None
