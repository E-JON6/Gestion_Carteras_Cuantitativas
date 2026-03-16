"""
Enfoque 1: Davis-Norman con Parámetros Adaptativos.
El más fiel al TFG: Merton + bandas DN recalibradas periódicamente.
"""

import numpy as np
from strategies.base import Strategy
from models.merton import merton_fraction, clip_allocation
from models.davis_norman import dn_bands_asymptotic
from config import GAMMA, RECALIBRATION_FREQ


class E1_DN_Adaptive(Strategy):
    """
    Davis-Norman con parámetros adaptativos.
    - α* de Merton recalculado periódicamente
    - Bandas asimétricas (fórmula asintótica Shreve-Soner)
    - Regla bang-bang: operar a la frontera, no al centro
    """

    def __init__(self, gamma=None, recalib_freq=None):
        super().__init__(name="E1: DN Adaptativo")
        self.gamma = gamma if gamma is not None else GAMMA
        self.recalib_freq = recalib_freq or RECALIBRATION_FREQ

        # Estado interno: últimas fronteras calculadas
        self._alpha_star = None
        self._alpha_lower = None
        self._alpha_upper = None
        self._last_recalib = -999

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)
        alpha = merton_fraction(mu, sigma, r, self.gamma)
        alpha = clip_allocation(alpha)

        # Calcular bandas iniciales
        lower, upper = dn_bands_asymptotic(alpha, sigma, lambda_L, lambda_M, self.gamma)
        self._alpha_star = alpha
        self._alpha_lower = lower
        self._alpha_upper = upper

        return alpha

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        1. Recalibrar fronteras periódicamente (cada recalib_freq días)
        2. Comprobar si α actual está fuera de la banda
        3. Si fuera → operar hacia la frontera (bang-bang)
        """
        # --- Recalibrar parámetros periódicamente ---
        if day_index - self._last_recalib >= self.recalib_freq:
            mu, sigma, r = estimator.get_params_at(date)

            if not np.isnan(mu) and not np.isnan(sigma) and not np.isnan(r):
                self._alpha_star = merton_fraction(mu, sigma, r, self.gamma)
                self._alpha_star = clip_allocation(self._alpha_star)

                self._alpha_lower, self._alpha_upper = dn_bands_asymptotic(
                    self._alpha_star, sigma, lambda_L, lambda_M, self.gamma
                )
                self._last_recalib = day_index

        # --- Verificar que tenemos fronteras ---
        if self._alpha_lower is None or self._alpha_upper is None:
            return None

        # --- Comprobar posición actual contra bandas ---
        current_alpha = portfolio.alpha

        if current_alpha < self._alpha_lower:
            # Región de COMPRA: llevar α hasta la frontera inferior
            return {
                'trade': True,
                'target_alpha': self._alpha_lower,
                'reason': (f'COMPRA (bang-bang): α={current_alpha:.3f} < '
                          f'lower={self._alpha_lower:.3f}')
            }

        elif current_alpha > self._alpha_upper:
            # Región de VENTA: llevar α hasta la frontera superior
            return {
                'trade': True,
                'target_alpha': self._alpha_upper,
                'reason': (f'VENTA (bang-bang): α={current_alpha:.3f} > '
                          f'upper={self._alpha_upper:.3f}')
            }

        else:
            # Región de NO-TRANSACCIÓN: no hacer nada
            return None
