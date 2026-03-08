"""
Enfoque 2: Davis-Norman como Filtro de Ejecución sobre Momentum.
Señal de momentum (12-1 meses, Jegadeesh-Titman) genera α_target.
Davis-Norman decide CUÁNDO ejecutar: solo si la desviación supera la frontera.
"""

import numpy as np
from gestion_cuantitativa.strategies.base import Strategy
from gestion_cuantitativa.models.merton import merton_fraction, clip_allocation
from gestion_cuantitativa.models.davis_norman import dn_bands_asymptotic
from gestion_cuantitativa.config import (
    GAMMA, RECALIBRATION_FREQ,
    MOMENTUM_LOOKBACK, MOMENTUM_SKIP, ALPHA_MIN_MOMENTUM
)


class E2_DN_Momentum(Strategy):
    """
    Momentum + Davis-Norman como filtro de ejecución.
    - Señal: retorno acumulado 12 meses excluyendo último mes
    - Si momentum > 0 → α_target = α* (Merton, posición larga completa)
    - Si momentum < 0 → α_target = α_min (posición defensiva)
    - Filtro DN: solo operar si α actual se sale de la banda alrededor de α_target
    - Bandas se recalculan con σ estimada actual (EWMA)
    """

    def __init__(self, gamma=None, recalib_freq=None,
                 momentum_lookback=None, momentum_skip=None,
                 alpha_min=None):
        super().__init__(name="E2: DN + Momentum")
        self.gamma = gamma if gamma is not None else GAMMA
        self.recalib_freq = recalib_freq or RECALIBRATION_FREQ
        self.momentum_lookback = momentum_lookback or MOMENTUM_LOOKBACK
        self.momentum_skip = momentum_skip or MOMENTUM_SKIP
        self.alpha_min = alpha_min if alpha_min is not None else ALPHA_MIN_MOMENTUM

        # Estado interno
        self._alpha_target = None
        self._alpha_lower = None
        self._alpha_upper = None
        self._last_recalib = -999
        self._momentum_signal = None

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)
        alpha = merton_fraction(mu, sigma, r, self.gamma)
        alpha = clip_allocation(alpha)
        self._alpha_target = alpha

        lower, upper = dn_bands_asymptotic(alpha, sigma, lambda_L, lambda_M, self.gamma)
        self._alpha_lower = lower
        self._alpha_upper = upper

        return alpha

    def _compute_momentum(self, date, prices):
        """
        Calcula la señal de momentum (12-1):
        retorno acumulado de los últimos 12 meses excluyendo el último mes.

        Returns:
            float: señal de momentum (positivo = alcista)
        """
        # Encontrar la posición de la fecha actual
        idx = prices.index.get_indexer([date], method='ffill')[0]

        if idx < self.momentum_lookback:
            return 0.0  # No hay suficientes datos

        # Precio hace 12 meses
        price_12m = prices.iloc[idx - self.momentum_lookback]
        # Precio hace 1 mes
        price_1m = prices.iloc[idx - self.momentum_skip]

        if price_12m <= 0:
            return 0.0

        # Retorno 12-1 meses
        momentum = (price_1m / price_12m) - 1.0

        return momentum

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        1. Calcular señal de momentum
        2. Determinar α_target según momentum
        3. Recalibrar bandas DN alrededor de α_target
        4. Solo operar si α actual se sale de la banda (filtro DN)
        """
        prices = extra_data.get('prices')
        if prices is None:
            return None

        # --- Recalibrar periódicamente ---
        if day_index - self._last_recalib >= self.recalib_freq:
            mu, sigma, r = estimator.get_params_at(date)

            if np.isnan(mu) or np.isnan(sigma) or np.isnan(r):
                return None

            # 1. Señal de momentum
            self._momentum_signal = self._compute_momentum(date, prices)

            # 2. α_target según señal
            alpha_merton = merton_fraction(mu, sigma, r, self.gamma)
            alpha_merton = clip_allocation(alpha_merton)

            if self._momentum_signal > 0:
                # Momentum positivo → posición larga completa
                self._alpha_target = alpha_merton
            else:
                # Momentum negativo → posición defensiva
                self._alpha_target = self.alpha_min

            # 3. Bandas DN alrededor de α_target
            if self._alpha_target > 0.01:
                self._alpha_lower, self._alpha_upper = dn_bands_asymptotic(
                    self._alpha_target, sigma, lambda_L, lambda_M, self.gamma
                )
            else:
                # Posición muy pequeña: banda estrecha
                self._alpha_lower = 0.0
                self._alpha_upper = self.alpha_min * 2

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
                'reason': (f'COMPRA (momentum={self._momentum_signal:.3f}): '
                          f'α={current_alpha:.3f} < lower={self._alpha_lower:.3f}')
            }

        elif current_alpha > self._alpha_upper:
            return {
                'trade': True,
                'target_alpha': self._alpha_upper,
                'reason': (f'VENTA (momentum={self._momentum_signal:.3f}): '
                          f'α={current_alpha:.3f} > upper={self._alpha_upper:.3f}')
            }

        else:
            # Dentro de la banda → no operar (filtro DN dice: espera)
            return None
