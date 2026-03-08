"""
Enfoque 4: Davis-Norman con Vol Targeting Condicional.

Mejora sobre E3 (Vol Targeting): aplica la reducción de exposición
SOLO cuando la tendencia es negativa (precio < SMA200). En mercados
alcistas, mantiene exposición completa incluso con vol moderadamente
elevada → menos trades innecesarios, más retorno capturado.

Clave: E3 reduce exposición SIEMPRE que σ > σ_target, incluyendo en
bull markets donde la vol elevada es transitoria y no precede crash.
E4 discrimina: en bull markets acepta la vol, en bear markets se protege.

Válvula de seguridad: si σ ≥ 2×σ_target, reduce SIEMPRE (flash crash).

Lógica:
  1. σ_target / σ (misma base que E3)
  2. Si tendencia POSITIVA y σ < σ_crisis: α_adj = 1.0 (no reducir)
  3. Si tendencia POSITIVA y σ ≥ σ_crisis: α_adj = σ_target / σ (seguridad)
  4. Si tendencia NEGATIVA: α_adj = σ_target / σ (vol targeting completo)
  5. Bandas DN alrededor de α_adj

Ventajas sobre E3:
  - En bull markets: E4 mantiene α=1.0 mientras E3 reduce → más retorno
  - En bear markets: misma protección que E3
  - Menos trades → menos costes de transacción
"""

import numpy as np
from gestion_cuantitativa.strategies.base import Strategy
from gestion_cuantitativa.models.merton import clip_allocation
from gestion_cuantitativa.models.davis_norman import dn_bands_asymptotic
from gestion_cuantitativa.config import (
    GAMMA, RECALIBRATION_FREQ, SIGMA_TARGET, MAX_LEVERAGE,
    E4_VOL_CRISIS_MULT, E4_TREND_WINDOW, E4_BAND_COST_MULT
)


class E4_DN_Defensive(Strategy):
    """
    Vol Targeting Condicional + Davis-Norman.
    - Tendencia positiva + vol normal: mantener exposición completa
    - Tendencia negativa o vol crisis: aplicar vol targeting (como E3)
    - Bandas DN más anchas → menos trades de ruido
    """

    def __init__(self, gamma=None, recalib_freq=None,
                 sigma_target=None, max_leverage=None,
                 vol_crisis_mult=None,
                 trend_window=None, band_cost_mult=None):
        super().__init__(name="E4: DN + Régimen Defensivo")
        self.gamma = gamma if gamma is not None else GAMMA
        self.recalib_freq = recalib_freq or RECALIBRATION_FREQ
        self.sigma_target = sigma_target or SIGMA_TARGET
        self.max_leverage = max_leverage or MAX_LEVERAGE
        self.vol_crisis_mult = vol_crisis_mult or E4_VOL_CRISIS_MULT
        self.trend_window = trend_window or E4_TREND_WINDOW
        self.band_cost_mult = band_cost_mult or E4_BAND_COST_MULT

        # Umbral de volatilidad crisis
        self.sigma_crisis = self.sigma_target * self.vol_crisis_mult

        # Estado interno
        self._alpha_adj = None
        self._alpha_lower = None
        self._alpha_upper = None
        self._last_recalib = -999
        self._sigma_current = None
        self._trend_positive = True

    def _compute_trend(self, date, prices):
        """
        Tendencia positiva = precio actual ≥ SMA(trend_window).
        """
        idx = prices.index.get_indexer([date], method='ffill')[0]

        if idx < self.trend_window:
            return True  # Sin datos suficientes → asumir alcista

        sma = prices.iloc[idx - self.trend_window + 1:idx + 1].mean()
        current_price = prices.iloc[idx]

        return current_price >= sma

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)

        if np.isnan(sigma) or sigma <= 0:
            return 0.5

        # Misma lógica que E3 para el arranque
        alpha_adj = min(self.sigma_target / sigma, self.max_leverage)
        alpha_adj = clip_allocation(alpha_adj)

        self._alpha_adj = alpha_adj
        self._sigma_current = sigma

        # Bandas DN con coste amplificado → más anchas
        lower, upper = dn_bands_asymptotic(
            alpha_adj, sigma,
            lambda_L * self.band_cost_mult,
            lambda_M * self.band_cost_mult,
            self.gamma
        )
        self._alpha_lower = lower
        self._alpha_upper = upper

        return alpha_adj

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        1. Recalibrar σ y tendencia periódicamente
        2. Decidir α_adj según tendencia y volatilidad
        3. Bandas DN anchas alrededor de α_adj
        4. Bang-bang: solo operar si α sale de la banda
        """
        prices = extra_data.get('prices')
        if prices is None:
            return None

        # --- Recalibrar periódicamente ---
        if day_index - self._last_recalib >= self.recalib_freq:
            mu, sigma, r = estimator.get_params_at(date)

            if np.isnan(sigma) or sigma <= 0:
                return None

            self._sigma_current = sigma

            # 1. Evaluar tendencia
            self._trend_positive = self._compute_trend(date, prices)

            # 2. Calcular α_adj según régimen
            if self._trend_positive and sigma < self.sigma_crisis:
                # BULL MARKET + vol no extrema: mantener exposición completa
                # E3 reduciría aquí → E4 captura más retorno
                alpha_adj = self.max_leverage
            else:
                # BEAR MARKET o VOL CRISIS: vol targeting (como E3)
                alpha_adj = self.sigma_target / sigma
                alpha_adj = min(alpha_adj, self.max_leverage)

            alpha_adj = clip_allocation(alpha_adj)
            self._alpha_adj = alpha_adj

            # 3. Bandas DN más anchas
            if alpha_adj > 0.01:
                self._alpha_lower, self._alpha_upper = dn_bands_asymptotic(
                    alpha_adj, sigma,
                    lambda_L * self.band_cost_mult,
                    lambda_M * self.band_cost_mult,
                    self.gamma
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
                'reason': (f'COMPRA (trend={"+" if self._trend_positive else "-"}): '
                          f'σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} < lower={self._alpha_lower:.3f}')
            }

        elif current_alpha > self._alpha_upper:
            return {
                'trade': True,
                'target_alpha': self._alpha_upper,
                'reason': (f'VENTA (trend={"+" if self._trend_positive else "-"}): '
                          f'σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} > upper={self._alpha_upper:.3f}')
            }

        else:
            return None
