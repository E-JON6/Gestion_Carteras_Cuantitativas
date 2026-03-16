"""
Enfoque 5: Davis-Norman con Drawdown Shield.

Mejora sobre E4 (Régimen Defensivo): en lugar de usar SMA(200) como
indicador de tendencia, usa el drawdown del precio respecto al máximo
de 252 días como señal defensiva. Ventajas:

  1. Reacciona más RÁPIDO que SMA200 a caídas del mercado
     - Drawdown detecta una caída del -8% inmediatamente
     - SMA200 tarda ~40 días en girar tras un shock

  2. Mide RIESGO DIRECTO: el drawdown es exactamente lo que queremos evitar
     - SMA200 es un indicador de tendencia indirecto

  3. HISTÉRESIS para evitar whipsaws:
     - Entrar en modo defensivo cuando DD ≤ -8% (E5_DD_ENTER)
     - Salir de defensivo solo cuando DD > -3% (E5_DD_EXIT)
     - Esto evita oscilaciones rápidas entre modos

Lógica:
  1. Calcular drawdown = (precio - peak_252d) / peak_252d
  2. Modo NORMAL (DD > -8% o histéresis):
     - α_adj = max_leverage (exposición completa)
     - Bandas DN MUY anchas (4× λ) → casi nunca opera
  3. Modo DEFENSIVO (DD ≤ -8%):
     - α_adj = σ_target / σ (vol targeting, como E3)
     - Bandas DN estándar (1× λ) → más reactivo
  4. Válvula de seguridad: si σ ≥ 2×σ_target → defensivo SIEMPRE
  5. Bandas DN alrededor de α_adj, bang-bang execution

Ventajas sobre E4:
  - Reacción más rápida a crashes (días vs semanas)
  - Histéresis reduce whipsaws vs SMA200 crossings
  - En bull markets: misma exposición completa que E4
  - En bear markets: protección más temprana
"""

import numpy as np
from strategies.base import Strategy
from models.merton import clip_allocation
from models.davis_norman import dn_bands_asymptotic
from config import (
    GAMMA, RECALIBRATION_FREQ, SIGMA_TARGET, MAX_LEVERAGE,
    E5_DD_ENTER, E5_DD_EXIT, E5_PEAK_WINDOW,
    E5_BAND_COST_MULT_NORMAL, E5_BAND_COST_MULT_DEFENSIVE,
    E5_VOL_CRISIS_MULT
)


class E5_DN_Drawdown(Strategy):
    """
    Drawdown Shield + Davis-Norman.
    - Drawdown desde peak 252d como señal defensiva
    - Histéresis: -8% entra defensivo, -3% sale
    - Normal: exposición completa, bandas muy anchas
    - Defensivo: vol targeting, bandas estándar
    - Válvula de seguridad: σ ≥ 2×σ_target → defensivo siempre
    """

    def __init__(self, gamma=None, recalib_freq=None,
                 sigma_target=None, max_leverage=None,
                 dd_enter=None, dd_exit=None,
                 peak_window=None,
                 band_cost_mult_normal=None,
                 band_cost_mult_defensive=None,
                 vol_crisis_mult=None):
        super().__init__(name="E5: DN + Drawdown Shield")
        self.gamma = gamma if gamma is not None else GAMMA
        self.recalib_freq = recalib_freq or RECALIBRATION_FREQ
        self.sigma_target = sigma_target or SIGMA_TARGET
        self.max_leverage = max_leverage or MAX_LEVERAGE
        self.dd_enter = dd_enter if dd_enter is not None else E5_DD_ENTER
        self.dd_exit = dd_exit if dd_exit is not None else E5_DD_EXIT
        self.peak_window = peak_window or E5_PEAK_WINDOW
        self.band_cost_mult_normal = band_cost_mult_normal or E5_BAND_COST_MULT_NORMAL
        self.band_cost_mult_defensive = band_cost_mult_defensive or E5_BAND_COST_MULT_DEFENSIVE
        self.vol_crisis_mult = vol_crisis_mult or E5_VOL_CRISIS_MULT

        # Umbral de volatilidad crisis
        self.sigma_crisis = self.sigma_target * self.vol_crisis_mult

        # Estado interno
        self._alpha_adj = None
        self._alpha_lower = None
        self._alpha_upper = None
        self._last_recalib = -999
        self._sigma_current = None
        self._defensive = False  # Estado de histéresis
        self._drawdown = 0.0

    def _compute_drawdown(self, date, prices):
        """
        Calcula el drawdown del precio actual respecto al máximo
        de los últimos peak_window días.

        Returns:
            float: drawdown (negativo, e.g., -0.08 = -8%)
        """
        idx = prices.index.get_indexer([date], method='ffill')[0]

        if idx < 1:
            return 0.0  # Sin datos suficientes → no drawdown

        # Ventana para el peak (máximo de los últimos N días)
        window_start = max(0, idx - self.peak_window + 1)
        peak = prices.iloc[window_start:idx + 1].max()
        current_price = prices.iloc[idx]

        if peak <= 0 or np.isnan(peak):
            return 0.0

        drawdown = (current_price - peak) / peak
        return drawdown

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)

        if np.isnan(sigma) or sigma <= 0:
            return 0.5

        # Arranque: asumir modo normal (sin drawdown)
        alpha_adj = min(self.max_leverage, clip_allocation(self.max_leverage))
        self._alpha_adj = alpha_adj
        self._sigma_current = sigma

        # Bandas DN anchas (modo normal al inicio)
        lower, upper = dn_bands_asymptotic(
            alpha_adj, sigma,
            lambda_L * self.band_cost_mult_normal,
            lambda_M * self.band_cost_mult_normal,
            self.gamma
        )
        self._alpha_lower = lower
        self._alpha_upper = upper

        return alpha_adj

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        1. Recalibrar σ y drawdown periódicamente
        2. Determinar modo (normal/defensivo) con histéresis
        3. Calcular α_adj según modo
        4. Bandas DN (anchas en normal, estándar en defensivo)
        5. Bang-bang: solo operar si α sale de la banda
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

            # 1. Calcular drawdown
            self._drawdown = self._compute_drawdown(date, prices)

            # 2. Determinar modo con HISTÉRESIS
            if not self._defensive:
                # Actualmente en modo NORMAL
                # Entrar en defensivo si:
                #   a) drawdown supera umbral de entrada, o
                #   b) volatilidad en crisis (válvula de seguridad)
                if self._drawdown <= self.dd_enter or sigma >= self.sigma_crisis:
                    self._defensive = True
            else:
                # Actualmente en modo DEFENSIVO
                # Salir de defensivo solo si:
                #   a) drawdown se ha recuperado por encima del umbral de salida, Y
                #   b) volatilidad NO está en crisis
                if self._drawdown > self.dd_exit and sigma < self.sigma_crisis:
                    self._defensive = False

            # 3. Calcular α_adj según modo
            if self._defensive:
                # MODO DEFENSIVO: vol targeting (como E3)
                alpha_adj = self.sigma_target / sigma
                alpha_adj = min(alpha_adj, self.max_leverage)
                band_mult = self.band_cost_mult_defensive
            else:
                # MODO NORMAL: exposición completa
                alpha_adj = self.max_leverage
                band_mult = self.band_cost_mult_normal

            alpha_adj = clip_allocation(alpha_adj)
            self._alpha_adj = alpha_adj

            # 4. Bandas DN
            if alpha_adj > 0.01:
                self._alpha_lower, self._alpha_upper = dn_bands_asymptotic(
                    alpha_adj, sigma,
                    lambda_L * band_mult,
                    lambda_M * band_mult,
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
        mode = "DEF" if self._defensive else "NOR"

        if current_alpha < self._alpha_lower:
            return {
                'trade': True,
                'target_alpha': self._alpha_lower,
                'reason': (f'COMPRA ({mode}, DD={self._drawdown:.1%}): '
                          f'σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} < lower={self._alpha_lower:.3f}')
            }

        elif current_alpha > self._alpha_upper:
            return {
                'trade': True,
                'target_alpha': self._alpha_upper,
                'reason': (f'VENTA ({mode}, DD={self._drawdown:.1%}): '
                          f'σ={self._sigma_current:.3f}, '
                          f'α_adj={self._alpha_adj:.3f}, '
                          f'α={current_alpha:.3f} > upper={self._alpha_upper:.3f}')
            }

        else:
            return None
