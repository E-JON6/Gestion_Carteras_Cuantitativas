"""
Estrategia Benchmark: Merton Puro.
Rebalanceo a calendario fijo (semanal) sin considerar costes.
Sirve como referencia para demostrar que ignorar fricciones es subóptimo.
"""

import numpy as np
from gestion_cuantitativa.strategies.base import Strategy
from gestion_cuantitativa.models.merton import merton_fraction, clip_allocation
from gestion_cuantitativa.config import GAMMA, REBALANCE_FREQ_BENCHMARK


class BenchmarkMerton(Strategy):
    """
    Merton puro: rebalanceo semanal al α* óptimo sin bandas.
    """

    def __init__(self, gamma=None, rebalance_freq=None):
        super().__init__(name="Benchmark: Merton Puro")
        self.gamma = gamma if gamma is not None else GAMMA
        self.rebalance_freq = rebalance_freq or REBALANCE_FREQ_BENCHMARK

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        mu, sigma, r = estimator.get_params_at(start_date)
        alpha = merton_fraction(mu, sigma, r, self.gamma)
        return clip_allocation(alpha)

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        Cada `rebalance_freq` días, rebalancear a α* de Merton.
        No usa bandas ni considera costes en la decisión.
        """
        # Solo rebalancear en días programados
        if day_index % self.rebalance_freq != 0:
            return None

        # Calcular α* de Merton
        mu, sigma, r = estimator.get_params_at(date)

        if np.isnan(mu) or np.isnan(sigma) or np.isnan(r):
            return None

        alpha_star = merton_fraction(mu, sigma, r, self.gamma)
        alpha_star = clip_allocation(alpha_star)

        # Siempre rebalancear (no importa el coste)
        current_alpha = portfolio.alpha

        if abs(alpha_star - current_alpha) < 0.001:
            return None  # Diferencia despreciable

        return {
            'trade': True,
            'target_alpha': alpha_star,
            'reason': f'Rebalanceo semanal: α*={alpha_star:.3f}, actual={current_alpha:.3f}'
        }
