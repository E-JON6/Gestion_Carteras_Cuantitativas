"""
Clase base para todas las estrategias.
"""

from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Interfaz base para las estrategias de trading.
    Todas las estrategias heredan de esta clase.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        """
        Devuelve la fracción inicial en el activo de riesgo.

        Args:
            estimator: ParameterEstimator
            start_date: fecha de inicio del backtest
            lambda_L, lambda_M: costes de transacción

        Returns:
            float: fracción inicial α
        """
        pass

    @abstractmethod
    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        """
        Toma la decisión de trading para el día actual.

        Args:
            date: fecha actual
            portfolio: Portfolio con estado actual
            estimator: ParameterEstimator para obtener μ, σ, r
            lambda_L, lambda_M: costes de transacción
            day_index: número de día desde el inicio del backtest
            extra_data: dict con datos adicionales (VIX, precios, etc.)

        Returns:
            dict con:
                'trade': bool (True si se debe operar)
                'target_alpha': float (fracción objetivo)
                'reason': str (motivo de la decisión)
            o None si no se opera
        """
        pass
