"""
Estrategia Buy & Hold: referencia pasiva.
Invierte el 100% del patrimonio en el activo de riesgo y no rebalancea.
Sirve como referencia para evaluar si las estrategias activas aportan valor.
"""

from strategies.base import Strategy


class BuyAndHold(Strategy):
    """
    Buy & Hold: α = 1 siempre, sin operaciones.
    Equivale a replicar el ETF (total return, acumulativo).
    """

    def __init__(self, etf_name=None):
        name = f"Buy & Hold ({etf_name})" if etf_name else "Buy & Hold"
        super().__init__(name=name)

    def get_initial_alpha(self, estimator, start_date, lambda_L, lambda_M):
        return 1.0  # 100% en activo de riesgo

    def decide(self, date, portfolio, estimator, lambda_L, lambda_M,
               day_index, extra_data):
        return None  # NUNCA opera
