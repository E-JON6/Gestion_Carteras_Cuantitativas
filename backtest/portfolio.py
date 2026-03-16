"""
Gestión del portfolio: posiciones, riqueza, ejecución de trades con costes.
"""

import numpy as np
from config import INITIAL_WEALTH, MAX_LEVERAGE, MIN_POSITION


class Portfolio:
    """
    Portfolio de 2 activos: bono (libre de riesgo) + activo de riesgo.
    Gestiona posiciones en EUR, ejecuta compras/ventas con costes.
    """

    def __init__(self, initial_wealth=None, initial_alpha=0.0):
        """
        Args:
            initial_wealth: patrimonio inicial en EUR
            initial_alpha: fracción inicial en activo de riesgo
        """
        self.initial_wealth = initial_wealth or INITIAL_WEALTH

        # Posiciones en EUR
        self.risky_value = self.initial_wealth * initial_alpha     # Y(t) en EUR
        self.bond_value = self.initial_wealth * (1 - initial_alpha)  # X(t) en EUR
        self.wealth = self.initial_wealth

        # Historial
        self.history = []
        self.trades = []
        self.total_costs = 0.0

    @property
    def alpha(self):
        """Fracción actual en activo de riesgo."""
        if self.wealth <= 0:
            return 0.0
        return self.risky_value / self.wealth

    def update_prices(self, risky_return, rf_daily_rate, date=None):
        """
        Actualiza el valor del portfolio con los retornos del día.
        Se ejecuta ANTES de cualquier trade.

        Args:
            risky_return: retorno diario del activo de riesgo (ej. 0.01 = +1%)
            rf_daily_rate: tasa libre de riesgo diaria (anual / 252)
            date: fecha (para historial)
        """
        # Actualizar valores
        self.risky_value *= (1 + risky_return)
        self.bond_value *= (1 + rf_daily_rate)
        self.wealth = self.bond_value + self.risky_value

    def execute_trade(self, target_alpha, lambda_L, lambda_M, date=None):
        """
        Ejecuta un trade para mover la fracción de α_actual a target_alpha.
        Aplica costes proporcionales.

        Args:
            target_alpha: fracción objetivo en activo de riesgo
            lambda_L: coste proporcional de compra
            lambda_M: coste proporcional de venta
            date: fecha del trade

        Returns:
            dict con detalles del trade, o None si no se opera
        """
        current_alpha = self.alpha
        target_alpha = np.clip(target_alpha, 0.0, MAX_LEVERAGE)

        # Calcular el valor objetivo en riesgo
        target_risky = target_alpha * self.wealth
        delta_risky = target_risky - self.risky_value

        if abs(delta_risky) < 1.0:  # Menos de 1 EUR -> no operar
            return None

        if delta_risky > 0:
            # COMPRAR activo de riesgo
            # Coste: pagamos (1 + λ_L) por cada unidad
            cost = lambda_L * delta_risky
            # Sale del bono: delta_risky + cost
            total_debit = delta_risky + cost

            if self.bond_value - total_debit < -self.wealth * (MAX_LEVERAGE - 1):
                # No permitir exceder apalancamiento máximo
                max_debit = self.bond_value + self.wealth * (MAX_LEVERAGE - 1)
                delta_risky = max_debit / (1 + lambda_L)
                cost = lambda_L * delta_risky
                total_debit = delta_risky + cost

            self.risky_value += delta_risky
            self.bond_value -= total_debit
            trade_type = 'buy'

        else:
            # VENDER activo de riesgo
            delta_sell = abs(delta_risky)
            # Coste: recibimos (1 - λ_M) por cada unidad
            cost = lambda_M * delta_sell
            total_credit = delta_sell - cost

            self.risky_value -= delta_sell
            self.bond_value += total_credit
            trade_type = 'sell'

        self.total_costs += cost
        self.wealth = self.bond_value + self.risky_value

        trade_info = {
            'date': date,
            'type': trade_type,
            'delta': delta_risky,
            'cost': cost,
            'alpha_before': current_alpha,
            'alpha_after': self.alpha,
            'wealth_after': self.wealth
        }
        self.trades.append(trade_info)

        return trade_info

    def record_state(self, date):
        """Guarda el estado actual en el historial."""
        self.history.append({
            'date': date,
            'wealth': self.wealth,
            'risky_value': self.risky_value,
            'bond_value': self.bond_value,
            'alpha': self.alpha,
            'total_costs': self.total_costs
        })

    def get_history_df(self):
        """Devuelve el historial como DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history).set_index('date')

    def get_trades_df(self):
        """Devuelve los trades como DataFrame."""
        import pandas as pd
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
