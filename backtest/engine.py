"""
Motor de backtesting.
Ejecuta una estrategia sobre datos históricos respetando las normas del proyecto.
"""

import numpy as np
import pandas as pd
from backtest.portfolio import Portfolio
from models.param_estimation import (
    ParameterEstimator, compute_log_returns
)
from data.transaction_costs import estimate_transaction_costs
from config import (
    INITIAL_WEALTH, TRADING_DAYS_PER_YEAR,
    DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_M
)


class BacktestEngine:
    """
    Motor de backtesting que ejecuta una estrategia día a día.
    """

    def __init__(self, risky_data, risk_free_series, vix_series=None,
                 start_date=None, end_date=None,
                 lambda_L=None, lambda_M=None,
                 initial_wealth=None,
                 sigma_method='ewma'):
        """
        Args:
            risky_data: pd.DataFrame con OHLCV del activo de riesgo
            risk_free_series: pd.Series con tasa libre de riesgo anual (decimal)
            vix_series: pd.Series con VIX (opcional, para E3)
            start_date: inicio del backtest
            end_date: fin del backtest
            lambda_L: coste proporcional compra
            lambda_M: coste proporcional venta
            initial_wealth: patrimonio inicial
            sigma_method: 'ewma' o 'rolling'
        """
        self.risky_data = risky_data
        self.risk_free_series = risk_free_series
        self.vix_series = vix_series

        # Fechas
        self.start_date = pd.Timestamp(start_date) if start_date else risky_data.index[252]
        self.end_date = pd.Timestamp(end_date) if end_date else risky_data.index[-1]

        # Filtrar datos al rango
        self.prices = risky_data['Close']
        self.returns = compute_log_returns(self.prices)  # Log returns para estimación (μ, σ)
        # Retornos simples para actualización de portfolio (correcto para multiplicar wealth)
        self.simple_returns = self.prices.pct_change().dropna()

        # Costes
        self.lambda_L = lambda_L if lambda_L is not None else DEFAULT_LAMBDA_L
        self.lambda_M = lambda_M if lambda_M is not None else DEFAULT_LAMBDA_M

        # Estimador de parámetros
        self.estimator = ParameterEstimator(
            self.prices, risk_free_series, sigma_method=sigma_method
        )

        self.initial_wealth = initial_wealth or INITIAL_WEALTH

    def run(self, strategy):
        """
        Ejecuta la estrategia sobre los datos históricos.

        Args:
            strategy: objeto Strategy con método decide(date, portfolio, estimator, ...)

        Returns:
            dict con resultados del backtest
        """
        # Fechas de trading dentro del rango (usar simple_returns que está alineado)
        trading_dates = self.simple_returns.index[
            (self.simple_returns.index >= self.start_date) &
            (self.simple_returns.index <= self.end_date)
        ]

        if len(trading_dates) == 0:
            raise ValueError(f"No hay datos entre {self.start_date} y {self.end_date}")

        # Inicializar portfolio con α=0 (todo en bonos)
        # La compra inicial se ejecuta como un trade real con costes
        initial_alpha = strategy.get_initial_alpha(
            self.estimator, trading_dates[0], self.lambda_L, self.lambda_M
        )
        portfolio = Portfolio(
            initial_wealth=self.initial_wealth,
            initial_alpha=0.0  # Empieza todo en bonos
        )

        # Ejecutar compra inicial con costes de transacción
        if initial_alpha > 0:
            portfolio.execute_trade(
                initial_alpha, self.lambda_L, self.lambda_M,
                date=trading_dates[0]
            )

        # Datos extra para estrategias que lo necesiten
        extra_data = {
            'vix': self.vix_series,
            'risky_data': self.risky_data,
            'prices': self.prices,
            'returns': self.returns
        }

        # Registro de bandas DN y parámetros de la estrategia
        bands_history = []

        # Bucle diario
        for i, date in enumerate(trading_dates):
            # 1) Actualizar portfolio con retornos SIMPLES del día
            # NOTA: Portfolio.update_prices necesita retornos simples (r_s = P_t/P_{t-1} - 1)
            # porque hace: value *= (1 + r_s). Los ETFs son de acumulación (total return),
            # por lo que los retornos del precio YA incluyen dividendos reinvertidos.
            daily_return = self.simple_returns.loc[date]
            mu_est, sigma_est, r_est = self.estimator.get_params_at(date)
            rf_daily = r_est / TRADING_DAYS_PER_YEAR if not np.isnan(r_est) else 0.0

            portfolio.update_prices(daily_return, rf_daily, date=date)

            # 2) Pedir decisión a la estrategia
            decision = strategy.decide(
                date=date,
                portfolio=portfolio,
                estimator=self.estimator,
                lambda_L=self.lambda_L,
                lambda_M=self.lambda_M,
                day_index=i,
                extra_data=extra_data
            )

            # 3) Ejecutar trade si la estrategia lo indica
            if decision is not None and decision.get('trade', False):
                target_alpha = decision['target_alpha']
                portfolio.execute_trade(
                    target_alpha, self.lambda_L, self.lambda_M, date=date
                )

            # 4) Registrar estado
            portfolio.record_state(date)

            # 5) Capturar bandas DN y α* de la estrategia (si las tiene)
            alpha_lower = getattr(strategy, '_alpha_lower', None)
            alpha_upper = getattr(strategy, '_alpha_upper', None)
            # Centro de la banda: _alpha_star (E1), _alpha_target (E2), _alpha_adj (E3)
            alpha_center = (
                getattr(strategy, '_alpha_star', None) or
                getattr(strategy, '_alpha_target', None) or
                getattr(strategy, '_alpha_adj', None)
            )
            bands_history.append({
                'date': date,
                'alpha_lower': alpha_lower,
                'alpha_upper': alpha_upper,
                'alpha_center': alpha_center,
            })

        # Resultados
        history = portfolio.get_history_df()
        trades_df = portfolio.get_trades_df()

        # DataFrame de bandas DN
        bands_df = pd.DataFrame(bands_history).set_index('date')

        return {
            'history': history,
            'trades': trades_df,
            'bands': bands_df,
            'portfolio': portfolio,
            'strategy_name': strategy.name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'lambda_L': self.lambda_L,
            'lambda_M': self.lambda_M,
            'initial_wealth': self.initial_wealth
        }
