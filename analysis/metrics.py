"""
Métricas de evaluación del backtesting.
Sharpe, drawdown, Calmar, turnover, costes, etc.
"""

import numpy as np
import pandas as pd
from config import TRADING_DAYS_PER_YEAR


def compute_metrics(result, avg_risk_free=None):
    """
    Calcula todas las métricas relevantes de un resultado de backtest.

    Args:
        result: dict devuelto por BacktestEngine.run()
        avg_risk_free: tasa libre de riesgo media del periodo (para Sharpe)

    Returns:
        dict con todas las métricas
    """
    history = result['history']
    trades_df = result['trades']
    portfolio = result['portfolio']

    wealth = history['wealth']
    n_days = len(wealth)
    n_years = n_days / TRADING_DAYS_PER_YEAR

    # --- Retornos ---
    daily_returns = wealth.pct_change().dropna()

    # --- CAGR (bruto) ---
    total_return = wealth.iloc[-1] / wealth.iloc[0]
    cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

    # --- CAGR neto (post-costes) ---
    # Los costes ya están incorporados en el wealth, así que CAGR es neto
    cagr_net = cagr  # Ya es neto

    # --- Volatilidad anualizada ---
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # --- Sharpe Ratio ---
    # Usar la tasa libre de riesgo media del periodo
    rf = avg_risk_free if avg_risk_free is not None else 0.01  # ~1% media ponderada
    excess_return = cagr_net - rf
    sharpe = excess_return / vol if vol > 0 else 0

    # --- Maximum Drawdown ---
    cummax = wealth.cummax()
    drawdown = (wealth - cummax) / cummax
    max_drawdown = drawdown.min()

    # --- Calmar Ratio ---
    calmar = cagr_net / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # --- Número de operaciones ---
    n_trades = len(trades_df) if len(trades_df) > 0 else 0
    n_buys = len(trades_df[trades_df['type'] == 'buy']) if n_trades > 0 else 0
    n_sells = len(trades_df[trades_df['type'] == 'sell']) if n_trades > 0 else 0

    # --- Costes totales ---
    total_costs = portfolio.total_costs
    costs_pct = total_costs / result['initial_wealth'] * 100  # % del patrimonio inicial

    # --- Turnover anual ---
    if n_trades > 0 and n_years > 0:
        total_traded = trades_df['delta'].abs().sum()
        avg_wealth = wealth.mean()
        turnover = (total_traded / avg_wealth) / n_years if avg_wealth > 0 else 0
    else:
        turnover = 0

    # --- Tiempo en mercado ---
    alpha_series = history['alpha']
    time_in_market = (alpha_series > 0.01).mean()

    # --- Alpha medio ---
    avg_alpha = alpha_series.mean()

    return {
        'Estrategia': result['strategy_name'],
        'CAGR (%)': round(cagr_net * 100, 2),
        'Volatilidad (%)': round(vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Max Drawdown (%)': round(max_drawdown * 100, 2),
        'Calmar Ratio': round(calmar, 3),
        'Nº Operaciones': n_trades,
        'Compras': n_buys,
        'Ventas': n_sells,
        'Costes Totales (EUR)': round(total_costs, 2),
        'Costes (% patrimonio)': round(costs_pct, 2),
        'Turnover Anual': round(turnover, 3),
        'Tiempo en Mercado (%)': round(time_in_market * 100, 1),
        'Alpha Medio': round(avg_alpha, 3),
        'Riqueza Final (EUR)': round(wealth.iloc[-1], 2),
        'Retorno Total (%)': round((total_return - 1) * 100, 2),
    }


def compare_strategies(results_list, avg_risk_free=None):
    """
    Genera una tabla comparativa de múltiples estrategias.

    Args:
        results_list: lista de dicts devueltos por BacktestEngine.run()
        avg_risk_free: tasa libre de riesgo media (para Sharpe ratio)

    Returns:
        pd.DataFrame con métricas comparativas
    """
    metrics_list = [compute_metrics(r, avg_risk_free=avg_risk_free) for r in results_list]
    return pd.DataFrame(metrics_list).set_index('Estrategia')
