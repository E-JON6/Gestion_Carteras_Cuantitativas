"""
Métricas de evaluación para la estrategia BL-Omega.
Compatibles con los outputs pedidos en Info/P5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _coerce_wealth_series(wealth_series: Any) -> pd.Series:
    wealth = pd.Series(wealth_series, copy=True)
    wealth = wealth.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if wealth.empty:
        raise ValueError("La serie de wealth está vacía")
    if not isinstance(wealth.index, pd.DatetimeIndex):
        wealth.index = pd.RangeIndex(len(wealth))
    return wealth.sort_index()


def _year_fraction(index: pd.Index) -> float:
    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return max(len(index) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    delta_days = max((index[-1] - index[0]).days, 1)
    return max(delta_days / 365.25, 1.0 / TRADING_DAYS)


def _annualized_sharpe(daily_returns: pd.Series, avg_risk_free: float) -> float:
    if daily_returns.empty:
        return 0.0
    rf_daily = (1.0 + float(avg_risk_free)) ** (1.0 / TRADING_DAYS) - 1.0
    excess = daily_returns - rf_daily
    sigma = float(excess.std(ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS))


def _annualized_sortino(daily_returns: pd.Series, avg_risk_free: float) -> float:
    if daily_returns.empty:
        return 0.0
    rf_daily = (1.0 + float(avg_risk_free)) ** (1.0 / TRADING_DAYS) - 1.0
    excess = daily_returns - rf_daily
    downside = excess[excess < 0.0]
    sigma_down = float(downside.std(ddof=1)) if not downside.empty else 0.0
    if sigma_down <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma_down * np.sqrt(TRADING_DAYS))


def compute_metrics(
    wealth_series,
    trades_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    strategy_name: str = "Estrategia",
    avg_risk_free: float = 0.025,
    initial_wealth: float = 10_000_000,
    xeon_col: str | None = None,
) -> dict[str, Any]:
    """Calcula métricas comparables para una curva de patrimonio."""
    try:
        wealth = _coerce_wealth_series(wealth_series)
    except ValueError:
        return {"Estrategia": strategy_name, "Error": "Serie wealth vacía"}

    n_days = len(wealth)
    n_years = _year_fraction(wealth.index)
    if n_years < 0.1:
        return {"Estrategia": strategy_name, "Error": "Muy pocos datos"}

    daily_returns = wealth.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)
    cagr = float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / n_years) - 1.0)
    vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if not daily_returns.empty else 0.0
    sharpe = _annualized_sharpe(daily_returns, avg_risk_free)
    sortino = _annualized_sortino(daily_returns, avg_risk_free)

    rolling_max = wealth.cummax()
    drawdown = wealth / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-12 else 0.0

    n_trades = 0
    total_costs = 0.0
    turnover_anual = 0.0
    if trades_df is not None and not trades_df.empty:
        trades_local = trades_df.copy()
        n_trades = int(len(trades_local))
        if "cost" in trades_local.columns:
            total_costs = float(pd.to_numeric(trades_local["cost"], errors="coerce").fillna(0.0).sum())
        elif history_df is not None and "total_costs" in history_df.columns:
            total_costs = float(pd.to_numeric(history_df["total_costs"], errors="coerce").fillna(method="ffill").fillna(0.0).iloc[-1])
        if "delta_eur" in trades_local.columns:
            total_traded = float(pd.to_numeric(trades_local["delta_eur"], errors="coerce").fillna(0.0).abs().sum())
            avg_wealth = float(wealth.mean()) if not wealth.empty else 0.0
            turnover_anual = float((total_traded / avg_wealth) / n_years) if avg_wealth > 1e-12 else 0.0
    elif history_df is not None and "total_costs" in history_df.columns:
        total_costs = float(pd.to_numeric(history_df["total_costs"], errors="coerce").fillna(method="ffill").fillna(0.0).iloc[-1])

    tiempo_en_riesgo = math.nan
    xeon_medio = math.nan
    if history_df is not None and xeon_col and xeon_col in history_df.columns:
        xeon_weights = pd.to_numeric(history_df[xeon_col], errors="coerce").dropna()
        if not xeon_weights.empty:
            xeon_medio = float(xeon_weights.mean())
            tiempo_en_riesgo = float((xeon_weights < 0.20).mean())

    cost_base = max(float(initial_wealth), 1.0)
    return {
        "Estrategia": strategy_name,
        "Inicio": wealth.index[0],
        "Fin": wealth.index[-1],
        "Días": n_days,
        "Años": round(n_years, 3),
        "CAGR (%)": round(cagr * 100.0, 2),
        "Volatilidad (%)": round(vol * 100.0, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_drawdown * 100.0, 2),
        "Calmar Ratio": round(calmar, 3),
        "Nº Rebalanceos": n_trades,
        "Turnover Anual": round(turnover_anual, 3),
        "Costes (% patrimonio)": round(total_costs / cost_base * 100.0, 3),
        "XEON.DE medio (%)": round(xeon_medio * 100.0, 1) if not np.isnan(xeon_medio) else "N/A",
        "Tiempo max riesgo (%)": round(tiempo_en_riesgo * 100.0, 1) if not np.isnan(tiempo_en_riesgo) else "N/A",
        "Retorno Total (%)": round(total_return * 100.0, 1),
        "Riqueza Final (EUR)": round(float(wealth.iloc[-1]), 0),
    }


def compute_buyhold_wealth(prices_df: pd.DataFrame, ticker: str, initial_wealth: float = 10_000_000) -> pd.Series:
    """Buy & Hold en un ETF: comprar al inicio y mantener."""
    if ticker not in prices_df.columns:
        raise KeyError(f"{ticker} no existe en prices_df")
    prices = pd.to_numeric(prices_df[ticker], errors="coerce").dropna()
    if prices.empty:
        raise ValueError(f"No hay precios válidos para {ticker}")
    return float(initial_wealth) * (prices / float(prices.iloc[0]))


def compare_strategies(results_dict: dict[str, dict[str, Any]], avg_risk_free: float = 0.025) -> pd.DataFrame:
    """Construye la tabla comparativa de estrategias."""
    rows: list[dict[str, Any]] = []
    for name, data in results_dict.items():
        rows.append(
            compute_metrics(
                data["wealth"],
                trades_df=data.get("trades"),
                history_df=data.get("history"),
                strategy_name=name,
                avg_risk_free=avg_risk_free,
                initial_wealth=float(data.get("initial_wealth", 10_000_000)),
                xeon_col=data.get("xeon_col"),
            )
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return pd.DataFrame()
    return comparison.set_index("Estrategia")
