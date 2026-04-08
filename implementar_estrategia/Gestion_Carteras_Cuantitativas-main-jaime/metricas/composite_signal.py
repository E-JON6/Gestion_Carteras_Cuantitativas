"""
Senal compuesta con pesos por CATEGORIA + factor drawdown_buy.

Factores:
  1. Momentum (6M excl. ultimo mes): captura tendencias
  2. Reversal (1M negado): mean reversion corto plazo
  3. Trend (precio vs SMA 200): filtra activos en tendencia
  4. Vol penalty (-vol anualizada): penaliza volatilidad excesiva
  5. Drawdown buy (distancia al maximo): COMPRA LA CAIDA

Filosofia contrarian: el factor drawdown_buy da mayor score a activos
que han caido mas desde su maximo reciente. Combinado con momentum y
trend, genera una senal que compra caidas CALIFICADAS (no falling knives).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-10 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def compute_momentum(returns_df: pd.DataFrame, window: int = 126, skip: int = 21) -> pd.Series:
    total_needed = window + skip
    if len(returns_df) < total_needed:
        usable = len(returns_df) - skip
        if usable > 20:
            slice_df = returns_df.iloc[:(-skip if skip > 0 else len(returns_df))]
        else:
            return pd.Series(0.0, index=returns_df.columns, name="momentum")
    else:
        end_idx = -skip if skip > 0 else None
        slice_df = returns_df.iloc[-total_needed:end_idx]
    return ((1 + slice_df).prod() - 1).rename("momentum")


def compute_reversal(returns_df: pd.DataFrame, window: int = 21) -> pd.Series:
    effective = min(window, max(len(returns_df), 5))
    return (-(1 + returns_df.tail(effective)).prod() + 1).rename("reversal")


def compute_trend(prices_df: pd.DataFrame, window: int = 200) -> pd.Series:
    effective = min(window, len(prices_df))
    if effective < 20:
        return pd.Series(0.0, index=prices_df.columns, name="trend")
    sma = prices_df.tail(effective).mean()
    return ((prices_df.iloc[-1] / sma) - 1).rename("trend")


def compute_vol_penalty(returns_df: pd.DataFrame, window: int = 63) -> pd.Series:
    effective = min(window, max(len(returns_df), 20))
    vol = returns_df.tail(effective).std() * np.sqrt(252)
    return (-vol).rename("vol_penalty")


def compute_drawdown_buy(prices_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Buy the dip: mayor score para activos mas lejos de su maximo reciente.

    Si un ETF cayo 30% desde su maximo → score = +0.30
    Si un ETF esta en maximos → score = 0.0
    """
    effective = min(window, len(prices_df))
    if effective < 20:
        return pd.Series(0.0, index=prices_df.columns, name="drawdown_buy")
    rolling_max = prices_df.tail(effective).cummax()
    current = prices_df.iloc[-1]
    drawdown = current / rolling_max.iloc[-1] - 1
    return (-drawdown).rename("drawdown_buy")


def compute_composite_signal(
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    momentum_window: int = 126,
    momentum_skip: int = 21,
    reversal_window: int = 21,
    trend_window: int = 200,
    vol_window: int = 63,
    drawdown_window: int = 252,
    weights: dict | None = None,
    ticker_categories: dict | None = None,
    category_weights: dict | None = None,
) -> pd.Series:
    if weights is None:
        weights = {
            "momentum": 0.30, "reversal": 0.15, "trend": 0.25,
            "vol_penalty": 0.10, "drawdown_buy": 0.20,
        }

    factors_z = {
        "momentum":    _zscore(compute_momentum(returns_df, momentum_window, momentum_skip)),
        "reversal":    _zscore(compute_reversal(returns_df, reversal_window)),
        "trend":       _zscore(compute_trend(prices_df, trend_window)),
        "vol_penalty": _zscore(compute_vol_penalty(returns_df, vol_window)),
        "drawdown_buy": _zscore(compute_drawdown_buy(prices_df, drawdown_window)),
    }

    tickers = returns_df.columns
    composite = pd.Series(0.0, index=tickers)

    for ticker in tickers:
        cat = ticker_categories.get(ticker, "default") if ticker_categories else "default"
        w = category_weights.get(cat, weights) if category_weights else weights

        score = 0.0
        for factor_name, factor_z in factors_z.items():
            val = factor_z.get(ticker, 0.0)
            if pd.isna(val):
                val = 0.0
            score += w.get(factor_name, 0.0) * val
        composite[ticker] = score

    composite = composite.fillna(0.0)
    composite.name = "composite_signal"
    return composite.sort_values(ascending=False)
