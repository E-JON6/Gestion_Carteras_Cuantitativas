"""
Created on Tue Mar 24 12:26:40 2026

@author: andre
"""

"""
Version 0 de Omega.
Recibe: un DataFrame de retornos.
Devuelve: un score simple por ETF.
"""

import numpy as np
import pandas as pd

def compute_omega_scores(
    returns_df: pd.DataFrame,
    threshold: float = 0.0,
    window: int = 42,
    min_periods: int = 20,
) -> pd.Series:
    """
    Calcula el Omega Ratio de cada ETF sobre la ventana más reciente.
    Devuelve un Series con score por ticker.
    """
    recent = returns_df.tail(window).copy()

    def omega_one_asset(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < min_periods:
            return np.nan

        gains = np.maximum(x - threshold, 0.0).sum()
        losses = np.maximum(threshold - x, 0.0).sum()

        # Casos borde
        if losses == 0 and gains > 0:
            return np.inf
        if losses == 0 and gains == 0:
            return 1.0

        return gains / losses

    omega = recent.apply(omega_one_asset, axis=0)
    omega.name = "omega_score"
    return omega.sort_values(ascending=False)