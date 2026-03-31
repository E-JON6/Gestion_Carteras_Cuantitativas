import numpy as np
import pandas as pd

def compute_sharpe_scores(
    returns_df: pd.DataFrame,
    rf_annual: float = 0.02,
    window: int = 42,
    min_periods: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """
    Calcula el Sharpe ratio anualizado de cada ETF usando la ventana reciente.
    returns_df: DataFrame de retornos diarios
    rf_annual: tipo libre de riesgo anual
    """
    recent = returns_df.tail(window).copy()
    rf_daily = rf_annual / annualization

    def sharpe_one_asset(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < min_periods:
            return np.nan

        excess = x - rf_daily
        vol = excess.std(ddof=1)

        if vol == 0:
            if excess.mean() > 0:
                return np.inf
            if excess.mean() < 0:
                return -np.inf
            return 0.0

        sharpe = excess.mean() / vol
        return sharpe * np.sqrt(annualization)

    sharpe = recent.apply(sharpe_one_asset, axis=0)
    sharpe.name = "sharpe_score"
    return sharpe.sort_values(ascending=False)