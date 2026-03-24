"""
Version 0 de Omega.
Recibe: un DataFrame de retornos.
Devuelve: un score simple por ETF.
"""

import pandas as pd


def compute_omega_scores_v0(returns_df):
    tickers = list(returns_df.columns)
    scores = [len(tickers) - i for i in range(len(tickers))]
    return pd.Series(scores, index=tickers, name="omega_score")


if __name__ == "__main__":
    ejemplo = pd.DataFrame(columns=["SPY", "VGK", "EWJ", "EEM", "XEON.DE"])
    print(compute_omega_scores_v0(ejemplo))
