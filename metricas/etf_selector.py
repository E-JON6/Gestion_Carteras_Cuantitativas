"""
Selector version 0.
Recibe: un Series con score por ETF.
Devuelve: los 5 ETFs con mejor score.
"""


def select_top_etfs_v0(scores, top_n=5):
    return list(scores.sort_values(ascending=False).head(top_n).index)


if __name__ == "__main__":
    import pandas as pd

    ejemplo = pd.Series(
        [5, 4, 3, 2, 1],
        index=["SPY", "VGK", "EWJ", "EEM", "XEON.DE"],
    )
    print(select_top_etfs_v0(ejemplo))