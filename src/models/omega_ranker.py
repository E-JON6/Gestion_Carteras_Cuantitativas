from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.asset import PriceHistory


@dataclass
class OmegaRanker:
    """Ranks assets by Omega ratio over a price window.

    Omega = sum(positive excess returns) / sum(|negative excess returns|).
    Higher = better risk-adjusted performance.
    """

    threshold: float = 0.0

    def rank(self, history: PriceHistory, tickers: list[str] | None = None) -> dict[str, float]:
        """Compute Omega ratio per ticker, sorted descending."""
        cols = tickers or history.tickers
        cols = [c for c in cols if c in history.tickers]

        excess = history.df[cols].pct_change(fill_method=None).dropna() - self.threshold
        gains = excess.clip(lower=0).sum()
        losses = (-excess).clip(lower=0).sum()

        scores = np.where(losses > 0, gains / losses, np.inf)
        result = dict(zip(cols, scores))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def top(self, history: PriceHistory, n: int, tickers: list[str] | None = None) -> list[str]:
        """Return top N tickers by Omega score."""
        return list(self.rank(history, tickers).keys())[:n]
