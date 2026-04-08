"""Multi-factor composite signal (contrarian flavour).

Combines five z-scored factors per ticker:
  - momentum:     12-1 month return (Jegadeesh-Titman)
  - reversal:     -1M return (mean reversion)
  - trend:        price vs SMA
  - vol_penalty:  -annualised volatility
  - drawdown_buy: -drawdown from rolling peak  (contrarian: "buy the dip")

Returns a dict ticker -> composite score (z-scored cross-section).

Per-ticker factor weights can be customised via ``ticker_categories`` and
``category_weights``: a ticker mapped to a category uses that category's
factor weights; tickers without a category fall back to ``weights``.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.domain.asset import PriceHistory


# Jaime's defaults (config.COMPOSITE_WEIGHTS): drawdown_buy is *off* by design.
_DEFAULT_WEIGHTS = {
    "momentum": 0.40,
    "reversal": 0.10,
    "trend": 0.35,
    "vol_penalty": 0.15,
    "drawdown_buy": 0.00,
}


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-10 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


@dataclass
class CompositeSignal:
    """Cross-sectional composite score from multiple factors."""

    momentum_window: int = 126
    momentum_skip: int = 21
    reversal_window: int = 21
    trend_window: int = 200
    vol_window: int = 63
    drawdown_window: int = 252
    trading_days: int = 252
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    ticker_categories: dict[str, str] | None = None
    category_weights: dict[str, dict[str, float]] | None = None

    def compute(self, history: PriceHistory) -> dict[str, float]:
        prices = history.df
        rets = history.log_returns(dropna=False).dropna(how="all")

        factors = {
            "momentum":     _zscore(self._momentum(rets)),
            "reversal":     _zscore(self._reversal(rets)),
            "trend":        _zscore(self._trend(prices)),
            "vol_penalty":  _zscore(self._vol_penalty(rets)),
            "drawdown_buy": _zscore(self._drawdown_buy(prices)),
        }

        composite: dict[str, float] = {}
        for ticker in prices.columns:
            w = self._weights_for(ticker)
            score = 0.0
            for name, factor in factors.items():
                val = factor.get(ticker, 0.0)
                if pd.isna(val):
                    val = 0.0
                score += w.get(name, 0.0) * val
            composite[ticker] = score

        return composite

    # ── Helpers ─────────────────────────────────────────────────────────

    def _weights_for(self, ticker: str) -> dict[str, float]:
        if not self.ticker_categories or not self.category_weights:
            return self.weights
        cat = self.ticker_categories.get(ticker)
        if cat is None:
            return self.weights
        return self.category_weights.get(cat, self.weights)

    # ── Factor implementations ──────────────────────────────────────────

    def _momentum(self, rets: pd.DataFrame) -> pd.Series:
        n = self.momentum_window + self.momentum_skip
        if len(rets) < n:
            return pd.Series(0.0, index=rets.columns)
        end = -self.momentum_skip if self.momentum_skip > 0 else None
        slice_ = rets.iloc[-n:end]
        return (1 + slice_).prod() - 1

    def _reversal(self, rets: pd.DataFrame) -> pd.Series:
        eff = min(self.reversal_window, max(len(rets), 5))
        return -((1 + rets.tail(eff)).prod() - 1)

    def _trend(self, prices: pd.DataFrame) -> pd.Series:
        eff = min(self.trend_window, len(prices))
        if eff < 20:
            return pd.Series(0.0, index=prices.columns)
        sma = prices.tail(eff).mean()
        return (prices.iloc[-1] / sma) - 1

    def _vol_penalty(self, rets: pd.DataFrame) -> pd.Series:
        eff = min(self.vol_window, max(len(rets), 20))
        vol = rets.tail(eff).std() * np.sqrt(self.trading_days)
        return -vol

    def _drawdown_buy(self, prices: pd.DataFrame) -> pd.Series:
        eff = min(self.drawdown_window, len(prices))
        if eff < 20:
            return pd.Series(0.0, index=prices.columns)
        rolling_max = prices.tail(eff).cummax().iloc[-1]
        dd = prices.iloc[-1] / rolling_max - 1
        return -dd
