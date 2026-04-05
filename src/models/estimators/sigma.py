from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.asset import PriceHistory


class SigmaEstimator(ABC):
    """Estimator for volatility (sigma).

    All estimators accept a PriceHistory and return a dict
    mapping ticker -> annualised volatility.
    """

    @abstractmethod
    def estimate(self, history: PriceHistory) -> dict[str, float]: ...


# ── Point estimators ────────────────────────────────────────────────────

@dataclass
class HistoricalVolatility(SigmaEstimator):
    """Annualised standard deviation of daily log returns."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.min_periods:
            raise ValueError(f"HistoricalVolatility needs at least {self.min_periods} returns, got {len(ret)}")
        return (ret.std() * np.sqrt(self.trading_days)).to_dict()


@dataclass
class EwmaVolatility(SigmaEstimator):
    """Exponentially weighted volatility."""

    span: int = 60
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.span:
            raise ValueError(f"EwmaVolatility needs at least {self.span} returns, got {len(ret)}")
        ewma_var = ret.ewm(span=self.span).var().iloc[-1]
        return np.sqrt(ewma_var * self.trading_days).to_dict()


# ── Series estimators ───────────────────────────────────────────────────

@dataclass
class RollingSigma(SigmaEstimator):
    """Rolling annualised volatility."""

    window: int = 252
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.window:
            raise ValueError(f"RollingSigma needs at least {self.window} returns, got {len(ret)}")
        return self.estimate_series(history).iloc[-1].to_dict()

    def estimate_series(self, history: PriceHistory) -> pd.DataFrame:
        return history.log_returns(dropna=False).rolling(self.window).std() * np.sqrt(self.trading_days)


@dataclass
class EwmaSigma(SigmaEstimator):
    """EWMA annualised volatility."""

    span: int = 60
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.span:
            raise ValueError(f"EwmaSigma needs at least {self.span} returns, got {len(ret)}")
        return self.estimate_series(history).iloc[-1].to_dict()

    def estimate_series(self, history: PriceHistory) -> pd.DataFrame:
        return np.sqrt(history.log_returns(dropna=False).ewm(span=self.span).var() * self.trading_days)
