from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.asset import PriceHistory
from src.models.black_litterman import BlackLittermanModel
from src.models.omega_ranker import OmegaRanker


class MuEstimator(ABC):
    """Estimator for expected returns (mu).

    All estimators accept a PriceHistory and return a dict
    mapping ticker -> annualised expected return.
    """

    @abstractmethod
    def estimate(self, history: PriceHistory) -> dict[str, float]: ...


# ── Point estimators ────────────────────────────────────────────────────

@dataclass
class HistoricalMean(MuEstimator):
    """Annualised mean of daily log returns."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.min_periods:
            raise ValueError(f"HistoricalMean needs at least {self.min_periods} returns, got {len(ret)}")
        return (ret.mean() * self.trading_days).to_dict()


@dataclass
class EwmaMean(MuEstimator):
    """Exponentially weighted mean of daily log returns."""

    span: int = 60
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.span:
            raise ValueError(f"EwmaMean needs at least {self.span} returns, got {len(ret)}")
        return (ret.ewm(span=self.span).mean().iloc[-1] * self.trading_days).to_dict()


@dataclass
class JamesSteinMean(MuEstimator):
    """James-Stein shrinkage: shrinks individual means toward the grand mean."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.min_periods:
            raise ValueError(f"JamesSteinMean needs at least {self.min_periods} returns, got {len(ret)}")

        mu_sample = ret.mean() * self.trading_days

        n, t = len(mu_sample), len(ret)
        if n < 3:
            return mu_sample.to_dict()

        grand_mean = mu_sample.mean()
        dispersion = np.sum((mu_sample.values - grand_mean) ** 2)
        if dispersion == 0:
            return mu_sample.to_dict()

        alpha = np.clip((n - 2) / (t * dispersion), 0.0, 1.0)
        return ((1 - alpha) * mu_sample + alpha * grand_mean).to_dict()


# ── Series estimators ───────────────────────────────────────────────────

@dataclass
class RollingMu(MuEstimator):
    """Rolling annualised mean return."""

    window: int = 252
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.window:
            raise ValueError(f"RollingMu needs at least {self.window} returns, got {len(ret)}")
        return self.estimate_series(history).iloc[-1].to_dict()

    def estimate_series(self, history: PriceHistory) -> pd.DataFrame:
        return history.log_returns(dropna=False).rolling(self.window).mean() * self.trading_days


@dataclass
class EwmaMu(MuEstimator):
    """EWMA annualised mean return."""

    span: int = 60
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        ret = history.log_returns()
        if len(ret) < self.span:
            raise ValueError(f"EwmaMu needs at least {self.span} returns, got {len(ret)}")
        return self.estimate_series(history).iloc[-1].to_dict()

    def estimate_series(self, history: PriceHistory) -> pd.DataFrame:
        return history.log_returns(dropna=False).ewm(span=self.span).mean() * self.trading_days


# ── Composite estimators ────────────────────────────────────────────────

@dataclass
class BLOmegaMu(MuEstimator):
    """Estimates mu using Black-Litterman with Omega-based views.

    1. Estimates raw mu and cov from history.
    2. Ranks assets by Omega ratio.
    3. Top N assets generate BL views (boosted by view_boost).
    4. Returns BL posterior mu.
    """

    base_mu_estimator: MuEstimator
    cov_estimator: "CovEstimator" = None
    risk_aversion: float = 2.5
    tau: float = 0.025
    n_top_views: int = 3
    view_boost: float = 0.10

    def __post_init__(self):
        if self.cov_estimator is None:
            from src.models.estimators.covariance import SampleCovariance
            self.cov_estimator = SampleCovariance()
        self._omega_ranker = OmegaRanker()

    def estimate(self, history: PriceHistory) -> dict[str, float]:
        mu = self.base_mu_estimator.estimate(history)
        cov = self.cov_estimator.estimate(history)
        bl = BlackLittermanModel(risk_aversion=self.risk_aversion, tau=self.tau)

        tickers = history.tickers
        market_weights = {t: 1.0 / len(tickers) for t in tickers}

        P, Q = self._build_views(history, tickers, mu)
        if len(Q) > 0:
            return bl.posterior_returns(cov, market_weights, P, Q)
        return bl.implied_returns(cov, market_weights)

    def _build_views(
        self, history: PriceHistory, tickers: list[str], mu: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        top = self._omega_ranker.top(history, self.n_top_views, tickers)

        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        P = np.zeros((len(top), n))
        Q = np.zeros(len(top))
        for k, ticker in enumerate(top):
            P[k, ticker_idx[ticker]] = 1.0
            Q[k] = mu[ticker] * (1 + self.view_boost)

        return P, Q
