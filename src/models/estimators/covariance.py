from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.asset import PriceHistory


class CovEstimator(ABC):
    """Estimator for covariance matrix.

    All estimators accept a PriceHistory and return an annualised
    covariance as a DataFrame (tickers as index and columns).
    """

    @abstractmethod
    def estimate(self, history: PriceHistory) -> pd.DataFrame: ...


# ── Shrinkage helper ────────────────────────────────────────────────────

def _shrinkage_alpha(returns: np.ndarray, sample_cov: np.ndarray, target: np.ndarray) -> float:
    """Optimal shrinkage intensity (Ledoit-Wolf style)."""
    t = returns.shape[0]
    centered = returns - returns.mean(axis=0)
    delta_sq_sum = np.sum((sample_cov - target) ** 2)

    if delta_sq_sum == 0:
        return 1.0

    outers = np.einsum("ti,tj->tij", centered, centered)
    pi_sum = np.sum((outers - sample_cov) ** 2)
    pi_hat = pi_sum / t

    return float(np.clip(pi_hat / (t * delta_sq_sum), 0.0, 1.0))


def _to_cov_df(matrix: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """Wrap a numpy covariance matrix as a labelled DataFrame."""
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


# ── Point estimators ────────────────────────────────────────────────────

@dataclass
class SampleCovariance(CovEstimator):
    """Standard sample covariance matrix, annualised."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> pd.DataFrame:
        ret = history.log_returns()
        if len(ret) < self.min_periods:
            raise ValueError(f"SampleCovariance needs at least {self.min_periods} returns, got {len(ret)}")
        return ret.cov() * self.trading_days


@dataclass
class LedoitWolfCovariance(CovEstimator):
    """Ledoit-Wolf shrinkage toward scaled identity matrix."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> pd.DataFrame:
        ret = history.log_returns()
        n = len(history.tickers)
        if len(ret) < max(self.min_periods, n + 1):
            raise ValueError(
                f"LedoitWolfCovariance needs at least {max(self.min_periods, n + 1)} returns "
                f"(T > N={n}), got {len(ret)}"
            )

        ret_arr = ret.values
        sample_cov = np.cov(ret_arr, rowvar=False, bias=False)
        mu_target = np.trace(sample_cov) / n
        target = mu_target * np.eye(n)

        alpha = _shrinkage_alpha(ret_arr, sample_cov, target)
        matrix = (alpha * target + (1 - alpha) * sample_cov) * self.trading_days
        return _to_cov_df(matrix, history.tickers)


@dataclass
class ConstantCorrelationCovariance(CovEstimator):
    """Shrinkage toward constant correlation matrix."""

    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> pd.DataFrame:
        ret = history.log_returns()
        n = len(history.tickers)
        if len(ret) < max(self.min_periods, n + 1):
            raise ValueError(
                f"ConstantCorrelationCovariance needs at least {max(self.min_periods, n + 1)} returns "
                f"(T > N={n}), got {len(ret)}"
            )

        ret_arr = ret.values
        sample_cov = np.cov(ret_arr, rowvar=False, bias=False)
        stds = np.sqrt(np.diag(sample_cov))

        corr = sample_cov / np.outer(stds, stds)
        np.fill_diagonal(corr, 0.0)
        avg_corr = corr.sum() / (n * (n - 1)) if n > 1 else 0.0

        target = avg_corr * np.outer(stds, stds)
        np.fill_diagonal(target, np.diag(sample_cov))

        alpha = _shrinkage_alpha(ret_arr, sample_cov, target)
        matrix = (alpha * target + (1 - alpha) * sample_cov) * self.trading_days
        return _to_cov_df(matrix, history.tickers)


# ── Series / rolling estimator ──────────────────────────────────────────

@dataclass
class RollingCovariance(CovEstimator):
    """Rolling annualised covariance matrix."""

    window: int = 252
    trading_days: int = 252

    def estimate(self, history: PriceHistory) -> pd.DataFrame:
        """Latest rolling covariance matrix."""
        ret = history.log_returns()
        if len(ret) < self.window:
            raise ValueError(f"RollingCovariance needs at least {self.window} returns, got {len(ret)}")
        return ret.iloc[-self.window:].cov() * self.trading_days

    def estimate_at(self, history: PriceHistory, date) -> pd.DataFrame:
        """Covariance matrix at a specific date."""
        ret = history.log_returns()
        end = ret.index.get_loc(date)
        start = max(0, end - self.window + 1)
        return ret.iloc[start:end + 1].cov() * self.trading_days
