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


@dataclass
class BlendedEwmaCovariance(CovEstimator):
    """Robust covariance: blended sample correlation + EWMA diagonal vol.

    Pipeline:
      1. Sample covariances on a short and a long window, blended linearly.
      2. Extract correlation from the blend.
      3. Reconstruct with EWMA volatilities on the diagonal:
         Sigma = D_ewma · Corr · D_ewma.
      4. Optional shrinkage toward scaled identity.
      5. PSD projection (clip negative eigenvalues).

    This is the covariance used by the Jaime contrarian pipeline; the EWMA
    diagonal captures volatility clusters while the blended correlation
    keeps structural information.
    """

    short_window: int = 63
    long_window: int = 252
    blend_alpha: float = 0.6
    ewma_lambda: float = 0.94
    shrinkage: float = 0.10
    trading_days: int = 252
    min_periods: int = 20

    def estimate(self, history: PriceHistory) -> pd.DataFrame:
        ret = history.log_returns()
        if len(ret) < self.min_periods:
            raise ValueError(
                f"BlendedEwmaCovariance needs at least {self.min_periods} returns, got {len(ret)}"
            )

        as_short = min(self.short_window, len(ret))
        as_long = min(self.long_window, len(ret))
        cov_short = ret.tail(as_short).cov().values * self.trading_days
        cov_long = ret.tail(as_long).cov().values * self.trading_days
        cov_blend = self.blend_alpha * cov_short + (1.0 - self.blend_alpha) * cov_long

        d = np.sqrt(np.maximum(np.diag(cov_blend), 1e-10))
        d_inv = 1.0 / d
        corr = cov_blend * np.outer(d_inv, d_inv)
        np.fill_diagonal(corr, 1.0)

        if 0 < self.ewma_lambda < 1:
            halflife = np.log(0.5) / np.log(self.ewma_lambda)
        else:
            halflife = 60.0
        ewma_var = ret.ewm(halflife=halflife, min_periods=20).var().iloc[-1].values
        ewma_vol = np.sqrt(np.maximum(ewma_var, 1e-10) * self.trading_days)

        cov = (ewma_vol[:, None] * corr) * ewma_vol[None, :]

        if self.shrinkage > 0:
            n = cov.shape[0]
            mu_target = np.trace(cov) / n
            cov = (1 - self.shrinkage) * cov + self.shrinkage * mu_target * np.eye(n)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov = (cov + cov.T) / 2.0

        return _to_cov_df(cov, history.tickers)


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
