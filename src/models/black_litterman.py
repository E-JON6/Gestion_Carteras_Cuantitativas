from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BlackLittermanModel:
    """Black-Litterman model for combining market equilibrium with investor views.

    Produces posterior expected returns that blend:
    1. Implied equilibrium returns (from market cap weights)
    2. Investor views (absolute or relative) with confidence levels

    Methods accept dicts (mu, weights) and DataFrames (cov).
    Ticker ordering is derived from the covariance DataFrame.
    """

    risk_aversion: float = 2.5
    tau: float = 0.05

    def implied_returns(
        self,
        cov: pd.DataFrame,
        market_weights: dict[str, float],
    ) -> dict[str, float]:
        """Equilibrium implied returns: Pi = risk_aversion * Sigma * w_market."""
        tickers = list(cov.columns)
        cov_arr = cov.values
        w_arr = np.array([market_weights[t] for t in tickers])
        pi = self.risk_aversion * cov_arr @ w_arr
        return dict(zip(tickers, pi))

    def posterior_returns(
        self,
        cov: pd.DataFrame,
        market_weights: dict[str, float],
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Black-Litterman posterior expected returns.

        Args:
            cov: Covariance DataFrame (N, N).
            market_weights: Market cap weights dict.
            P: Pick matrix (K, N).
            Q: View returns vector (K,).
            omega: View uncertainty (K, K). Defaults to tau * P @ Sigma @ P.T.
        """
        tickers = list(cov.columns)
        cov_arr = cov.values
        pi = np.array([self.implied_returns(cov, market_weights)[t] for t in tickers])
        tau_cov = self.tau * cov_arr

        if omega is None:
            omega = np.diag(np.diag(self.tau * P @ cov_arr @ P.T))

        omega_inv_P = np.linalg.solve(omega, P)
        precision = np.linalg.solve(tau_cov, np.eye(len(pi))) + P.T @ omega_inv_P
        mean_part = np.linalg.solve(tau_cov, pi) + P.T @ np.linalg.solve(omega, Q)

        posterior = np.linalg.solve(precision, mean_part)
        return dict(zip(tickers, posterior))

    def posterior_covariance(
        self,
        cov: pd.DataFrame,
        P: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Black-Litterman posterior covariance.

        Args:
            cov: Covariance DataFrame (N, N).
            P: Pick matrix (K, N).
            omega: View uncertainty (K, K).
        """
        tickers = list(cov.columns)
        cov_arr = cov.values
        tau_cov = self.tau * cov_arr

        if omega is None:
            omega = np.diag(np.diag(self.tau * P @ cov_arr @ P.T))

        omega_inv_P = np.linalg.solve(omega, P)
        posterior_precision = np.linalg.solve(tau_cov, np.eye(cov_arr.shape[0])) + P.T @ omega_inv_P

        matrix = np.linalg.inv(posterior_precision) + cov_arr
        return pd.DataFrame(matrix, index=tickers, columns=tickers)
