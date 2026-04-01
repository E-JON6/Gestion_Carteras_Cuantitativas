"""
Estrategia stateful BL-Omega + Merton + Davis-Norman.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Models.black_litterman import run_black_litterman
from Models.davis_norman import dn_bands_asymptotic
from Models.merton import compute_optimal_weights

RECALIB_FREQ = 5
GAMMA = -2
XEON_TICKER = "XEON.DE"


class BLOmegaStrategy:
    def __init__(
        self,
        tickers,
        xeon_ticker: str = XEON_TICKER,
        recalib_freq: int = RECALIB_FREQ,
        gamma: float = GAMMA,
        categoria_por_ticker: dict[str, str] | None = None,
        omega_window_fast: int = 42,
    ):
        self.name = "BL-Omega: Omega+BL+Merton+DN+XEON"
        self.tickers = list(tickers)
        self.N = len(self.tickers)
        self.xeon_ticker = xeon_ticker
        self.recalib_freq = recalib_freq
        self.gamma = gamma
        self.categoria_por_ticker = categoria_por_ticker or {}
        self.omega_window_fast = omega_window_fast
        self.omega_window_slow = max(omega_window_fast * 2, 84)
        self.xeon_idx = self.tickers.index(xeon_ticker) if xeon_ticker in self.tickers else -1

        self._target_weights = np.zeros(self.N, dtype=float)
        if self.xeon_idx >= 0:
            self._target_weights[self.xeon_idx] = 1.0
        self._bands_lower = np.zeros(self.N, dtype=float)
        self._bands_upper = np.ones(self.N, dtype=float)
        self._selected_tickers: list[str] = []
        self._frozen_tickers: list[str] = []
        self._last_bl_result = None
        self._last_merton = None

    def _compute_sigma_mercado(self, returns_df: pd.DataFrame) -> float:
        risk_cols = [col for col in returns_df.columns if col != self.xeon_ticker]
        if not risk_cols:
            return 0.15
        vols = returns_df[risk_cols].std().fillna(0.0) * np.sqrt(252)
        return float(vols.mean()) if not vols.empty else 0.15

    def _recalibrate(self, returns_df: pd.DataFrame, risk_free_rate: float, lambda_per_ticker: dict[str, float]):
        if returns_df.empty:
            return
        aligned = returns_df.reindex(columns=self.tickers)
        sigma_mercado = self._compute_sigma_mercado(aligned)

        bl_result = run_black_litterman(
            aligned,
            risk_free_rate,
            xeon_ticker=self.xeon_ticker,
            omega_window_fast=self.omega_window_fast,
            omega_window_slow=self.omega_window_slow,
        )
        merton_result = compute_optimal_weights(
            bl_result,
            risk_free_rate,
            xeon_ticker=self.xeon_ticker,
            sigma_mercado=sigma_mercado,
            categoria_por_ticker=self.categoria_por_ticker,
            current_weights=self._target_weights,
            frozen_tickers=self._frozen_tickers,
            gamma=self.gamma,
        )

        self._last_bl_result = bl_result
        self._last_merton = merton_result

        new_target = np.asarray(merton_result["weights_array"], dtype=float).copy()
        new_selected = list(merton_result["selected_tickers"])
        liquidar = list(merton_result.get("liquidar", []))

        for ticker in self._selected_tickers:
            if ticker not in new_selected and ticker not in self._frozen_tickers:
                self._frozen_tickers.append(ticker)
        for ticker in liquidar:
            if ticker in self._frozen_tickers:
                self._frozen_tickers.remove(ticker)
                if ticker in self.tickers:
                    new_target[self.tickers.index(ticker)] = 0.0

        risk_sum = float(sum(new_target[idx] for idx, ticker in enumerate(self.tickers) if ticker != self.xeon_ticker))
        if self.xeon_idx >= 0:
            new_target[self.xeon_idx] = max(0.0, 1.0 - risk_sum)

        total = float(new_target.sum())
        if total > 1e-12 and abs(total - 1.0) > 1e-8:
            new_target /= total

        self._target_weights = new_target
        self._selected_tickers = new_selected

        sigma = np.asarray(bl_result["Sigma"], dtype=float)
        vol_diag = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
        for idx, ticker in enumerate(self.tickers):
            alpha_star = float(self._target_weights[idx])
            if alpha_star < 0.005:
                self._bands_lower[idx] = 0.0
                self._bands_upper[idx] = 0.02
                continue
            sigma_i = float(vol_diag[idx]) if idx < len(vol_diag) else 0.15
            lambda_i = float(lambda_per_ticker.get(ticker, 0.002))
            lower, upper = dn_bands_asymptotic(alpha_star, sigma_i, lambda_i, lambda_i, self.gamma)
            self._bands_lower[idx] = lower
            self._bands_upper[idx] = upper

    def _check_bands(self, current_weights: np.ndarray) -> tuple[bool, str]:
        for idx in range(self.N):
            w = float(current_weights[idx])
            lower = float(self._bands_lower[idx])
            upper = float(self._bands_upper[idx])
            target = float(self._target_weights[idx])
            if target < 0.005 and w < 0.005:
                continue
            if w < lower or w > upper:
                ticker = self.tickers[idx]
                return True, f"Banda DN cruzada: {ticker} peso={w:.3f} fuera de [{lower:.3f}, {upper:.3f}]"
        return False, ""

    def get_initial_weights(self, returns_df: pd.DataFrame, risk_free_rate: float, lambda_per_ticker: dict[str, float]):
        self._recalibrate(returns_df, risk_free_rate, lambda_per_ticker)
        return self._target_weights.copy()

    def decide(
        self,
        date: pd.Timestamp,
        current_weights: np.ndarray,
        returns_df: pd.DataFrame,
        risk_free_rate: float,
        lambda_per_ticker: dict[str, float],
        day_index: int,
    ):
        if day_index % max(self.recalib_freq, 1) == 0:
            self._recalibrate(returns_df, risk_free_rate, lambda_per_ticker)
        rebalance, reason = self._check_bands(np.asarray(current_weights, dtype=float))
        if rebalance:
            return {
                "rebalance": True,
                "target_weights": self._target_weights.copy(),
                "reason": reason,
            }
        return None
