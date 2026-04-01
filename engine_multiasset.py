"""Motor de backtesting multiactivo para la estrategia BL-Omega."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
INITIAL_WEALTH = 10_000_000.0
MIN_TRADE_EUR = 1e-10


@dataclass(slots=True)
class RebalanceResult:
    cost: float
    n_trades: int
    post_trade_wealth: float


class PortfolioMultiAsset:
    """Portfolio expresado en pesos y riqueza, con costes proporcionales exactos."""

    def __init__(self, todos_tickers, xeon_ticker, initial_wealth: float = INITIAL_WEALTH):
        self.todos_tickers = list(todos_tickers)
        self.xeon_ticker = xeon_ticker
        self.N = len(self.todos_tickers)
        self.xeon_idx = self.todos_tickers.index(xeon_ticker)
        self.wealth = float(initial_wealth)
        self.initial_wealth = float(initial_wealth)
        self.weights = np.zeros(self.N, dtype=float)
        self.weights[self.xeon_idx] = 1.0
        self.history: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.total_costs = 0.0

    def initialize_with_prices(self, prices_today):
        _ = prices_today
        self.weights = np.zeros(self.N, dtype=float)
        self.weights[self.xeon_idx] = 1.0

    def update_prices(self, returns_today, rf_daily: float = 0.0):
        del rf_daily
        if isinstance(returns_today, pd.Series):
            returns_array = returns_today.reindex(self.todos_tickers).fillna(0.0).to_numpy(dtype=float)
        else:
            returns_array = np.nan_to_num(np.asarray(returns_today, dtype=float), nan=0.0)

        current_values = self.wealth * self.weights
        current_values *= (1.0 + returns_array)
        self.wealth = float(current_values.sum())
        if self.wealth > 1e-12:
            self.weights = current_values / self.wealth
        else:
            self.weights = np.zeros(self.N, dtype=float)
            self.weights[self.xeon_idx] = 1.0
            self.wealth = 0.0

    def _solve_post_trade_wealth(self, target_weights, lambda_per_ticker):
        target_weights = np.asarray(target_weights, dtype=float)
        current_values = self.wealth * self.weights
        wealth_before = float(self.wealth)
        lambdas = np.array([float(lambda_per_ticker.get(ticker, 0.002)) for ticker in self.todos_tickers])

        wealth_after = wealth_before
        for _ in range(200):
            trade_amounts = target_weights * wealth_after - current_values
            total_cost = float(np.sum(lambdas * np.abs(trade_amounts)))
            new_wealth_after = wealth_before - total_cost
            if new_wealth_after < 0:
                raise ValueError("Los costes de transacción superan el patrimonio disponible.")
            if abs(new_wealth_after - wealth_after) <= 1e-8 * max(wealth_before, 1.0):
                wealth_after = new_wealth_after
                break
            wealth_after = new_wealth_after
        trade_amounts = target_weights * wealth_after - current_values
        total_cost = float(np.sum(lambdas * np.abs(trade_amounts)))
        return wealth_after, trade_amounts, total_cost

    def execute_rebalance(self, target_weights, lambda_per_ticker, date=None, reason: str | None = None):
        target_weights = np.asarray(target_weights, dtype=float).flatten()
        if target_weights.shape != (self.N,):
            raise ValueError(f"target_weights debe tener shape ({self.N},), no {target_weights.shape}.")

        target_weights = np.nan_to_num(target_weights, nan=0.0)
        target_weights = np.maximum(target_weights, 0.0)
        total = float(target_weights.sum())
        if total <= 1e-12:
            raise ValueError("target_weights no puede sumar 0.")
        if abs(total - 1.0) > 1e-10:
            target_weights = target_weights / total

        wealth_after, trade_amounts, total_cost = self._solve_post_trade_wealth(target_weights, lambda_per_ticker)
        trades_hoy = []
        for idx, ticker in enumerate(self.todos_tickers):
            delta_eur = float(trade_amounts[idx])
            if abs(delta_eur) < MIN_TRADE_EUR:
                continue
            lam = float(lambda_per_ticker.get(ticker, 0.002))
            trades_hoy.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "type": "buy" if delta_eur > 0 else "sell",
                    "delta_eur": delta_eur,
                    "cost": lam * abs(delta_eur),
                    "weight_before": float(self.weights[idx]),
                    "weight_after": float(target_weights[idx]),
                    "reason": reason or "rebalance",
                }
            )

        self.wealth = float(wealth_after)
        self.weights = target_weights.copy()
        self.total_costs += float(total_cost)
        self.trades.extend(trades_hoy)
        return RebalanceResult(cost=float(total_cost), n_trades=len(trades_hoy), post_trade_wealth=float(wealth_after))

    def record_state(self, date, **extra):
        state = {
            "date": date,
            "wealth": float(self.wealth),
            "total_costs": float(self.total_costs),
        }
        for idx, ticker in enumerate(self.todos_tickers):
            state[f"weight_{ticker}"] = float(self.weights[idx])
        state.update(extra)
        self.history.append(state)

    def get_history_df(self):
        if not self.history:
            return pd.DataFrame(columns=["wealth", "total_costs"])
        history = pd.DataFrame(self.history)
        history["date"] = pd.to_datetime(history["date"])
        return history.set_index("date").sort_index()

    def get_trades_df(self):
        if not self.trades:
            return pd.DataFrame(columns=["date", "ticker", "type", "delta_eur", "cost", "weight_before", "weight_after", "reason"])
        trades = pd.DataFrame(self.trades)
        trades["date"] = pd.to_datetime(trades["date"])
        return trades.sort_values(["date", "ticker"]).reset_index(drop=True)


class MultiAssetBacktestEngine:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        risk_free_series: pd.Series,
        xeon_ticker: str,
        lambda_per_ticker: dict[str, float] | None = None,
        start_date=None,
        end_date=None,
        initial_wealth: float | None = None,
        warmup_days: int = 252,
    ):
        self.prices_df = prices_df.copy().sort_index()
        self.returns_df = returns_df.copy().sort_index()
        self.risk_free_series = risk_free_series.copy().sort_index()
        self.xeon_ticker = xeon_ticker
        self.todos_tickers = list(prices_df.columns)
        self.lambda_per_ticker = lambda_per_ticker or {ticker: 0.002 for ticker in self.todos_tickers}
        self.initial_wealth = float(initial_wealth or INITIAL_WEALTH)
        self.warmup_days = int(warmup_days)

        all_dates = self.returns_df.index
        default_start_idx = min(max(self.warmup_days, 0), max(len(all_dates) - 1, 0))
        self.start_date = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(all_dates[default_start_idx])
        self.end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(all_dates[-1])

    def run(self, strategy):
        trading_dates = self.returns_df.index[(self.returns_df.index >= self.start_date) & (self.returns_df.index <= self.end_date)]
        if len(trading_dates) == 0:
            raise ValueError("No hay fechas en el rango del backtest.")

        rf = self.risk_free_series.reindex(self.returns_df.index).ffill().fillna(0.02)
        portfolio = PortfolioMultiAsset(
            todos_tickers=self.todos_tickers,
            xeon_ticker=self.xeon_ticker,
            initial_wealth=self.initial_wealth,
        )
        first_date = pd.Timestamp(trading_dates[0])
        portfolio.initialize_with_prices(self.prices_df.loc[first_date])
        warmup_returns = self.returns_df.loc[:first_date]
        rf_init = float(rf.loc[:first_date].iloc[-1]) if not rf.loc[:first_date].empty else 0.02
        initial_weights = strategy.get_initial_weights(warmup_returns, rf_init, self.lambda_per_ticker)
        init_rebalance = portfolio.execute_rebalance(initial_weights, self.lambda_per_ticker, date=first_date, reason="initial_allocation")
        portfolio.record_state(first_date, rebalance=True, rebalance_reason="initial_allocation", trade_cost=float(init_rebalance.cost), n_trades=int(init_rebalance.n_trades))

        for day_index, date in enumerate(trading_dates[1:], start=1):
            date = pd.Timestamp(date)
            returns_today = self.returns_df.loc[date]
            portfolio.update_prices(returns_today)

            returns_hasta_hoy = self.returns_df.loc[:date]
            rf_hoy = float(rf.loc[date]) if date in rf.index else 0.02
            decision = strategy.decide(
                date=date,
                current_weights=portfolio.weights.copy(),
                returns_df=returns_hasta_hoy,
                risk_free_rate=rf_hoy,
                lambda_per_ticker=self.lambda_per_ticker,
                day_index=day_index,
            )
            rebalance = False
            rebalance_reason = ""
            trade_cost = 0.0
            n_trades = 0
            if decision is not None and decision.get("rebalance", False):
                rebalance = True
                rebalance_reason = str(decision.get("reason", "rebalance"))
                rebalance_result = portfolio.execute_rebalance(decision["target_weights"], self.lambda_per_ticker, date=date, reason=rebalance_reason)
                trade_cost = float(rebalance_result.cost)
                n_trades = int(rebalance_result.n_trades)
            portfolio.record_state(date, rebalance=rebalance, rebalance_reason=rebalance_reason, trade_cost=trade_cost, n_trades=n_trades)

        return {
            "history": portfolio.get_history_df(),
            "trades": portfolio.get_trades_df(),
            "portfolio": portfolio,
            "strategy_name": getattr(strategy, "name", strategy.__class__.__name__),
            "tickers": self.todos_tickers,
            "xeon_ticker": self.xeon_ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_wealth": self.initial_wealth,
        }
