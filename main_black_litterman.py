"""
Backtest principal de la estrategia BL-Omega.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Data.data_loader import build_lambda_dict, download_market_data, load_universe_data
from engine_multiasset import MultiAssetBacktestEngine
from strategies.bl_omega_strategy import BLOmegaStrategy
from config import (
    ETF_UNIVERSE,
    XEON_TICKER,
    INITIAL_WEALTH,
    DATA_START,
    BACKTEST_END,
    RISK_FREE_FALLBACK,
    WARMUP_DAYS,
    BL_RECALIB_FREQ,
    BL_OMEGA_WINDOW_FAST,
    OUTPUT_DIR_BL,
)

OUTPUT_DIR = OUTPUT_DIR_BL
_DATA_CACHE = None
_RF_CACHE = None


def _build_risk_free_series(data: dict) -> pd.Series:
    returns = data["returns"]
    xeon_ticker = data["xeon_ticker"]
    if xeon_ticker in returns.columns:
        xeon_daily = returns[xeon_ticker].fillna(0.0)
        rf_series = xeon_daily.ewm(span=21, min_periods=5).mean() * 252
        rf_series = rf_series.clip(lower=0.0).ffill()
        if rf_series.notna().any():
            return rf_series.fillna(float(rf_series.dropna().iloc[0]))
    return pd.Series(float(RISK_FREE_FALLBACK), index=returns.index)


def _load_benchmark_wealth(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        benchmark = download_market_data(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            tickers=["SPY"],
        )
    except Exception:
        return None

    if "SPY" not in benchmark["prices"]:
        return None

    prices = benchmark["prices"]["SPY"].dropna()
    if prices.empty:
        return None

    return float(INITIAL_WEALTH) * (prices / prices.iloc[0])


def _load_shared_data():
    global _DATA_CACHE, _RF_CACHE
    if _DATA_CACHE is None:
        data = load_universe_data(
            start=DATA_START,
            end=BACKTEST_END,
            tickers_riesgo=list(ETF_UNIVERSE.keys()),
            xeon_ticker=XEON_TICKER,
        )
        cache_dir = Path("/Users/pablo/Desktop/Cartera_QUANT/Data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        data["prices"].to_csv(cache_dir / "universe_prices.csv")
        _DATA_CACHE = data
    if _RF_CACHE is None:
        _RF_CACHE = _build_risk_free_series(_DATA_CACHE)
    return _DATA_CACHE, _RF_CACHE


def run_backtest(start=None, end=None, params=None, save_results=True, output_dir=OUTPUT_DIR):
    params = params or {}
    omega_window_fast = int(params.get("omega_window_fast", BL_OMEGA_WINDOW_FAST))
    gamma = float(params.get("gamma", -2))
    recalib_freq = int(params.get("recalib_freq", BL_RECALIB_FREQ))

    data, rf = _load_shared_data()
    prices_df = data["prices"]
    returns_df = data["returns"]
    todos_tickers = data["todos_tickers"]

    start = pd.Timestamp(start) if start is not None else pd.Timestamp(data["backtest_start"])
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(returns_df.index[-1])
    avg_rf = float(rf.loc[start:end].mean()) if not rf.loc[start:end].empty else float(RISK_FREE_FALLBACK)
    lambda_dict = build_lambda_dict(todos_tickers, xeon_ticker=XEON_TICKER)

    strategy = BLOmegaStrategy(
        tickers=todos_tickers,
        xeon_ticker=XEON_TICKER,
        recalib_freq=recalib_freq,
        gamma=gamma,
        categoria_por_ticker=ETF_UNIVERSE,
        omega_window_fast=omega_window_fast,
    )
    engine = MultiAssetBacktestEngine(
        prices_df=prices_df,
        returns_df=returns_df,
        risk_free_series=rf,
        xeon_ticker=XEON_TICKER,
        lambda_per_ticker=lambda_dict,
        start_date=start,
        end_date=end,
        initial_wealth=INITIAL_WEALTH,
        warmup_days=WARMUP_DAYS,
    )
    result = engine.run(strategy)
    history = result["history"]
    wealth = history["wealth"]
    benchmark_wealth = _load_benchmark_wealth(start, end)

    n_years = max(len(wealth) / 252.0, 1 / 252.0)
    cagr = float((wealth.iloc[-1] / wealth.iloc[0]) ** (1 / n_years) - 1) if len(wealth) > 1 else 0.0
    vol = float(wealth.pct_change().dropna().std() * np.sqrt(252)) if len(wealth) > 2 else 0.0
    sharpe = float((cagr - avg_rf) / vol) if vol > 1e-12 else 0.0
    max_dd = float(((wealth - wealth.cummax()) / wealth.cummax()).min()) if len(wealth) > 1 else 0.0

    if save_results:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(out_dir / "wealth_history.csv")
        result["trades"].to_csv(out_dir / "trades.csv", index=False)
        if benchmark_wealth is not None:
            benchmark_wealth.rename("wealth").to_csv(out_dir / "benchmark_wealth.csv")

    return {
        "history": history,
        "trades": result["trades"],
        "wealth": wealth,
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "avg_rf": avg_rf,
        "start": start,
        "end": end,
        "params": {
            "omega_window_fast": omega_window_fast,
            "gamma": gamma,
            "recalib_freq": recalib_freq,
        },
        "prices": prices_df,
        "returns": returns_df,
        "tickers": todos_tickers,
        "xeon_ticker": XEON_TICKER,
        "benchmark_wealth": benchmark_wealth,
    }


def main():
    result = run_backtest(save_results=True)
    print("=" * 60)
    print("BACKTEST SIMPLE — BL-Omega")
    print("=" * 60)
    print(f"Periodo: {result['start'].date()} → {result['end'].date()}")
    print(f"Sharpe: {result['sharpe']:.3f}")
    print(f"CAGR:   {result['cagr']:.2%}")
    print(f"MaxDD:  {result['max_dd']:.2%}")
    print(f"Output: {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    main()
