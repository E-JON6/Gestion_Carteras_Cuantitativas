"""
run_analysis.py — análisis completo de resultados BL-Omega.
Ejecutar después del backtest simple de main_black_litterman.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from config import INITIAL_WEALTH, OUTPUT_DIR_BL, XEON_TICKER
from main_black_litterman import _load_shared_data
from metrics_bl import compare_strategies, compute_buyhold_wealth
from plots_bl import (
    plot_drawdown,
    plot_metrics_table,
    plot_weights_over_time,
    plot_wealth_curves,
    plot_xeon_weight,
)

OUTPUT_DIR = Path(OUTPUT_DIR_BL)


def _choose_buyhold_ticker(prices_df: pd.DataFrame) -> str:
    preferred = ["IWDA.L", "SPY", "XLK", "IEMA.L"]
    candidates = [ticker for ticker in prices_df.columns if ticker != XEON_TICKER and prices_df[ticker].notna().any()]
    if not candidates:
        raise ValueError("No hay tickers de riesgo para construir Buy & Hold")
    for ticker in preferred:
        if ticker in candidates:
            return ticker
    return candidates[0]


def _load_backtest_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wealth_file = output_dir / "wealth_history.csv"
    trades_file = output_dir / "trades.csv"
    if not wealth_file.exists():
        raise FileNotFoundError(f"{wealth_file} no existe. Ejecutá primero main_black_litterman.py")
    history_df = pd.read_csv(wealth_file, index_col=0, parse_dates=True).sort_index()
    trades_df = pd.read_csv(trades_file) if trades_file.exists() else pd.DataFrame()
    return history_df, trades_df


def main(output_dir: str | Path = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df, trades_df = _load_backtest_outputs(output_dir)
    bl_wealth = pd.to_numeric(history_df["wealth"], errors="coerce").dropna()
    xeon_col = f"weight_{XEON_TICKER}"

    data, rf = _load_shared_data()
    prices_df = data["prices"].reindex(bl_wealth.index).ffill()
    avg_rf = float(rf.reindex(bl_wealth.index).dropna().mean()) if not rf.reindex(bl_wealth.index).dropna().empty else 0.025

    try:
        bh_ticker = _choose_buyhold_ticker(prices_df)
        bh_wealth = compute_buyhold_wealth(prices_df, bh_ticker, INITIAL_WEALTH)
        bh_wealth = bh_wealth.reindex(bl_wealth.index).ffill().dropna()
        bh_wealth = bh_wealth.reindex(bl_wealth.index, method="ffill")
    except Exception:
        bh_ticker = "Referencia"
        bh_wealth = pd.Series(
            INITIAL_WEALTH * np.cumprod(np.repeat(1.0 + 0.08 / 252.0, len(bl_wealth))),
            index=bl_wealth.index,
        )

    results_dict = {
        "BL-Omega (nuestra estrategia)": {
            "wealth": bl_wealth,
            "trades": trades_df,
            "history": history_df,
            "xeon_col": xeon_col,
            "initial_wealth": INITIAL_WEALTH,
        },
        f"Buy & Hold ({bh_ticker})": {
            "wealth": bh_wealth,
            "initial_wealth": INITIAL_WEALTH,
        },
    }
    comparison = compare_strategies(results_dict, avg_risk_free=avg_rf)
    comparison.to_csv(output_dir / "metricas_comparativas.csv")

    plot_wealth_curves(results_dict, save_path=output_dir / "grafico_wealth.png")
    plot_drawdown(results_dict, save_path=output_dir / "grafico_drawdown.png")

    todos_tickers = [column.replace("weight_", "") for column in history_df.columns if column.startswith("weight_")]
    if todos_tickers:
        plot_weights_over_time(history_df, todos_tickers, xeon_ticker=XEON_TICKER, save_path=output_dir / "grafico_pesos.png")
        plot_xeon_weight(history_df, xeon_ticker=XEON_TICKER, save_path=output_dir / "grafico_xeon.png")

    plot_metrics_table(comparison, save_path=output_dir / "metricas_comparativas.png")

    print("=" * 60)
    print("ANÁLISIS DE RESULTADOS — BL-Omega")
    print("=" * 60)
    print(f"Historial: {len(history_df)} filas")
    print(f"Trades:    {len(trades_df)}")
    print(f"Avg RF:    {avg_rf:.2%}")
    print()
    print(comparison.to_string())
    print()
    print(f"Outputs guardados en: {output_dir}")

    return {
        "metrics": comparison,
        "history": history_df,
        "trades": trades_df,
        "buyhold_ticker": bh_ticker,
        "avg_rf": avg_rf,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
