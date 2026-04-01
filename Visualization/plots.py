"""Wrappers de compatibilidad sobre los gráficos BL-Omega."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from plots_bl import (
    plot_drawdown,
    plot_walk_forward_sharpe,
    plot_xeon_weight,
    plot_wealth_curves,
)


def plot_strategy_vs_sp500_v0(wealth_csv="outputs/bl_omega/wealth_history.csv", output_path="results/plot_strategy_vs_sp500.png"):
    history_df = pd.read_csv(wealth_csv, index_col=0, parse_dates=True)
    wealth = pd.to_numeric(history_df["wealth"], errors="coerce").dropna()
    results_dict = {"BL-Omega": {"wealth": wealth}}
    plot_wealth_curves(results_dict, save_path=output_path, title="BL-Omega")
    return str(Path(output_path))


def plot_backtest_v0(backtest_csv="outputs/bl_omega/wealth_history.csv", output_path="results/plot_backtest.png"):
    history_df = pd.read_csv(backtest_csv, index_col=0, parse_dates=True)
    plot_xeon_weight(history_df, save_path=output_path)
    return str(Path(output_path))


def plot_walkforward_v0(walkforward_csv="outputs/walk_forward/resultados_wf.csv", output_path="results/plot_walkforward.png"):
    summary_df = pd.read_csv(walkforward_csv)
    plot_walk_forward_sharpe(summary_df, save_path=output_path)
    return str(Path(output_path))


def plot_drawdown_v0(backtest_csv="outputs/bl_omega/wealth_history.csv", output_path="results/plot_drawdown.png"):
    history_df = pd.read_csv(backtest_csv, index_col=0, parse_dates=True)
    plot_drawdown({"BL-Omega": {"wealth": history_df["wealth"]}}, save_path=output_path)
    return str(Path(output_path))
