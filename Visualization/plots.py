"""
Plots version 0.
Recibe: CSVs de results.
Devuelve: graficos simples guardados en results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_strategy_vs_sp500_v0(
    wealth_csv="results/wealth_history.csv",
    output_path="results/plot_strategy_vs_sp500.png",
):
    wealth_df = pd.read_csv(wealth_csv)
    wealth_df["Date"] = pd.to_datetime(wealth_df["Date"])

    plt.figure(figsize=(10, 5))
    plt.plot(wealth_df["Date"], wealth_df["Strategy"], label="Strategy")
    plt.plot(wealth_df["Date"], wealth_df["SP500"], label="SP500")
    plt.title("Strategy vs SP500")
    plt.xlabel("Date")
    plt.ylabel("Base 100")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_backtest_v0(
    backtest_csv="results/backtest_resumen.csv",
    output_path="results/plot_backtest.png",
):
    backtest_df = pd.read_csv(backtest_csv)
    backtest_df["Date"] = pd.to_datetime(backtest_df["Date"])

    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df["Date"], backtest_df["Weight XEON"], marker="o")
    plt.title("Peso XEON en Backtest")
    plt.xlabel("Date")
    plt.ylabel("Weight XEON")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_walkforward_v0(
    walkforward_csv="results/walkforward_resumen.csv",
    output_path="results/plot_walkforward.png",
):
    walkforward_df = pd.read_csv(walkforward_csv)
    walkforward_df["Test End"] = pd.to_datetime(walkforward_df["Test End"])

    plt.figure(figsize=(10, 5))
    plt.plot(walkforward_df["Test End"], walkforward_df["Weight XEON"], marker="o")
    plt.title("Peso XEON en Walkforward")
    plt.xlabel("Test End")
    plt.ylabel("Weight XEON")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path