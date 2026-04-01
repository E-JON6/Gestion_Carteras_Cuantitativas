"""
Gráficos para la estrategia BL-Omega y la validación walk-forward.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORES = {
    "BL-Omega": "#2196F3",
    "Buy & Hold": "#4CAF50",
    "Merton Puro": "#FF9800",
    "XEON.DE": "#9E9E9E",
    "Walk-Forward OOS": "#4CAF50",
}


def _color(name: str) -> str:
    for key, color in COLORES.items():
        if key.lower() in name.lower():
            return color
    return "#9C27B0"


def _save(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_wealth_curves(results_dict, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, data in results_dict.items():
        wealth = pd.Series(data["wealth"]).dropna()
        if wealth.empty:
            continue
        ax.plot(wealth.index, wealth.values / 1e6, label=name, color=_color(name), linewidth=2)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Patrimonio (M EUR)")
    ax.set_title(title or "Evolución del patrimonio")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_drawdown(results_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, data in results_dict.items():
        wealth = pd.Series(data["wealth"]).dropna()
        if wealth.empty:
            continue
        dd = (wealth / wealth.cummax() - 1.0) * 100.0
        color = _color(name)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=name)
        ax.plot(dd.index, dd.values, color=color, linewidth=1)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown histórico")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_weights_over_time(history_df, todos_tickers, xeon_ticker="XEON.DE", save_path=None, top_n=6):
    weight_cols = [f"weight_{ticker}" for ticker in todos_tickers if f"weight_{ticker}" in history_df.columns]
    if not weight_cols:
        return None

    weights_df = history_df[weight_cols].copy()
    weights_df.columns = [column.replace("weight_", "") for column in weight_cols]
    risk_cols = [column for column in weights_df.columns if column != xeon_ticker]
    mean_risk = weights_df[risk_cols].mean().sort_values(ascending=False) if risk_cols else pd.Series(dtype=float)
    top_risk = list(mean_risk.head(top_n).index)
    plot_cols = top_risk + ([xeon_ticker] if xeon_ticker in weights_df.columns else [])
    if not plot_cols:
        return None
    data_plot = weights_df[plot_cols].fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = list(plt.cm.tab10(np.linspace(0, 1, max(len(top_risk), 1))))[: len(top_risk)]
    if xeon_ticker in plot_cols:
        colors = colors + ["#BDBDBD"]
    ax.stackplot(data_plot.index, data_plot.T.values, labels=plot_cols, colors=colors[: len(plot_cols)], alpha=0.8)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Peso en cartera")
    ax.set_title("Composición de la cartera (gris = XEON.DE monetario)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.2, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_xeon_weight(history_df, xeon_ticker="XEON.DE", save_path=None):
    xeon_col = f"weight_{xeon_ticker}"
    if xeon_col not in history_df.columns:
        return None

    xeon_w = pd.to_numeric(history_df[xeon_col], errors="coerce").dropna()
    if xeon_w.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(xeon_w.index, xeon_w.values * 100.0, 0, alpha=0.5, color="#9E9E9E", label="XEON.DE")
    ax.plot(xeon_w.index, xeon_w.values * 100.0, color="#616161", linewidth=1)
    ax.axhline(30, color="#FF9800", linestyle="--", alpha=0.7, label="30% (caution)")
    ax.axhline(60, color="#F44336", linestyle="--", alpha=0.7, label="60% (crisis)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Peso XEON.DE (%)")
    ax.set_title("Peso del activo monetario (XEON.DE)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metrics_table(metrics_df, save_path=None, title=None):
    cols = [
        "CAGR (%)",
        "Volatilidad (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Calmar Ratio",
        "Nº Rebalanceos",
        "XEON.DE medio (%)",
        "Costes (% patrimonio)",
        "Retorno Total (%)",
    ]
    cols_ok = [column for column in cols if column in metrics_df.columns]
    df_plot = metrics_df[cols_ok].copy()

    fig, ax = plt.subplots(figsize=(16, max(3, len(df_plot) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        rowLabels=df_plot.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for col_idx in range(len(df_plot.columns)):
        table[0, col_idx].set_facecolor("#1F4E79")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(df_plot) + 1):
        fill = "#F5F5F5" if row_idx % 2 == 0 else "white"
        for col_idx in range(-1, len(df_plot.columns)):
            if (row_idx, col_idx) in table._cells:
                table[row_idx, col_idx].set_facecolor(fill)
    ax.set_title(title or "Métricas Comparativas", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_walk_forward_sharpe(summary_df: pd.DataFrame, save_path=None, title=None):
    if summary_df.empty:
        return None
    data = summary_df.sort_values("ventana").copy()
    x = data["ventana"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, data["sharpe_is"], 0.35, label="IS (optimización)", color="#2196F3", alpha=0.85)
    ax.bar(x + 0.2, data["sharpe_oos"], 0.35, label="OOS (validación)", color="#4CAF50", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Ventana")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(title or "Walk-Forward: Sharpe IS vs OOS por ventana")
    ax.set_xticks(x)
    ax.set_xticklabels([f"V{int(v)}" for v in x])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_walk_forward_wealth(wealth_oos: pd.Series, benchmark_wealth: pd.Series | None = None, save_path=None, title=None):
    wealth_oos = pd.Series(wealth_oos).dropna()
    if wealth_oos.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(wealth_oos.index, wealth_oos.values / 1e6, label="Walk-Forward OOS", color="#4CAF50", linewidth=2)
    if benchmark_wealth is not None and not pd.Series(benchmark_wealth).dropna().empty:
        benchmark_wealth = pd.Series(benchmark_wealth).dropna()
        ax.plot(
            benchmark_wealth.index,
            benchmark_wealth.values / 1e6,
            label="Backtest simple",
            color="#2196F3",
            linestyle="--",
            alpha=0.75,
            linewidth=1.5,
        )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Patrimonio (M EUR)")
    ax.set_title(title or "Walk-Forward OOS vs Backtest simple")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, save_path)
