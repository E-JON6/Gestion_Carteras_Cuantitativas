"""

Visualizacion de resultados del backtest y walkforward.



Incluye: wealth vs benchmarks, drawdown, metricas, decisiones, pesos apilados,

atribucion por ETF (EUR) y paneles mensuales de composicion.

"""



from __future__ import annotations



from pathlib import Path



import matplotlib



matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





def _benchmark_columns(wealth_df: pd.DataFrame) -> list[tuple[str, str, str]]:

    """Lista (columna, etiqueta, color) para benchmarks."""

    out = []

    if "SP500" in wealth_df.columns:

        out.append(("SP500", "SP500 (B&H)", "#FF9800"))

    for c in wealth_df.columns:

        if c.startswith("MSCI_World"):

            out.append((c, c.replace("MSCI_World ", "MSCI World "), "#9C27B0"))

            break

    return out





def plot_wealth_comparison(

    wealth_csv: str = "results/wealth_history.csv",

    output_path: str = "results/plot_wealth_comparison.png",

) -> str:

    """Curvas de wealth: Strategy vs benchmarks."""

    wealth_df = pd.read_csv(wealth_csv, parse_dates=["Date"]).set_index("Date")



    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(wealth_df.index, wealth_df["Strategy"], label="Strategy", linewidth=2, color="#2196F3")



    for col, lab, colr in _benchmark_columns(wealth_df):

        ax.plot(wealth_df.index, wealth_df[col], label=lab, linewidth=2, linestyle="--", color=colr)



    ax.set_title("Estrategia vs benchmarks (Buy & Hold)", fontsize=14, fontweight="bold")

    ax.set_xlabel("Fecha")

    base_val = wealth_df.iloc[0]["Strategy"]

    ax.set_ylabel(f"Valor de la cartera (base {base_val:,.0f})")

    ax.legend(fontsize=11)

    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] Wealth comparison -> {output_path}")

    return output_path





def plot_drawdown(

    wealth_csv: str = "results/wealth_history.csv",

    output_path: str = "results/plot_drawdown.png",

) -> str:

    """Drawdown: Strategy vs benchmarks."""

    wealth_df = pd.read_csv(wealth_csv, parse_dates=["Date"]).set_index("Date")



    series_specs = [("Strategy", "#2196F3", "Strategy")]

    for col, lab, colr in _benchmark_columns(wealth_df):

        series_specs.append((col, colr, lab))



    fig, ax = plt.subplots(figsize=(14, 5))

    for col, color, label in series_specs:

        if col not in wealth_df.columns:

            continue

        series = wealth_df[col]

        cummax = series.cummax()

        dd = (series - cummax) / cummax * 100

        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color, label=label)

        ax.plot(dd.index, dd.values, color=color, linewidth=0.8)



    ax.set_title("Drawdown", fontsize=14, fontweight="bold")

    ax.set_xlabel("Fecha")

    ax.set_ylabel("Drawdown (%)")

    ax.legend(fontsize=11)

    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] Drawdown -> {output_path}")

    return output_path





def plot_metrics_comparison(

    metrics_xlsx: str = "results/Metrics.xlsx",

    output_path: str = "results/plot_metrics_comparison.png",

) -> str | None:

    """Barras: Strategy vs cada columna benchmark en Comparison."""

    metrics_df = pd.read_excel(metrics_xlsx, sheet_name="Comparison")

    metric_col = "Metric"

    strat_col = "Strategy"

    if metric_col not in metrics_df.columns or strat_col not in metrics_df.columns:

        print("[Plot] Metrics.xlsx sin columnas esperadas")

        return None



    bench_cols = [c for c in metrics_df.columns if c not in (metric_col, strat_col)]



    def _parse_cell(s):

        s_str = str(s).strip()

        if s_str in ("-", "nan", "None"):

            return None

        try:

            if "%" in s_str:

                return float(s_str.replace("%", "")) / 100

            return float(s_str)

        except (ValueError, TypeError):

            return None



    rows = []

    for _, row in metrics_df.iterrows():

        mname = row[metric_col]

        sv = _parse_cell(row[strat_col])

        if sv is None:

            continue

        bench_vals = [_parse_cell(row[c]) for c in bench_cols]

        if all(v is None for v in bench_vals):

            continue

        rows.append((mname, sv, bench_vals))



    if not rows:

        print("[Plot] No hay metricas numericas para graficar")

        return None



    names = [r[0] for r in rows]

    n = len(names)

    n_b = len(bench_cols)

    x = np.arange(n)

    width = 0.8 / (1 + n_b)



    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 6))

    ax.bar(x - width * n_b / 2, [r[1] for r in rows], width, label="Strategy", color="#2196F3", alpha=0.88)



    colors = ["#FF9800", "#9C27B0", "#4CAF50", "#795548"]

    for j, bc in enumerate(bench_cols):

        vals = []

        for r in rows:

            v = r[2][j]

            vals.append(v if v is not None else np.nan)

        ax.bar(

            x - width * n_b / 2 + (j + 1) * width,

            vals,

            width,

            label=bc,

            color=colors[j % len(colors)],

            alpha=0.85,

        )



    ax.set_title("Metricas: estrategia vs benchmarks", fontsize=14, fontweight="bold")

    ax.set_xticks(x)

    ax.set_xticklabels(names, rotation=28, ha="right")

    ax.legend(fontsize=9)

    ax.grid(True, alpha=0.3, axis="y")

    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] Metrics comparison -> {output_path}")

    return output_path





def plot_backtest_decisions(

    backtest_csv: str = "results/backtest_resumen.csv",

    output_path: str = "results/plot_decisions.png",

) -> str:

    """Panel de decisiones del backtest: valor, peso XEON, N ETFs."""

    df = pd.read_csv(backtest_csv, parse_dates=["Date"])



    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)



    axes[0].plot(df["Date"], df["Portfolio Value"], color="#2196F3", linewidth=2)

    axes[0].set_ylabel("Portfolio Value")

    axes[0].set_title("Evolucion del Backtest", fontsize=14, fontweight="bold")

    axes[0].grid(True, alpha=0.3)



    axes[1].fill_between(df["Date"], df["Weight XEON"], alpha=0.5, color="#4CAF50")

    axes[1].set_ylabel("Weight XEON.DE")

    axes[1].set_ylim(0, 1)

    axes[1].grid(True, alpha=0.3)



    if "N ETFs" in df.columns:

        axes[2].bar(df["Date"], df["N ETFs"], color="#FF9800", alpha=0.7, width=20)

        axes[2].set_ylabel("N ETFs en cartera")

    axes[2].grid(True, alpha=0.3)

    axes[2].tick_params(axis="x", rotation=45)



    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] Decisions -> {output_path}")

    return output_path





def plot_positions_stacked(

    weights_csv: str = "results/weights_history.csv",

    output_path: str = "results/plot_positions.png",

) -> str:

    """Stacked area chart de la composicion diaria de la cartera."""

    weights_df = pd.read_csv(weights_csv, parse_dates=["Date"]).set_index("Date")



    held_cols = [c for c in weights_df.columns if weights_df[c].max() > 0.01]

    if not held_cols:

        print("[Plot] No hay posiciones para graficar")

        return output_path



    xeon_cols = [c for c in held_cols if "XEON" in c]

    risk_cols = sorted([c for c in held_cols if "XEON" not in c])

    ordered = xeon_cols + risk_cols



    colors_risk = plt.cm.tab20(np.linspace(0, 1, max(len(risk_cols), 1)))

    colors = ["#A5D6A7"] * len(xeon_cols) + list(colors_risk[: len(risk_cols)])



    fig, ax = plt.subplots(figsize=(16, 7))

    ax.stackplot(

        weights_df.index,

        [weights_df[col].values for col in ordered],

        labels=ordered,

        colors=colors,

        alpha=0.85,

    )

    ax.set_title("Composicion diaria de la cartera (pesos)", fontsize=14, fontweight="bold")

    ax.set_xlabel("Fecha")

    ax.set_ylabel("Peso")

    ax.set_ylim(0, 1.05)

    ax.legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.9)

    ax.grid(True, alpha=0.3)

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] Positions stacked -> {output_path}")

    return output_path





def plot_etf_cumulative_contribution(

    attribution_csv: str = "results/attribution_cumulative_eur.csv",

    output_path: str = "results/plot_etf_contribution_cumulative.png",

    top_n: int = 12,

) -> str | None:

    """Lineas: contribucion EUR acumulada de los ETFs con mayor impacto final."""

    path = Path(attribution_csv)

    if not path.exists():

        print("[Plot] Falta attribution_cumulative_eur.csv")

        return None



    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    last = df.iloc[-1].sort_values(ascending=False)

    tickers = list(last.head(top_n).index)



    fig, ax = plt.subplots(figsize=(14, 7))

    for t in tickers:

        ax.plot(df.index, df[t], label=t, linewidth=1.2)



    ax.set_title(

        f"Contribucion EUR acumulada (top {top_n} al final del periodo)",

        fontsize=14,

        fontweight="bold",

    )

    ax.set_xlabel("Fecha")

    ax.set_ylabel("EUR acumulados (aprox.)")

    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3)

    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] ETF cumulative contribution -> {output_path}")

    return output_path





def plot_etf_total_contribution_bar(

    attribution_csv: str = "results/attribution_cumulative_eur.csv",

    output_path: str = "results/plot_etf_contribution_total_bar.png",

    min_abs_eur: float = 50.0,

) -> str | None:

    """Barras horizontales: contribucion EUR total final por ETF."""

    path = Path(attribution_csv)

    if not path.exists():

        return None



    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    last = df.iloc[-1].dropna().sort_values()

    last = last[np.abs(last) >= min_abs_eur]

    if last.empty:

        last = df.iloc[-1].dropna().sort_values()



    fig, ax = plt.subplots(figsize=(10, max(6, len(last) * 0.22)))

    colors = ["#C62828" if v < 0 else "#2E7D32" for v in last.values]

    ax.barh(range(len(last)), last.values, color=colors, alpha=0.85)

    ax.set_yticks(range(len(last)))

    ax.set_yticklabels(last.index, fontsize=9)

    ax.set_xlabel("EUR (acumulado final, aprox.)")

    ax.set_title("Contribucion total por ETF al patrimonio", fontsize=14, fontweight="bold")

    ax.axvline(x=0, color="black", linewidth=0.6)

    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    print(f"[Plot] ETF total bar -> {output_path}")

    return output_path





def plot_individual_etf_cumulative(

    attribution_csv: str = "results/attribution_cumulative_eur.csv",

    output_dir: str = "results/plots_by_etf",

    min_abs_final_eur: float = 25.0,

) -> list[str]:

    """Un PNG por ETF con contribucion acumulada en EUR (solo |final| >= umbral)."""

    path = Path(attribution_csv)

    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():

        return []



    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    last = df.iloc[-1].abs()

    cols = [c for c in df.columns if last.get(c, 0) >= min_abs_final_eur]



    saved = []

    for col in cols:

        fig, ax = plt.subplots(figsize=(11, 4))

        ax.fill_between(df.index, 0, df[col], alpha=0.35, color="#1976D2")

        ax.plot(df.index, df[col], color="#0D47A1", linewidth=1.3)

        ax.set_title(f"{col} — contribucion EUR acumulada", fontsize=12, fontweight="bold")

        ax.set_xlabel("Fecha")

        ax.set_ylabel("EUR")

        ax.grid(True, alpha=0.3)

        ax.axhline(y=0, color="black", linewidth=0.5)

        fig.tight_layout()

        safe = col.replace(".", "_").replace("/", "_")

        fp = out_dir / f"cumulative_{safe}.png"

        fig.savefig(fp, dpi=120)

        plt.close(fig)

        saved.append(str(fp))



    print(f"[Plot] {len(saved)} graficos por ETF en {out_dir}/")

    return saved





def plot_monthly_weight_snapshots(

    weights_csv: str = "results/weights_history.csv",

    decisions_csv: str = "results/backtest_resumen.csv",

    output_dir: str = "results/plots_by_month",

) -> list[str]:

    """

    Un grafico por mes natural: barras de pesos al ultimo dia de datos de ese mes

    (y marca fechas de rebalanceo si coinciden).

    """

    wdf = pd.read_csv(weights_csv, parse_dates=["Date"]).set_index("Date")

    ddf = pd.read_csv(decisions_csv, parse_dates=["Date"]) if Path(decisions_csv).exists() else None



    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []



    months = wdf.index.to_period("M").unique()

    for per in months:

        sub = wdf[wdf.index.to_period("M") == per]

        if sub.empty:

            continue

        last_day = sub.index[-1]

        row = sub.loc[last_day]

        row = row[row > 0.008].sort_values(ascending=True)



        fig, ax = plt.subplots(figsize=(10, max(4, len(row) * 0.25)))

        colors = plt.cm.Set2(np.linspace(0, 1, len(row)))

        ax.barh(range(len(row)), row.values, color=colors)

        ax.set_yticks(range(len(row)))

        ax.set_yticklabels(row.index, fontsize=8)

        ax.set_xlabel("Peso")

        ax.set_title(

            f"Composicion — {per} (cierre {last_day.date()})",

            fontsize=12,

            fontweight="bold",

        )

        ax.set_xlim(0, min(1.05, row.max() * 1.15 if len(row) else 1))

        if ddf is not None:

            rb = ddf[(ddf["Date"].dt.to_period("M") == per) & (ddf["Rebalance"] == True)]

            if not rb.empty:

                ax.text(

                    0.02,

                    0.98,

                    f"Rebalanceos en mes: {len(rb)}",

                    transform=ax.transAxes,

                    va="top",

                    fontsize=9,

                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),

                )

        fig.tight_layout()

        fp = out_dir / f"weights_{per}.png"

        fig.savefig(fp, dpi=120)

        plt.close(fig)

        saved.append(str(fp))



    print(f"[Plot] {len(saved)} graficos mensuales en {out_dir}/")

    return saved





def generate_all_plots(output_dir: str = "results") -> None:

    """Genera todos los graficos del backtest."""

    d = output_dir

    plot_wealth_comparison(f"{d}/wealth_history.csv", f"{d}/plot_wealth_comparison.png")

    plot_drawdown(f"{d}/wealth_history.csv", f"{d}/plot_drawdown.png")

    mpath = Path(d) / "Metrics.xlsx"

    if mpath.exists():

        plot_metrics_comparison(str(mpath), f"{d}/plot_metrics_comparison.png")

    plot_backtest_decisions(f"{d}/backtest_resumen.csv", f"{d}/plot_decisions.png")

    weights_file = Path(d) / "weights_history.csv"

    if weights_file.exists():

        plot_positions_stacked(str(weights_file), f"{d}/plot_positions.png")



    att = Path(d) / "attribution_cumulative_eur.csv"

    if att.exists():

        plot_etf_cumulative_contribution(str(att), f"{d}/plot_etf_contribution_cumulative.png")

        plot_etf_total_contribution_bar(str(att), f"{d}/plot_etf_contribution_total_bar.png")

        plot_individual_etf_cumulative(str(att), f"{d}/plots_by_etf")

    if weights_file.exists():

        plot_monthly_weight_snapshots(

            str(weights_file),

            f"{d}/backtest_resumen.csv",

            f"{d}/plots_by_month",

        )



    print(f"\n[Plot] Graficos generados en {d}/")


