"""
main_momentum.py
================
Ejecuta el backtesting de E2_DN_Momentum sobre un ETF elegido,
comparando contra Buy & Hold y Benchmark Merton.

Los parámetros de E2 se leen automáticamente del JSON generado por
Walk_forward_e2.py. Si el JSON no existe, usa los valores por defecto.

Uso:
    python main_momentum.py

Configuración:
    Cambia ACTIVE_TICKER en la sección de configuración de abajo.

Outputs (carpeta resultados/<TICKER>/IS|OOS/):
    - Gráficas de evolución patrimonial
    - CSV de comparación de estrategias
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestion_cuantitativa.config import (
    BACKTEST_START, BACKTEST_END, OOS_START, OOS_END,
    INITIAL_WEALTH, ETF_CONFIGS, TRADING_DAYS_PER_YEAR,
    DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_M
)
from gestion_cuantitativa.data.download import download_all_data
from gestion_cuantitativa.backtest.engine import BacktestEngine
from gestion_cuantitativa.strategies.benchmark_merton import BenchmarkMerton
from gestion_cuantitativa.strategies.e2_dn_momentum import E2_DN_Momentum
from gestion_cuantitativa.strategies.buyhold import BuyAndHold
from gestion_cuantitativa.analysis.metrics import compute_metrics, compare_strategies
from gestion_cuantitativa.analysis.plots import generate_all_plots


# ===========================================================================
# CONFIGURACION — cambiar aqui para procesar otro ETF
# ===========================================================================
ACTIVE_TICKER = "IUSE.L"

# Parametros por defecto (fallback si no existe el JSON del walk-forward)
WF_LOOKBACK  = 252
WF_SKIP      = 42
WF_ALPHA_MIN = 0.15
# ===========================================================================

# Directorio donde Walk_forward_e2.py guarda los JSONs
_WF_JSON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'resultados', 'WALK_FORWARD'
)


def get_wf_params(ticker):
    """
    Lee el JSON del walk-forward y devuelve:
        - params_optimos : dict con los parámetros de la estrategia
        - oos_start      : fecha inicio OOS (corte 70/30 del historial real)
        - oos_end        : fecha fin OOS

    Si no existe el JSON, usa los defaults y OOS_START/OOS_END del config.
    """
    json_path = os.path.join(
        _WF_JSON_DIR,
        f"wf_params_optimos_{ticker.replace('.', '_')}.json"
    )
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        p         = d["params_optimos"]
        oos_start = d.get("oos_start", OOS_START)
        oos_end   = d.get("oos_end",   OOS_END)
        print(f"  Parametros WF leidos de: {os.path.basename(json_path)}")
        print(f"    lookback={p['momentum_lookback']}d  "
              f"skip={p['momentum_skip']}d  "
              f"alpha_min={p['alpha_min']}")
        print(f"    OOS: {oos_start} → {oos_end}  (corte 70/30 del historial real)")
        return p, oos_start, oos_end
    else:
        print(f"  AVISO: JSON no encontrado en {json_path}")
        print(f"  Usando defaults: lookback={WF_LOOKBACK}d / skip={WF_SKIP}d / alpha_min={WF_ALPHA_MIN}")
        print(f"  OOS desde config: {OOS_START} → {OOS_END}")
        return ({"momentum_lookback": WF_LOOKBACK,
                 "momentum_skip":     WF_SKIP,
                 "alpha_min":         WF_ALPHA_MIN},
                OOS_START, OOS_END)


def guardar_tabla_png(df: pd.DataFrame, png_path: str, titulo: str = "") -> None:
    """
    Guarda un DataFrame como imagen PNG con formato de tabla visual.
    Aplica colores semafóricos a Sharpe, CAGR y MaxDD.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()

    # Formatear valores para display
    fmt_map = {
        "Sharpe": "{:.3f}", "sharpe": "{:.3f}",
        "CAGR":   "{:+.2f}%", "cagr": "{:+.2f}%",
        "MaxDD":  "{:.2f}%", "maxdd": "{:.2f}%",
        "Calmar": "{:.3f}", "calmar": "{:.3f}",
        "Vol":    "{:.2f}%", "vol": "{:.2f}%",
        "Costes": "{:.3f}%", "costes": "{:.3f}%",
        "Trades": "{:.0f}", "trades": "{:.0f}",
        "Riqueza": "{:,.0f}", "riqueza": "{:,.0f}",
        "Tiempo": "{:.1f}%", "tiempo": "{:.1f}%",
    }

    def fmt_cell(col, val):
        if pd.isna(val):
            return "—"
        col_lower = col.lower()
        for key, fmt in fmt_map.items():
            if key.lower() in col_lower:
                try:
                    return fmt.format(float(val))
                except Exception:
                    pass
        try:
            return f"{float(val):.4f}"
        except Exception:
            return str(val)

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([fmt_cell(col, row[col]) for col in df.columns])

    # Colores semafóricos por columna
    def cell_color(col, val):
        if pd.isna(val):
            return "#F2F2F2"
        col_lower = col.lower()
        try:
            v = float(val)
        except Exception:
            return "#FFFFFF"
        if "sharpe" in col_lower:
            if v >= 0.6:  return "#C6EFCE"
            if v >= 0.2:  return "#FFEB9C"
            return "#FCE4D6"
        if "cagr" in col_lower:
            if v >= 8:    return "#C6EFCE"
            if v >= 2:    return "#FFEB9C"
            return "#FCE4D6"
        if "maxdd" in col_lower:
            if v >= -10:  return "#C6EFCE"
            if v >= -20:  return "#FFEB9C"
            return "#FCE4D6"
        if "calmar" in col_lower:
            if v >= 0.5:  return "#C6EFCE"
            if v >= 0.2:  return "#FFEB9C"
            return "#FCE4D6"
        return "#EBF3FB"

    cell_colors = [
        [cell_color(col, row[col]) for col in df.columns]
        for _, row in df.iterrows()
    ]

    # Colores de cabecera por grupo IS/OOS/SIM
    col_header_colors = []
    for col in df.columns:
        col_u = col.upper()
        if "_IS" in col_u or col_u.endswith("IS"):
            col_header_colors.append("#1F4E79")
        elif "_OOS" in col_u or col_u.endswith("OOS"):
            col_header_colors.append("#C00000")
        elif "_SIM" in col_u or col_u.endswith("SIM"):
            col_header_colors.append("#375623")
        else:
            col_header_colors.append("#404040")

    # Paleta para filas (estrategias)
    row_palette = ["#2E75B6", "#70AD47", "#FF9900", "#C00000", "#7030A0"]
    row_labels  = list(df.index)

    n_rows = len(cell_text)
    n_cols = len(df.columns)
    fig_w  = max(12, n_cols * 1.6)
    fig_h  = max(3,  n_rows * 0.9 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText    = cell_text,
        rowLabels   = row_labels,
        colLabels   = list(df.columns),
        cellColours = cell_colors,
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 2.2)

    # Estilo cabeceras de columna
    for j, hc in enumerate(col_header_colors):
        cell = tbl[0, j]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold")

    # Estilo cabeceras de fila
    for i in range(n_rows):
        cell = tbl[i + 1, -1]
        cell.set_facecolor(row_palette[i % len(row_palette)])
        cell.set_text_props(color="white", fontweight="bold")

    # Leyenda semafórica
    import matplotlib.patches as mpatches
    leyenda = [
        mpatches.Patch(color="#C6EFCE", label="Bueno"),
        mpatches.Patch(color="#FFEB9C", label="Moderado"),
        mpatches.Patch(color="#FCE4D6", label="Débil"),
        mpatches.Patch(color="#EBF3FB", label="Neutro"),
    ]
    ax.legend(handles=leyenda, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=4,
              fontsize=8, frameon=True, framealpha=0.9)

    if titulo:
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=16)

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG tabla guardado: {os.path.basename(png_path)}")


def compute_effective_backtest_start(risky_data, backtest_start, warmup_days=252):
    desired_start = pd.Timestamp(backtest_start)
    min_start = (risky_data.index[warmup_days]
                 if len(risky_data) > warmup_days else risky_data.index[-1])
    if min_start > desired_start:
        effective_start = min_start.strftime('%Y-%m-%d')
        print(f"  Warm-up insuficiente desde {backtest_start} -> ajustado a {effective_start}")
        return effective_start
    return backtest_start


def run_backtest_for_etf(ticker, etf_config, output_dir,
                          start_date, end_date, period_label, data=None,
                          wf_params_override=None):
    etf_name = etf_config['name']

    print("\n" + "=" * 70)
    print(f"  {period_label}: {etf_name} ({ticker})")
    print(f"  Periodo: {start_date} -> {end_date}")
    print("=" * 70)

    # 1. Datos
    if data is None:
        print(f"\n[1/4] Descargando datos para {ticker}...")
        data = download_all_data(ticker=ticker)
    else:
        print(f"\n[1/4] Reutilizando datos de {ticker}")

    risky_data = data['risky']
    risk_free  = data['risk_free']
    vix        = data['vix']

    # 2. Costes y fechas
    effective_start = compute_effective_backtest_start(risky_data, start_date)
    lambda_L = etf_config.get('lambda_L', DEFAULT_LAMBDA_L)
    lambda_M = etf_config.get('lambda_M', DEFAULT_LAMBDA_M)
    print(f"\n[2/4] Costes: lambda_L={lambda_L:.5f} ({lambda_L*100:.3f}%)  "
          f"lambda_M={lambda_M:.5f} ({lambda_M*100:.3f}%)")

    # 3. Estrategias
    print(f"\n[3/4] Inicializando estrategias...")
    if wf_params_override is not None:
        wf_params = wf_params_override
    else:
        wf_params, _, _ = get_wf_params(ticker)
    strategies = [
        BuyAndHold(etf_name=etf_name),
        BenchmarkMerton(),
        E2_DN_Momentum(
            momentum_lookback = wf_params["momentum_lookback"],
            momentum_skip     = wf_params["momentum_skip"],
            alpha_min         = wf_params["alpha_min"],
        ),
    ]
    for s in strategies:
        print(f"  + {s.name}")

    # 4. Backtests
    avg_rf = risk_free.loc[effective_start:end_date].mean()
    if np.isnan(avg_rf):
        avg_rf = 0.01

    print(f"\n[4/4] Ejecutando backtests ({effective_start} -> {end_date})...")
    print(f"  Patrimonio inicial: {INITIAL_WEALTH:,.0f} EUR  |  rf media: {avg_rf*100:.2f}%")

    results = []
    for strategy in strategies:
        print(f"\n  -> {strategy.name}...", end=" ", flush=True)
        engine = BacktestEngine(
            risky_data       = risky_data,
            risk_free_series = risk_free,
            vix_series       = vix,
            start_date       = effective_start,
            end_date         = end_date,
            lambda_L         = lambda_L,
            lambda_M         = lambda_M,
            initial_wealth   = INITIAL_WEALTH,
            sigma_method     = 'ewma',
        )
        result = engine.run(strategy)
        results.append(result)
        m = compute_metrics(result, avg_risk_free=avg_rf)
        print(f"CAGR={m['CAGR (%)']:.1f}%  "
              f"Sharpe={m['Sharpe Ratio']:.2f}  "
              f"MaxDD={m['Max Drawdown (%)']:.1f}%  "
              f"Trades={m['Num Operaciones']:.0f}" if 'Num Operaciones' in m
              else f"CAGR={m['CAGR (%)']:.1f}%  Sharpe={m['Sharpe Ratio']:.2f}  MaxDD={m['Max Drawdown (%)']:.1f}%")

    m_bh        = compute_metrics(results[0], avg_risk_free=avg_rf)
    m_benchmark = compute_metrics(results[1], avg_risk_free=avg_rf)
    m_e2        = compute_metrics(results[2], avg_risk_free=avg_rf)

    comparison = compare_strategies(results, avg_risk_free=avg_rf)

    # Guardar outputs
    os.makedirs(output_dir, exist_ok=True)
    generate_all_plots(results, save_dir=output_dir, etf_name=etf_name)

    # CSV de métricas comparativas
    csv_path = os.path.join(output_dir, "metricas_comparativas.csv")
    comparison.to_csv(csv_path)
    print(f"\n  CSV guardado: {os.path.basename(csv_path)}")

    # PNG tabla de métricas
    guardar_tabla_png(
        df       = comparison,
        png_path = os.path.join(output_dir, "metricas_comparativas.png"),
        titulo   = f"Métricas comparativas — {etf_name} ({period_label})",
    )

    # Resumen consola
    trades_key = next((k for k in m_e2 if 'perac' in k.lower() or 'rades' in k.lower()), None)

    print(f"\n  {'-'*68}")
    print(f"  RESUMEN {period_label} -- {etf_name}")
    print(f"  {'-'*68}")
    for m, nombre in [
        (m_bh,        f"Buy & Hold ({etf_name})"),
        (m_benchmark, "Benchmark: Merton Puro"),
        (m_e2,        f"E2 Momentum LK={wf_params['momentum_lookback']}d SK={wf_params['momentum_skip']}d AM={wf_params['alpha_min']}"),
    ]:
        t_val = f"{m[trades_key]:.0f}" if trades_key else "-"
        print(f"  {nombre:<45} "
              f"Sharpe={m['Sharpe Ratio']:>6.3f}  "
              f"CAGR={m['CAGR (%)']:>+6.2f}%  "
              f"MaxDD={m['Max Drawdown (%)']:>5.1f}%  "
              f"Trades={t_val}")

    # Ranking
    ranking = sorted(
        [(m_e2, "E2: DN + Momentum"),
         (m_benchmark, "Benchmark: Merton Puro"),
         (m_bh, f"Buy & Hold ({etf_name})")],
        key=lambda x: x[0]['Sharpe Ratio'], reverse=True
    )
    print(f"\n  RANKING {period_label}:")
    for rank, (m, name) in enumerate(ranking, 1):
        print(f"    {rank}. {name:<40} Sharpe={m['Sharpe Ratio']:.3f}  "
              f"CAGR={m['CAGR (%)']:+.2f}%")

    return results, comparison, etf_name, effective_start, data


def main():
    if ACTIVE_TICKER not in ETF_CONFIGS:
        raise ValueError(
            f"ACTIVE_TICKER '{ACTIVE_TICKER}' no esta en ETF_CONFIGS.\n"
            f"Disponibles: {list(ETF_CONFIGS.keys())}"
        )

    etf_config = ETF_CONFIGS[ACTIVE_TICKER]

    # Leer parámetros Y fechas OOS del JSON del walk-forward
    # OOS llega hasta ayer para cubrir todo el historial disponible
    wf_params, wf_oos_start, _ = get_wf_params(ACTIVE_TICKER)
    wf_oos_end = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("  BACKTEST E2 DN+MOMENTUM -- Parametros optimizados por Walk-Forward")
    print("  Master en Finanzas Cuantitativas -- AFI")
    print(f"  ETF: {etf_config['name']} ({ACTIVE_TICKER})")
    print(f"  IS  : {BACKTEST_START} → {wf_oos_start}  (70% historial)")
    print(f"  OOS : {wf_oos_start} → {wf_oos_end}  (30% historial + periodo reciente)")
    print("=" * 70)

    base_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'resultados'
    )

    # FASE 1: IN-SAMPLE
    results_is, _, etf_name, _, data = run_backtest_for_etf(
        ticker=ACTIVE_TICKER, etf_config=etf_config,
        output_dir=os.path.join(base_output_dir, ACTIVE_TICKER.replace('.', '_'), 'IS'),
        start_date=BACKTEST_START, end_date=wf_oos_start,
        period_label="IN-SAMPLE",
        wf_params_override=wf_params,
    )

    # FASE 2: OUT-OF-SAMPLE (corte 70/30 hasta hoy — incluye 2022-2026)
    results_oos, _, _, _, _ = run_backtest_for_etf(
        ticker=ACTIVE_TICKER, etf_config=etf_config,
        output_dir=os.path.join(base_output_dir, ACTIVE_TICKER.replace('.', '_'), 'OOS'),
        start_date=wf_oos_start, end_date=wf_oos_end,
        period_label="OUT-OF-SAMPLE",
        data=data,
        wf_params_override=wf_params,
    )

    # TABLA COMPARATIVA IS vs OOS
    avg_rf = data['risk_free'].mean()
    if np.isnan(avg_rf):
        avg_rf = 0.01

    m_is  = [compute_metrics(r, avg_risk_free=avg_rf) for r in results_is]
    m_oos = [compute_metrics(r, avg_risk_free=avg_rf) for r in results_oos]
    nombres = [f"Buy & Hold ({etf_name})", "Benchmark: Merton Puro", "E2: DN + Momentum"]

    print("\n\n" + "=" * 70)
    print(f"  COMPARACION IS vs OOS -- {etf_name} ({ACTIVE_TICKER})")
    print("=" * 70)
    print(f"  {'Estrategia':<35} {'Sharpe IS':>10} {'Sharpe OOS':>11} "
          f"{'CAGR IS':>9} {'CAGR OOS':>10} {'MaxDD IS':>9} {'MaxDD OOS':>10}")
    print(f"  {'-'*96}")
    for i, nombre in enumerate(nombres):
        print(f"  {nombre:<35} "
              f"{m_is[i]['Sharpe Ratio']:>10.3f} "
              f"{m_oos[i]['Sharpe Ratio']:>11.3f} "
              f"{m_is[i]['CAGR (%)']:>+8.2f}% "
              f"{m_oos[i]['CAGR (%)']:>+9.2f}% "
              f"{m_is[i]['Max Drawdown (%)']:>8.1f}% "
              f"{m_oos[i]['Max Drawdown (%)']:>9.1f}%")

    # CSV + PNG tabla comparativa global
    rows = []
    for i, nombre in enumerate(nombres):
        rows.append({
            "Estrategia": nombre,
            "Sharpe_IS":  round(m_is[i]['Sharpe Ratio'], 4),
            "Sharpe_OOS": round(m_oos[i]['Sharpe Ratio'], 4),
            "CAGR_IS":    round(m_is[i]['CAGR (%)'], 2),
            "CAGR_OOS":   round(m_oos[i]['CAGR (%)'], 2),
            "MaxDD_IS":   round(m_is[i]['Max Drawdown (%)'], 2),
            "MaxDD_OOS":  round(m_oos[i]['Max Drawdown (%)'], 2),
        })
    tabla_global = pd.DataFrame(rows).set_index("Estrategia")
    tabla_path = os.path.join(
        base_output_dir, ACTIVE_TICKER.replace('.', '_'), "comparativa_IS_OOS.csv"
    )
    os.makedirs(os.path.dirname(tabla_path), exist_ok=True)
    tabla_global.to_csv(tabla_path)
    print(f"\n  CSV global guardado: comparativa_IS_OOS.csv")

    guardar_tabla_png(
        df       = tabla_global,
        png_path = os.path.join(
            base_output_dir, ACTIVE_TICKER.replace('.', '_'), "comparativa_IS_OOS.png"
        ),
        titulo   = f"Comparativa IS vs OOS — {etf_name} ({ACTIVE_TICKER})",
    )

    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETADO")
    print("=" * 70)

    return results_is, results_oos


if __name__ == "__main__":
    main()