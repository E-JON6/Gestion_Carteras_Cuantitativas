"""
walk_forward_e2.py
==================
Walk-Forward Analysis para optimizar los parámetros de E2_DN_Momentum.
Script completamente independiente — no modifica ningun archivo existente.

Esquema de folds:
    Fold 1: Train 2010-2016 | Val 2017-2018
    Fold 2: Train 2010-2018 | Val 2019-2020
    Fold 3: Train 2010-2020 | Val 2021-2022
    Fold 4: Train 2010-2022 | Val 2023-2024
    OOS:    2025 (intocable, solo se evalua al final con los params ganadores)

Grid de busqueda (18 combinaciones x 4 folds = 72 backtests):
    momentum_lookback : [126, 189, 252]   dias (6m, 9m, 12m)
    momentum_skip     : [21, 42]          dias (1m, 2m)
    alpha_min         : [0.0, 0.05, 0.10] posicion defensiva

Uso:
    # Desde la carpeta raiz del proyecto (donde esta main.py):
    python walk_forward_e2.py --ticker IUSE.L
    python walk_forward_e2.py --ticker IUSE.L --oos
    python walk_forward_e2.py --ticker MSE.PA --oos

Outputs (carpeta resultados/WALK_FORWARD/):
    wf_resultados_TICKER.csv   -> Sharpe de cada combinacion en cada fold
    wf_resumen_TICKER.csv      -> Parametros ganadores por fold + OOS final
    wf_params_optimos.json     -> Parametros finales listos para usar
"""

import os
import sys
import json
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import matplotlib
matplotlib.use("Agg")          # Sin ventana — genera PNGs directamente
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path al proyecto — funciona tanto si se ejecuta desde la raiz del proyecto
# como desde cualquier subdirectorio
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
# Buscar la carpeta gestion_cuantitativa subiendo niveles si hace falta
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent]:
    if (candidate / "gestion_cuantitativa").exists():
        PROJECT_ROOT = candidate
        break
sys.path.insert(0, str(PROJECT_ROOT))

from gestion_cuantitativa.backtest.engine            import BacktestEngine
from gestion_cuantitativa.strategies.e2_dn_momentum  import E2_DN_Momentum
from gestion_cuantitativa.analysis.metrics           import compute_metrics
from gestion_cuantitativa.config import (
    INITIAL_WEALTH, DATA_START, TRADING_DAYS_PER_YEAR,
    ETF_CONFIGS, DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_M
)

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"

# Folds por ETF — adaptados al historial disponible de cada uno
# ETFs con historia larga (2010-2011): 4 folds de ~2 años
# IUSN.DE (desde 2019): 2 folds reducidos
FOLDS_STANDARD = [
    {"label": "Fold 1", "train_end": "2016-12-31",
     "val_start": "2017-01-01", "val_end": "2018-12-31"},
    {"label": "Fold 2", "train_end": "2018-12-31",
     "val_start": "2019-01-01", "val_end": "2020-12-31"},
    {"label": "Fold 3", "train_end": "2020-12-31",
     "val_start": "2021-01-01", "val_end": "2022-12-31"},
    {"label": "Fold 4", "train_end": "2022-12-31",
     "val_start": "2023-01-01", "val_end": "2024-12-31"},
]

FOLDS_SHORT = [
    {"label": "Fold 1", "train_end": "2021-12-31",
     "val_start": "2022-01-01", "val_end": "2023-06-30"},
    {"label": "Fold 2", "train_end": "2023-06-30",
     "val_start": "2023-07-01", "val_end": "2024-12-31"},
]

# Tickers con historial corto que usan folds reducidos
TICKERS_SHORT_HISTORY = {"IUSN.DE"}

def get_folds(ticker: str) -> list:
    """Devuelve los folds adecuados segun el historial del ticker."""
    return FOLDS_SHORT if ticker in TICKERS_SHORT_HISTORY else FOLDS_STANDARD

PARAM_GRID = {
    "momentum_lookback": [126, 189, 252],
    "momentum_skip":     [21, 42],
    "alpha_min":         [0.0, 0.05, 0.10],
}

# Los 4 ETFs del proyecto
ALL_TICKERS = ["MSE.PA", "IUSE.L", "IEMA.L", "IUSN.DE"]

OUTPUT_DIR = PROJECT_ROOT / "resultados" / "WALK_FORWARD"


# ---------------------------------------------------------------------------
# DESCARGA DE DATOS
# ---------------------------------------------------------------------------

def download_data(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Descarga precios del ETF y tasa libre de riesgo.
    Reutiliza exactamente la misma logica que main.py para consistencia.
    """
    print(f"\n  Descargando precios {ticker} (2008-2025)...")
    raw = yf.download(ticker, start=DATA_START, end="2025-12-31", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel("Ticker")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    data = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(data) < 252:
        raise ValueError(f"Datos insuficientes para {ticker}: {len(data)} filas")

    print(f"  Descargando tasa libre de riesgo (ECBDFR)...")
    try:
        rf_raw = pdr.get_data_fred("ECBDFR", start=DATA_START, end="2025-12-31")
        rf = rf_raw["ECBDFR"].ffill().bfill() / 100.0
        rf.index = pd.to_datetime(rf.index).tz_localize(None)
    except Exception:
        print("  AVISO: No se pudo descargar ECBDFR, usando rf=2% fijo")
        rf = pd.Series(0.02, index=data.index)

    print(f"  Datos: {data.index[0].date()} → {data.index[-1].date()} "
          f"({len(data)} dias habiles)")
    return data, rf


def get_lambda(ticker: str) -> float:
    """Obtiene lambda del ETF desde config, igual que main.py."""
    cfg = ETF_CONFIGS.get(ticker, {})
    return cfg.get("lambda_L", DEFAULT_LAMBDA_L)


# ---------------------------------------------------------------------------
# EJECUTAR UN BACKTEST CON UNA COMBINACION DE PARAMETROS
# ---------------------------------------------------------------------------

def run_one(data: pd.DataFrame, rf: pd.Series,
            start: str, end: str,
            lk: int, sk: int, am: float,
            lam: float) -> dict:
    """
    Ejecuta BacktestEngine con E2_DN_Momentum para una combinacion de params.
    Usa compute_metrics() del proyecto — misma logica que main.py.
    Devuelve todas las metricas clave: Sharpe, CAGR, Vol, MaxDD, Calmar,
    Costes%, Trades, Alpha Medio, Tiempo en Mercado.
    """
    t_start = pd.Timestamp(start)
    idx     = data.index.searchsorted(t_start)
    if idx < 252:
        t_start = data.index[252]

    try:
        engine = BacktestEngine(
            risky_data       = data,
            risk_free_series = rf,
            start_date       = t_start,
            end_date         = end,
            lambda_L         = lam,
            lambda_M         = lam,
            initial_wealth   = INITIAL_WEALTH,
            sigma_method     = "ewma",
        )
        strategy = E2_DN_Momentum(
            momentum_lookback = lk,
            momentum_skip     = sk,
            alpha_min         = am,
        )
        result = engine.run(strategy)

        # Tasa libre de riesgo media del periodo
        mask    = (rf.index >= t_start) & (rf.index <= pd.Timestamp(end))
        rf_mean = float(rf[mask].mean()) if mask.sum() > 0 else 0.02

        # Usar compute_metrics del proyecto — idéntico a main.py
        m = compute_metrics(result, avg_risk_free=rf_mean)

        return {
            "sharpe":      m["Sharpe Ratio"],
            "cagr":        m["CAGR (%)"] / 100,
            "vol":         m["Volatilidad (%)"] / 100,
            "maxdd":       m["Max Drawdown (%)"] / 100,
            "calmar":      m["Calmar Ratio"],
            "costes_pct":  m["Costes (% patrimonio)"],
            "trades":      m["Nº Operaciones"],
            "alpha_medio": m["Alpha Medio"],
            "t_mercado":   m["Tiempo en Mercado (%)"],
            "riqueza":     m["Riqueza Final (EUR)"],
            "ok":          True,
        }
    except Exception as e:
        return {
            "sharpe": -99.0, "cagr": 0.0, "vol": 0.0,
            "maxdd": 0.0, "calmar": 0.0, "costes_pct": 0.0,
            "trades": 0, "alpha_medio": 0.0, "t_mercado": 0.0,
            "riqueza": 0.0, "ok": False, "error": str(e),
        }


# ---------------------------------------------------------------------------
# WALK-FORWARD PRINCIPAL
# ---------------------------------------------------------------------------

def run_walk_forward(ticker: str, run_oos: bool = False) -> dict:
    """
    Ejecuta el Walk-Forward completo para el ticker dado.

    Returns:
        dict con params_optimos y dataframes de resultados
    """
    print("\n" + "=" * 65)
    print(f"  WALK-FORWARD E2 MOMENTUM — {ticker}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 65)

    # 1. Datos
    data, rf = download_data(ticker)
    lam      = get_lambda(ticker)
    folds    = get_folds(ticker)
    print(f"  Lambda: {lam:.6f} ({lam*100:.4f}%)")
    if ticker in TICKERS_SHORT_HISTORY:
        print(f"  AVISO: {ticker} usa {len(folds)} folds reducidos (historial corto desde 2019)")

    # 2. Grid de combinaciones
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    print(f"\n  Grid: {len(combos)} combinaciones × {len(folds)} folds "
          f"= {len(combos)*len(folds)} backtests\n")

    # 3. Ejecutar cada fold
    all_rows   = []
    fold_best  = []

    for fold in folds:
        print(f"  {fold['label']}: train → {fold['train_end']} | "
              f"val {fold['val_start']} → {fold['val_end']}")

        fold_results = []
        for combo in combos:
            params = dict(zip(keys, combo))
            res    = run_one(
                data, rf,
                start = fold["val_start"],
                end   = fold["val_end"],
                lk    = params["momentum_lookback"],
                sk    = params["momentum_skip"],
                am    = params["alpha_min"],
                lam   = lam,
            )
            row = {
                "fold":              fold["label"],
                "val_start":         fold["val_start"],
                "val_end":           fold["val_end"],
                "momentum_lookback": params["momentum_lookback"],
                "momentum_skip":     params["momentum_skip"],
                "alpha_min":         params["alpha_min"],
                "sharpe":            round(res["sharpe"],     4),
                "cagr":              round(res["cagr"],       4),
                "vol":               round(res["vol"],        4),
                "maxdd":             round(res["maxdd"],      4),
                "calmar":            round(res["calmar"],     4),
                "costes_pct":        round(res["costes_pct"], 4),
                "trades":            res["trades"],
            }
            all_rows.append(row)
            fold_results.append((res["sharpe"], params))

        # Mejor combo en este fold (por Sharpe)
        best_sharpe, best_params = max(fold_results, key=lambda x: x[0])
        fold_best.append({
            "fold":       fold["label"],
            "val_start":  fold["val_start"],
            "val_end":    fold["val_end"],
            "best_sharpe": round(best_sharpe, 4),
            **best_params,
        })
        print(f"    → Mejor: lookback={best_params['momentum_lookback']}d  "
              f"skip={best_params['momentum_skip']}d  "
              f"alpha_min={best_params['alpha_min']}  "
              f"Sharpe={best_sharpe:.3f}")

    # 4. Parametros optimos: score combinado sobre TODAS las combinaciones
    #
    #    Score = 0.40 × Sharpe_medio
    #          + 0.30 × Calmar_medio
    #          + 0.20 × Consistencia  (fraccion de folds con Sharpe > 0)
    #          + 0.10 × (- Vol_media) (penaliza volatilidad alta)
    #
    #    Todo normalizado a [0,1] antes de ponderar para que las escalas
    #    no distorsionen el resultado.
    #    Ventaja vs mayoría: penaliza combinaciones que brillan en 1 fold
    #    y fallan en otros (exactamente el problema del criterio anterior).

    df_all_tmp = pd.DataFrame(all_rows)
    keys_combo = ["momentum_lookback", "momentum_skip", "alpha_min"]

    agg = df_all_tmp.groupby(keys_combo).agg(
        sharpe_medio  = ("sharpe",  "mean"),
        calmar_medio  = ("calmar",  "mean"),
        vol_media     = ("vol",     "mean"),
        consistencia  = ("sharpe",  lambda x: (x > 0).mean()),  # % folds > 0
    ).reset_index()

    # Normalizar cada componente a [0, 1]
    def norm01(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-9 else pd.Series(0.5, index=s.index)

    agg["n_sharpe"]  = norm01(agg["sharpe_medio"])
    agg["n_calmar"]  = norm01(agg["calmar_medio"])
    agg["n_consist"] = norm01(agg["consistencia"])
    agg["n_vol"]     = norm01(-agg["vol_media"])   # negativo: menor vol = mejor

    agg["score"] = (
        0.40 * agg["n_sharpe"]  +
        0.30 * agg["n_calmar"]  +
        0.20 * agg["n_consist"] +
        0.10 * agg["n_vol"]
    )

    best_row = agg.loc[agg["score"].idxmax()]
    params_optimos = {
        "momentum_lookback": int(best_row["momentum_lookback"]),
        "momentum_skip":     int(best_row["momentum_skip"]),
        "alpha_min":         float(best_row["alpha_min"]),
    }

    # Imprimir top 5 para transparencia
    print(f"\n  TOP 5 COMBINACIONES POR SCORE COMBINADO:")
    print(f"  {'LK':>4} {'SK':>4} {'AM':>5}  {'Sharpe':>7} {'Calmar':>7} "
          f"{'Consist':>8} {'Score':>7}")
    print(f"  {'-'*55}")
    for _, r in agg.nlargest(5, "score").iterrows():
        print(f"  {int(r['momentum_lookback']):>4}d {int(r['momentum_skip']):>3}d "
              f"{r['alpha_min']:>5.2f}  "
              f"{r['sharpe_medio']:>7.3f} {r['calmar_medio']:>7.3f} "
              f"{r['consistencia']*100:>7.0f}%  {r['score']:>7.4f}")

    print(f"\n  PARAMETROS OPTIMOS (score combinado):")
    print(f"    momentum_lookback = {params_optimos['momentum_lookback']} dias")
    print(f"    momentum_skip     = {params_optimos['momentum_skip']} dias")
    print(f"    alpha_min         = {params_optimos['alpha_min']}")
    print(f"    score             = {float(best_row['score']):.4f}")
    print(f"    sharpe_medio      = {float(best_row['sharpe_medio']):.3f}")
    print(f"    calmar_medio      = {float(best_row['calmar_medio']):.3f}")
    print(f"    consistencia      = {float(best_row['consistencia'])*100:.0f}% folds positivos")

    # 5. Evaluacion OOS 2025 (solo si se pide)
    oos_result = None
    if run_oos:
        print(f"\n  EVALUACION OOS 2025 con params optimos...")
        oos_result = run_one(
            data, rf,
            start = OOS_START,
            end   = OOS_END,
            lk    = params_optimos["momentum_lookback"],
            sk    = params_optimos["momentum_skip"],
            am    = params_optimos["alpha_min"],
            lam   = lam,
        )
        print(f"    Sharpe OOS : {oos_result['sharpe']:.3f}")
        print(f"    CAGR OOS   : {oos_result['cagr']*100:.2f}%")
        print(f"    MaxDD OOS  : {oos_result['maxdd']*100:.2f}%")
        print(f"    Trades OOS : {oos_result['trades']}")

    # 6. Guardar resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV detallado: todas las combinaciones en todos los folds
    df_all = pd.DataFrame(all_rows)
    path_all = OUTPUT_DIR / f"wf_resultados_{ticker.replace('.','_')}.csv"
    df_all.to_csv(path_all, index=False)

    # CSV resumen: mejor combo por fold
    resumen_rows = fold_best.copy()
    if oos_result:
        resumen_rows.append({
            "fold":              "OOS 2025",
            "val_start":         OOS_START,
            "val_end":           OOS_END,
            "best_sharpe":       round(oos_result["sharpe"], 4),
            "momentum_lookback": params_optimos["momentum_lookback"],
            "momentum_skip":     params_optimos["momentum_skip"],
            "alpha_min":         params_optimos["alpha_min"],
        })
    df_resumen = pd.DataFrame(resumen_rows)
    path_resumen = OUTPUT_DIR / f"wf_resumen_{ticker.replace('.','_')}.csv"
    df_resumen.to_csv(path_resumen, index=False)

    # JSON con params optimos
    params_output = {
        "ticker":            ticker,
        "fecha":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params_optimos":    params_optimos,
        "sharpe_medio_val":  round(float(best_row["sharpe_medio"]), 4),
        "calmar_medio_val":  round(float(best_row["calmar_medio"]), 4),
        "consistencia":      round(float(best_row["consistencia"]) * 100, 1),
        "score":             round(float(best_row["score"]), 4),
        "nota": (
            "Para usar estos parametros en la simulacion diaria, "
            "pasa momentum_lookback, momentum_skip y alpha_min al constructor "
            "de E2_DN_Momentum, o actualiza MOMENTUM_LOOKBACK, MOMENTUM_SKIP "
            "y ALPHA_MIN_MOMENTUM en config.py"
        )
    }
    path_json = OUTPUT_DIR / f"wf_params_optimos_{ticker.replace('.','_')}.json"
    with open(path_json, "w") as f:
        json.dump(params_output, f, indent=2)

    print(f"\n  Archivos guardados en: {OUTPUT_DIR}")
    print(f"    {path_all.name}")
    print(f"    {path_resumen.name}")
    print(f"    {path_json.name}")
    print("=" * 65 + "\n")

    # Métricas medias del combo óptimo sobre todos los folds
    df_opt = pd.DataFrame(all_rows)
    mask_opt = (
        (df_opt["momentum_lookback"] == params_optimos["momentum_lookback"]) &
        (df_opt["momentum_skip"]     == params_optimos["momentum_skip"])     &
        (df_opt["alpha_min"]         == params_optimos["alpha_min"])
    )
    df_opt_rows = df_opt[mask_opt]

    return {
        "params_optimos":   params_optimos,
        "df_all":           df_all,
        "df_resumen":       df_resumen,
        "oos_result":       oos_result,
        "sharpe_medio_val": round(float(df_opt_rows["sharpe"].mean()),     4),
        "cagr_medio_val":   round(float(df_opt_rows["cagr"].mean()),       4),
        "vol_medio_val":    round(float(df_opt_rows["vol"].mean()),        4),
        "maxdd_medio_val":  round(float(df_opt_rows["maxdd"].mean()),      4),
        "calmar_medio_val": round(float(df_opt_rows["calmar"].mean()),     4),
        "costes_medio_val": round(float(df_opt_rows["costes_pct"].mean()), 4),
        "trades_medio_val": round(float(df_opt_rows["trades"].mean()),     1),
    }


# ---------------------------------------------------------------------------
# GRAFICAS
# ---------------------------------------------------------------------------

# Paleta corporativa coherente con el resto del proyecto
C_BLUE   = "#1F4E79"
C_GREEN  = "#375623"
C_RED    = "#C00000"
C_ORANGE = "#C55A11"
C_GREY   = "#595959"
C_LGREY  = "#D9D9D9"
FOLD_COLORS = ["#2E75B6", "#70AD47", "#FF9900", "#C00000"]


def _label_combo(lk, sk, am):
    """Etiqueta corta para una combinacion de parametros."""
    return f"LK{lk}-SK{sk}-AM{am:.2f}"


def generar_graficas(df_all: pd.DataFrame, df_resumen: pd.DataFrame,
                     params_optimos: dict, oos_result: dict,
                     ticker: str, output_dir: Path) -> None:
    """
    Genera las 4 graficas profesionales del Walk-Forward y las guarda en
    output_dir como PNGs.

    Graficas:
        1. wf_heatmap_TICKER.png       — Sharpe de cada combo por fold
        2. wf_estabilidad_TICKER.png   — Params ganadores por fold
        3. wf_sensibilidad_TICKER.png  — Sharpe medio al variar cada param
        4. wf_degradacion_TICKER.png   — Sharpe IS → Val media → OOS
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t = ticker.replace(".", "_")
    folds = df_all["fold"].unique().tolist()

    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "axes.titlesize":  12,
        "axes.labelsize":  10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    # ------------------------------------------------------------------
    # GRAFICA 1 — Heatmap: Sharpe por combinacion y fold
    # ------------------------------------------------------------------
    # Crear etiqueta unica por combinacion
    df_all["combo"] = df_all.apply(
        lambda r: _label_combo(r["momentum_lookback"],
                               r["momentum_skip"],
                               r["alpha_min"]), axis=1
    )
    combos_sorted = sorted(df_all["combo"].unique())
    pivot = df_all.pivot_table(
        index="combo", columns="fold", values="sharpe", aggfunc="mean"
    ).reindex(index=combos_sorted, columns=folds)

    fig, ax = plt.subplots(figsize=(10, 7))
    vmin, vmax = pivot.values.min(), pivot.values.max()
    # Centrar colormap en 0
    vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f.replace("Fold ", "F") for f in folds], fontsize=9)
    ax.set_yticks(range(len(combos_sorted)))
    ax.set_yticklabels(combos_sorted, fontsize=7.5)

    # Anotar valores en cada celda
    for i in range(len(combos_sorted)):
        for j in range(len(folds)):
            val = pivot.values[i, j]
            color = "black" if -0.5 < val < 1.2 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    # Marcar fila del combo optimo
    opt_label = _label_combo(params_optimos["momentum_lookback"],
                              params_optimos["momentum_skip"],
                              params_optimos["alpha_min"])
    if opt_label in combos_sorted:
        opt_idx = combos_sorted.index(opt_label)
        for j in range(len(folds)):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, opt_idx - 0.5), 1, 1,
                fill=False, edgecolor=C_BLUE, linewidth=2.5
            ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Sharpe Ratio", fontsize=9)
    ax.set_title(
        f"Heatmap Sharpe — Walk-Forward E2 Momentum ({ticker})\n"
        f"Recuadro azul = parámetros óptimos seleccionados",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlabel("Fold de validación", fontsize=10)
    ax.set_ylabel("Combinación de parámetros  (LK=lookback  SK=skip  AM=alpha_min)",
                  fontsize=9)
    fig.tight_layout()
    path1 = output_dir / f"wf_heatmap_{t}.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grafica 1 guardada: {path1.name}")

    # ------------------------------------------------------------------
    # GRAFICA 2 — Estabilidad: params ganadores por fold
    # ------------------------------------------------------------------
    best_df = df_resumen[df_resumen["fold"].str.startswith("Fold")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    params_plot = [
        ("momentum_lookback", "Lookback (dias)", [126, 189, 252]),
        ("momentum_skip",     "Skip (dias)",     [21, 42]),
        ("alpha_min",         "Alpha min",        [0.0, 0.05, 0.10]),
    ]

    for ax, (col, label, vals) in zip(axes, params_plot):
        counts = best_df[col].value_counts().reindex(vals, fill_value=0)
        bars = ax.bar(
            [str(v) for v in vals], counts.values,
            color=[FOLD_COLORS[i % len(FOLD_COLORS)] for i in range(len(vals))],
            edgecolor="white", linewidth=1.2, width=0.55
        )
        # Marcar el optimo
        opt_val = params_optimos[col]
        opt_str = str(opt_val)
        for bar, v in zip(bars, [str(x) for x in vals]):
            if v == opt_str:
                bar.set_edgecolor(C_BLUE)
                bar.set_linewidth(3)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_xlabel("Valor del parámetro")
        ax.set_ylabel("Nº de folds ganados")
        ax.set_ylim(0, len(folds) + 0.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Anotar conteo
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Estabilidad de Parámetros por Fold — {ticker}\n"
        f"Borde azul = valor seleccionado como óptimo",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path2 = output_dir / f"wf_estabilidad_{t}.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grafica 2 guardada: {path2.name}")

    # ------------------------------------------------------------------
    # GRAFICA 3 — Sensibilidad: Sharpe medio al variar cada parametro
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (col, label, vals) in zip(axes, params_plot):
        medias = []
        stds   = []
        for v in vals:
            sub = df_all[df_all[col] == v]["sharpe"]
            medias.append(sub.mean())
            stds.append(sub.std())

        x      = range(len(vals))
        x_lbls = [str(v) for v in vals]
        ax.bar(x, medias, color=C_BLUE, alpha=0.75,
               edgecolor="white", linewidth=1, width=0.5)
        ax.errorbar(x, medias, yerr=stds, fmt="none",
                    color=C_GREY, capsize=5, linewidth=1.5)

        # Linea de referencia en 0
        ax.axhline(0, color=C_RED, linewidth=1, linestyle="--", alpha=0.7)

        # Marcar valor optimo
        opt_val = params_optimos[col]
        if opt_val in vals:
            opt_x = vals.index(opt_val)
            ax.bar(opt_x, medias[opt_x], color=C_GREEN, alpha=0.85,
                   edgecolor=C_BLUE, linewidth=2, width=0.5,
                   label="Óptimo seleccionado")
            ax.legend(fontsize=7, loc="upper right")

        ax.set_xticks(list(x))
        ax.set_xticklabels(x_lbls)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_xlabel("Valor del parámetro")
        ax.set_ylabel("Sharpe medio (todos los folds)")

        # Anotar valores
        for xi, (m, s) in enumerate(zip(medias, stds)):
            ax.text(xi, m + s + 0.02, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Sensibilidad del Sharpe a cada Parámetro — {ticker}\n"
        f"Barras de error = desviación estándar entre folds  |  Verde = óptimo",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path3 = output_dir / f"wf_sensibilidad_{t}.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grafica 3 guardada: {path3.name}")

    # ------------------------------------------------------------------
    # GRAFICA 4 — Degradacion temporal: dos series claramente separadas
    #
    # Serie A (azul): mejor combo DE CADA FOLD por separado
    #   → "techo teórico": el máximo Sharpe alcanzable en cada periodo
    #   → cada barra puede tener parámetros distintos
    #
    # Serie B (naranja): combo óptimo FINAL aplicado a todos los folds
    #   → "realidad operativa": lo que usamos en producción con params fijos
    #   → mismos parámetros en todos los folds
    #
    # La diferencia entre ambas series es el coste de usar parámetros fijos
    # en lugar de re-optimizar en cada periodo.
    # ------------------------------------------------------------------
    has_oos = oos_result is not None and oos_result.get("ok", False)

    # Datos Serie A — mejor combo de cada fold (ya en best_df)
    fold_labels   = [f.replace("Fold ", "Fold ") for f in best_df["fold"].tolist()]
    sharpes_best  = best_df["best_sharpe"].tolist()   # mejor posible por fold
    best_combos   = [
        f"LK{int(r['momentum_lookback'])}d / SK{int(r['momentum_skip'])}d / AM{r['alpha_min']:.2f}"
        for _, r in best_df.iterrows()
    ]

    # Datos Serie B — combo óptimo final en cada fold
    opt_lk = params_optimos["momentum_lookback"]
    opt_sk = params_optimos["momentum_skip"]
    opt_am = params_optimos["alpha_min"]
    opt_label = f"Óptimo final: LK{opt_lk}d / SK{opt_sk}d / AM{opt_am:.2f}"

    sharpes_opt = []
    for fold_lbl in best_df["fold"].tolist():
        mask = (
            (df_all["fold"]               == fold_lbl) &
            (df_all["momentum_lookback"]  == opt_lk)   &
            (df_all["momentum_skip"]      == opt_sk)   &
            (df_all["alpha_min"]          == opt_am)
        )
        sub = df_all[mask]
        sharpes_opt.append(float(sub["sharpe"].iloc[0]) if len(sub) > 0 else float("nan"))

    # Añadir OOS a ambas series
    x_labels = fold_labels.copy()
    if has_oos:
        x_labels.append("OOS 2025\n(test final)")
        sharpes_best.append(oos_result["sharpe"])   # OOS usa el óptimo final
        sharpes_opt.append(oos_result["sharpe"])    # mismo valor en OOS

    n      = len(x_labels)
    x      = np.arange(n)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 2), 6))

    # Barras Serie A — azul oscuro
    bars_a = ax.bar(x - width/2, sharpes_best, width,
                    color=C_BLUE, alpha=0.85, edgecolor="white",
                    linewidth=1.2, label="Mejor combo de cada fold (óptimo local)")

    # Barras Serie B — naranja
    bars_b = ax.bar(x + width/2, sharpes_opt, width,
                    color=C_ORANGE, alpha=0.85, edgecolor="white",
                    linewidth=1.2, label=f"Combo óptimo final aplicado a todos los folds\n({opt_label})")

    # Línea de referencia en 0
    ax.axhline(0, color=C_GREY, linewidth=0.8, linestyle="-", alpha=0.5)

    # Línea media del combo óptimo en validación
    media_opt_val = np.nanmean(sharpes_opt[:len(fold_labels)])
    ax.axhline(media_opt_val, color=C_ORANGE, linewidth=1.8, linestyle="--", alpha=0.8,
               label=f"Sharpe medio del óptimo en validación = {media_opt_val:.3f}")

    # Separador visual entre validación y OOS
    if has_oos:
        sep_x = len(fold_labels) - 0.5
        ax.axvline(sep_x, color=C_GREY, linewidth=1.5, linestyle=":", alpha=0.7)
        ax.text(sep_x - 0.1, ax.get_ylim()[1] * 0.95,
                "← Validación", ha="right", fontsize=8, color=C_GREY, style="italic")
        ax.text(sep_x + 0.1, ax.get_ylim()[1] * 0.95,
                "Test →", ha="left", fontsize=8, color=C_RED, style="italic",
                fontweight="bold")

    # Anotar valores sobre cada barra
    for bar, v in zip(bars_a, sharpes_best):
        if not np.isnan(v):
            ypos = v + 0.02 if v >= 0 else v - 0.07
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold", color=C_BLUE)

    for bar, v in zip(bars_b, sharpes_opt):
        if not np.isnan(v):
            ypos = v + 0.02 if v >= 0 else v - 0.07
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold", color=C_ORANGE)

    # Anotar el combo ganador de cada fold debajo del eje X
    for i, combo in enumerate(best_combos):
        ax.annotate(combo,
                    xy=(i - width/2, ax.get_ylim()[0]),
                    xytext=(i - width/2, ax.get_ylim()[0] - 0.12),
                    ha="center", va="top", fontsize=6.5,
                    color=C_BLUE, style="italic",
                    annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    ax.set_title(
        f"Degradación Temporal — Walk-Forward E2 Momentum ({ticker})\n"
        f"Azul = mejor combo posible en cada fold  |  "
        f"Naranja = combo óptimo final ({opt_lk}d/{opt_sk}d/AM{opt_am:.2f}) aplicado a todos",
        fontsize=10, fontweight="bold", pad=14
    )
    fig.tight_layout()
    path4 = output_dir / f"wf_degradacion_{t}.png"
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Grafica 4 guardada: {path4.name}")

    print(f"  → 4 graficas guardadas en: {output_dir}")


# ---------------------------------------------------------------------------
# TABLA RESUMEN COMPLETA
# ---------------------------------------------------------------------------

def generar_tabla_resumen(resumen_por_ticker: dict, output_dir: Path) -> None:
    """
    Tabla visual con las 7 métricas clave para cada ETF,
    separando validación media (folds) y OOS 2025.

    Métricas: Sharpe | CAGR | Vol | MaxDD | Calmar | Costes% | Trades
    """
    tickers  = list(resumen_por_ticker.keys())
    tiene_oos = any(
        resumen_por_ticker[t]["oos_result"] is not None
        and resumen_por_ticker[t]["oos_result"].get("ok", False)
        for t in tickers
    )

    # Columnas de la tabla
    METRICAS = ["Sharpe", "CAGR", "Vol", "MaxDD", "Calmar", "Costes%", "Trades"]

    # Formato display por métrica
    FMT = {
        "Sharpe":  lambda v: f"{v:.3f}",
        "CAGR":    lambda v: f"{v*100:.1f}%",
        "Vol":     lambda v: f"{v*100:.1f}%",
        "MaxDD":   lambda v: f"{v*100:.1f}%",
        "Calmar":  lambda v: f"{v:.2f}",
        "Costes%": lambda v: f"{v:.2f}%",
        "Trades":  lambda v: f"{int(v)}",
    }

    # Función de color por métrica y valor
    def color(metrica: str, v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "#F2F2F2"
        if metrica == "Sharpe":
            if v >= 0.6:  return "#C6EFCE"
            if v >= 0.2:  return "#FFEB9C"
            return "#FCE4D6"
        if metrica == "CAGR":
            if v >= 0.08:  return "#C6EFCE"
            if v >= 0.02:  return "#FFEB9C"
            return "#FCE4D6"
        if metrica == "Vol":
            if v <= 0.12:  return "#C6EFCE"
            if v <= 0.20:  return "#FFEB9C"
            return "#FCE4D6"
        if metrica == "MaxDD":
            if v >= -0.10: return "#C6EFCE"
            if v >= -0.20: return "#FFEB9C"
            return "#FCE4D6"
        if metrica == "Calmar":
            if v >= 0.5:   return "#C6EFCE"
            if v >= 0.2:   return "#FFEB9C"
            return "#FCE4D6"
        if metrica == "Costes%":
            if v <= 1.0:   return "#C6EFCE"
            if v <= 3.0:   return "#FFEB9C"
            return "#FCE4D6"
        return "#EBF3FB"   # Trades — neutro

    # Construir secciones: Val media | OOS (si existe)
    secciones = [("Validación media (folds)", "val")]
    if tiene_oos:
        secciones.append(("OOS 2025", "oos"))

    # Número de columnas: (nº métricas × nº secciones) + col params + col ETF
    n_secciones  = len(secciones)
    n_cols_datos = len(METRICAS) * n_secciones
    # Cabeceras de dos niveles: sección / métrica
    col_labels_top  = []   # nivel superior (sección)
    col_labels_bot  = []   # nivel inferior (métrica)
    col_section_idx = []   # índice de sección para colores

    for sec_label, sec_key in secciones:
        for m in METRICAS:
            col_labels_top.append(sec_label)
            col_labels_bot.append(m)
            col_section_idx.append(sec_key)

    # Añadir columnas de parámetros al final
    param_cols = ["Lookback", "Skip", "AlphaMin"]
    for p in param_cols:
        col_labels_top.append("Parámetros")
        col_labels_bot.append(p)
        col_section_idx.append("param")

    # Extraer valores de cada ETF
    cell_text   = []
    cell_colors = []

    for t in tickers:
        res = resumen_por_ticker[t]
        p   = res["params_optimos"]
        fila_txt    = []
        fila_colors = []

        for sec_label, sec_key in secciones:
            if sec_key == "val":
                vals = {
                    "Sharpe":  res.get("sharpe_medio_val",  float("nan")),
                    "CAGR":    res.get("cagr_medio_val",    float("nan")),
                    "Vol":     res.get("vol_medio_val",     float("nan")),
                    "MaxDD":   res.get("maxdd_medio_val",   float("nan")),
                    "Calmar":  res.get("calmar_medio_val",  float("nan")),
                    "Costes%": res.get("costes_medio_val",  float("nan")),
                    "Trades":  res.get("trades_medio_val",  float("nan")),
                }
            else:  # oos
                oos = res.get("oos_result") or {}
                vals = {
                    "Sharpe":  oos.get("sharpe",     float("nan")),
                    "CAGR":    oos.get("cagr",        float("nan")),
                    "Vol":     oos.get("vol",         float("nan")),
                    "MaxDD":   oos.get("maxdd",       float("nan")),
                    "Calmar":  oos.get("calmar",      float("nan")),
                    "Costes%": oos.get("costes_pct",  float("nan")),
                    "Trades":  oos.get("trades",      float("nan")),
                }
            for m in METRICAS:
                v = vals[m]
                if isinstance(v, float) and np.isnan(v):
                    fila_txt.append("—")
                    fila_colors.append("#F2F2F2")
                else:
                    fila_txt.append(FMT[m](v))
                    fila_colors.append(color(m, v))

        # Columnas de parámetros
        fila_txt   += [f"{p['momentum_lookback']}d",
                       f"{p['momentum_skip']}d",
                       f"{p['alpha_min']:.2f}"]
        fila_colors += ["#EBF3FB", "#EBF3FB", "#EBF3FB"]

        cell_text.append(fila_txt)
        cell_colors.append(fila_colors)

    # Colores de cabecera por sección
    SEC_COLORS = {"val": C_BLUE, "oos": C_RED, "param": C_GREY}
    col_header_colors = [SEC_COLORS[s] for s in col_section_idx]

    # Dibujar figura
    n_rows   = len(tickers)
    n_cols   = len(col_labels_bot)
    fig_w    = max(16, n_cols * 1.4)
    fig_h    = max(5,  n_rows * 1.1 + 3.5)
    fig, ax  = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Tabla principal (métricas, sin cabecera de sección aún)
    tbl = ax.table(
        cellText    = cell_text,
        rowLabels   = tickers,
        colLabels   = col_labels_bot,
        cellColours = cell_colors,
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 2.4)

    # Estilo cabeceras de columna (nivel métrica)
    for j, hc in enumerate(col_header_colors):
        cell = tbl[0, j]
        cell.set_facecolor(hc)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Estilo cabeceras de fila (tickers)
    ticker_colors_map = dict(zip(tickers, ["#2E75B6", "#70AD47", "#FF9900", "#C00000"]))
    for i, t in enumerate(tickers):
        cell = tbl[i + 1, -1]
        cell.set_facecolor(ticker_colors_map.get(t, C_GREY))
        cell.set_text_props(color="white", fontweight="bold")

    # Añadir cabecera de sección encima de la tabla como texto
    # (matplotlib table no soporta multi-nivel nativo, usamos anotaciones)
    renderer   = fig.canvas.get_renderer()
    ax_bbox    = ax.get_window_extent(renderer)
    tbl_bbox   = tbl.get_window_extent(renderer)

    # Leyenda de secciones y colores
    leyenda = [
        mpatches.Patch(color=C_BLUE,    label="Validación media (promedio de folds)"),
        mpatches.Patch(color=C_RED,     label="OOS 2025 — test final intocable") if tiene_oos else None,
        mpatches.Patch(color=C_GREY,    label="Parámetros óptimos seleccionados"),
        mpatches.Patch(color="#C6EFCE", label="Bueno"),
        mpatches.Patch(color="#FFEB9C", label="Moderado"),
        mpatches.Patch(color="#FCE4D6", label="Débil / Negativo"),
    ]
    leyenda = [p for p in leyenda if p is not None]
    ax.legend(handles=leyenda, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=8.5, frameon=True, framealpha=0.9)

    # Etiquetas de sección encima (texto manual)
    sec_starts = {}
    for i, (sec_label, sec_key) in enumerate(secciones):
        sec_starts[sec_key] = i * len(METRICAS)
    for sec_label, sec_key in secciones:
        mid_col  = sec_starts[sec_key] + len(METRICAS) / 2 - 0.5
        ax.annotate(
            sec_label,
            xy         = (mid_col / n_cols, 1.06),
            xycoords   = "axes fraction",
            ha         = "center", va  = "center",
            fontsize   = 10, fontweight = "bold",
            color      = SEC_COLORS[sec_key],
            annotation_clip = False,
        )

    ax.set_title(
        f"Tabla Resumen Walk-Forward — E2 DN+Momentum\n"
        f"Métricas clave del mejor combo de parámetros por ETF",
        fontsize=12, fontweight="bold", pad=28,
    )

    fig.tight_layout()
    path = output_dir / "wf_tabla_resumen.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tabla resumen guardada: {path.name}")


# ---------------------------------------------------------------------------
# GRAFICA COMPARATIVA ENTRE ETFs
# ---------------------------------------------------------------------------

def generar_grafica_comparativa(resumen_por_ticker: dict, output_dir: Path) -> None:
    """
    Grafica 5: Compara los parametros optimos y Sharpe medio entre los 4 ETFs.
    Permite ver si los parametros son consistentes entre activos (robustez)
    o cambian mucho (dependencia del activo).
    """
    tickers = list(resumen_por_ticker.keys())
    n       = len(tickers)
    if n < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Comparativa Walk-Forward entre los 4 ETFs\n"
        "¿Son los parámetros óptimos consistentes entre activos?",
        fontsize=12, fontweight="bold", y=1.01
    )

    params_plot = [
        ("momentum_lookback", "Lookback óptimo (días)", [126, 189, 252]),
        ("momentum_skip",     "Skip óptimo (días)",     [21, 42]),
        ("alpha_min",         "Alpha min óptimo",        [0.0, 0.05, 0.10]),
    ]
    ticker_colors = dict(zip(tickers, ["#2E75B6", "#70AD47", "#FF9900", "#C00000"]))

    # Subplot 1-3: param óptimo por ETF
    for ax, (col, label, vals) in zip(axes.flat[:3], params_plot):
        x      = range(n)
        opt_vals = [resumen_por_ticker[t]["params_optimos"][col] for t in tickers]
        bars   = ax.bar(x, opt_vals,
                        color=[ticker_colors[t] for t in tickers],
                        edgecolor="white", linewidth=1.2, width=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tickers, fontsize=9)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_ylabel("Valor óptimo")
        ax.set_ylim(0, max(vals) * 1.25)
        # Anotar valor
        for bar, v in zip(bars, opt_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    str(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        # Linea de referencia: valor más frecuente entre tickers
        from collections import Counter
        most_common = Counter(opt_vals).most_common(1)[0][0]
        ax.axhline(most_common, color=C_GREY, linewidth=1.2,
                   linestyle="--", alpha=0.6,
                   label=f"Más frecuente: {most_common}")
        ax.legend(fontsize=8)

    # Subplot 4: Sharpe medio de validación por ETF
    ax4 = axes.flat[3]
    sharpes = [resumen_por_ticker[t]["sharpe_medio_val"] for t in tickers]
    bars = ax4.bar(range(n), sharpes,
                   color=[ticker_colors[t] for t in tickers],
                   edgecolor="white", linewidth=1.2, width=0.5)
    ax4.axhline(0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax4.set_xticks(list(range(n)))
    ax4.set_xticklabels(tickers, fontsize=9)
    ax4.set_title("Sharpe medio de validación", fontweight="bold", pad=8)
    ax4.set_ylabel("Sharpe Ratio")
    for bar, v in zip(bars, sharpes):
        ypos = v + 0.02 if v >= 0 else v - 0.06
        ax4.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{v:.3f}", ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = output_dir / "wf_comparativa_etfs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Grafica comparativa guardada: {path.name}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Analysis para E2_DN_Momentum — 4 ETFs"
    )
    parser.add_argument(
        "--ticker", type=str, default="all",
        help="Ticker del ETF (default: all = los 4 ETFs)"
    )
    parser.add_argument(
        "--oos", action="store_true",
        help="Evaluar tambien en OOS 2025 con los params optimos"
    )
    args = parser.parse_args()

    # Determinar qué tickers procesar
    if args.ticker.lower() == "all":
        tickers = ALL_TICKERS
    else:
        tickers = [args.ticker]

    print(f"\n  ETFs a procesar: {tickers}")

    resumen_por_ticker = {}

    for ticker in tickers:
        resultado = run_walk_forward(ticker=ticker, run_oos=args.oos)

        resumen_por_ticker[ticker] = {
            "params_optimos":   resultado["params_optimos"],
            "sharpe_medio_val": resultado["sharpe_medio_val"],
            "cagr_medio_val":   resultado["cagr_medio_val"],
            "vol_medio_val":    resultado["vol_medio_val"],
            "maxdd_medio_val":  resultado["maxdd_medio_val"],
            "calmar_medio_val": resultado["calmar_medio_val"],
            "costes_medio_val": resultado["costes_medio_val"],
            "trades_medio_val": resultado["trades_medio_val"],
            "oos_result":       resultado["oos_result"],
        }

        print(f"\n  Generando graficas para {ticker}...")
        generar_graficas(
            df_all         = resultado["df_all"],
            df_resumen     = resultado["df_resumen"],
            params_optimos = resultado["params_optimos"],
            oos_result     = resultado["oos_result"],
            ticker         = ticker,
            output_dir     = OUTPUT_DIR,
        )

    # Gráfica comparativa entre ETFs (solo si se procesó más de uno)
    if len(resumen_por_ticker) > 1:
        print("\n  Generando grafica comparativa entre ETFs...")
        generar_grafica_comparativa(resumen_por_ticker, OUTPUT_DIR)

    # Tabla resumen completa con todas las métricas
    print("\n  Generando tabla resumen completa...")
    generar_tabla_resumen(resumen_por_ticker, OUTPUT_DIR)

    # Resumen final en consola
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL — PARAMETROS OPTIMOS POR ETF")
    print("=" * 65)
    print(f"  {'ETF':<12} {'Lookback':>10} {'Skip':>6} {'AlphaMin':>10} {'Sharpe Val':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*10} {'-'*12}")
    for ticker, res in resumen_por_ticker.items():
        p = res["params_optimos"]
        print(f"  {ticker:<12} {p['momentum_lookback']:>10} "
              f"{p['momentum_skip']:>6} {p['alpha_min']:>10.2f} "
              f"{res['sharpe_medio_val']:>12.3f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()