"""
Optimizacion de parametros por grid search.

Parametros optimizados:
  - MERTON_GAMMA: aversion al riesgo (menor abs = mas agresivo)
  - VIEW_SCALE: intensidad de la senal en las views de BL
  - BL_TAU: incertidumbre del prior
  - MERTON_MAX_SECTOR: limite por sector (mayor = apuestas mas concentradas)

RF NO se optimiza (usa tipo BCE real).
Regimen contrarian (siempre 100% invertido).

Uso:
    python main_optimization.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import pandas as pd

import config as cfg
from backtest.engine import run_backtest

# ============================================================
# GRID DE PARAMETROS
# ============================================================

PARAM_GRID = {
    "MERTON_GAMMA":      [-0.5, -0.7, -0.8, -1.0],
    "VIEW_SCALE":        [0.22, 0.28, 0.35],
    "BL_TAU":            [0.03, 0.05, 0.07],
    "MERTON_MAX_SECTOR": [0.30, 0.35, 0.40],
}

BACKTEST_START = "2020-01-01"
BACKTEST_END   = "2025-12-31"
OUTPUT_BASE    = "results_optimization"


# ============================================================
# FUNCIONES
# ============================================================

def run_optimization():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    n_combos = len(combinations)

    output_base = Path(OUTPUT_BASE)
    output_base.mkdir(exist_ok=True)

    originals = {k: getattr(cfg, k) for k in keys}

    print(f"{'=' * 70}")
    print(f"OPTIMIZACION: {n_combos} combinaciones")
    print(f"Periodo: {BACKTEST_START} -> {BACKTEST_END}")
    print(f"Parametros: {keys}")
    print(f"RF: DINAMICO (tipo BCE real)")
    print(f"Regimen: CONTRARIAN (100% invertido siempre)")
    print(f"{'=' * 70}")

    results: list[dict] = []
    t_total_start = time.time()

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        for k, v in params.items():
            setattr(cfg, k, v)

        combo_dir = output_base / f"combo_{i + 1:03d}"
        params_str = " | ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n[{i + 1}/{n_combos}] {params_str}")

        try:
            t0 = time.time()
            bt = run_backtest(
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
                output_dir=str(combo_dir),
                verbose=False,
            )
            elapsed = time.time() - t0

            row = {"combo": i + 1, **params}
            for k_m, v_m in bt["strategy_metrics"].items():
                row[f"Strategy_{k_m}"] = v_m
            for k_m, v_m in bt["sp500_metrics"].items():
                row[f"SP500_{k_m}"] = v_m
            row["n_rebalances"] = bt["n_rebalances"]
            row["beats_sp500"] = (
                bt["strategy_metrics"]["Total Return"]
                > bt["sp500_metrics"]["Total Return"]
            )
            row["elapsed_s"] = round(elapsed, 1)
            row["combo_dir"] = str(combo_dir)

            results.append(row)

            s_ret = bt["strategy_metrics"]["Total Return"]
            s_sharpe = bt["strategy_metrics"]["Sharpe"]
            s_dd = bt["strategy_metrics"]["Max Drawdown"]
            beat = "*** BEAT SP500 ***" if row["beats_sp500"] else ""
            print(
                f"  Return={s_ret:.2%} | Sharpe={s_sharpe:.2f} | "
                f"MaxDD={s_dd:.2%} | {elapsed:.1f}s {beat}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"combo": i + 1, **params, "error": str(e)})

    for k, v in originals.items():
        setattr(cfg, k, v)

    t_total = time.time() - t_total_start

    # --- Resumen ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base / "optimization_summary.csv", index=False)
    try:
        results_df.to_excel(output_base / "optimization_summary.xlsx", index=False)
    except PermissionError:
        pass

    print(f"\n{'=' * 70}")
    print(f"OPTIMIZACION COMPLETADA en {t_total:.0f}s ({n_combos} combinaciones)")
    print(f"{'=' * 70}")

    if "Strategy_Sharpe" in results_df.columns:
        valid = results_df.dropna(subset=["Strategy_Sharpe"])

        if not valid.empty:
            top5 = valid.nlargest(5, "Strategy_Sharpe")

            print(f"\nTOP 5 MEJORES COMBINACIONES (por Sharpe):")
            print("-" * 70)
            for _, row in top5.iterrows():
                print(
                    f"  Combo {int(row['combo']):3d} | "
                    f"gamma={row['MERTON_GAMMA']:5.1f} | "
                    f"view={row['VIEW_SCALE']:.2f} | "
                    f"tau={row['BL_TAU']:.2f} | "
                    f"sector={row['MERTON_MAX_SECTOR']:.2f} | "
                    f"Return={row['Strategy_Total Return']:.2%} | "
                    f"Sharpe={row['Strategy_Sharpe']:.2f} | "
                    f"MaxDD={row['Strategy_Max Drawdown']:.2%} | "
                    f"{'BEAT' if row.get('beats_sp500') else ''}"
                )

            top5_ret = valid.nlargest(5, "Strategy_Total Return")
            print(f"\nTOP 5 MEJORES COMBINACIONES (por Total Return):")
            print("-" * 70)
            for _, row in top5_ret.iterrows():
                print(
                    f"  Combo {int(row['combo']):3d} | "
                    f"gamma={row['MERTON_GAMMA']:5.1f} | "
                    f"view={row['VIEW_SCALE']:.2f} | "
                    f"tau={row['BL_TAU']:.2f} | "
                    f"sector={row['MERTON_MAX_SECTOR']:.2f} | "
                    f"Return={row['Strategy_Total Return']:.2%} | "
                    f"Sharpe={row['Strategy_Sharpe']:.2f} | "
                    f"MaxDD={row['Strategy_Max Drawdown']:.2%} | "
                    f"{'BEAT' if row.get('beats_sp500') else ''}"
                )

            n_beat = valid["beats_sp500"].sum()
            print(
                f"\n{int(n_beat)}/{len(valid)} combinaciones "
                f"superan al SP500 en retorno total"
            )

    print(f"\nResultados guardados en: {output_base}/")

    return results_df


if __name__ == "__main__":
    run_optimization()
