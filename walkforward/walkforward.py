"""
Walkforward validation.

Usa backtest.engine.run_backtest: misma pipeline (main.run_pipeline), misma regla
de rebalanceo que main/registrador (portfolio.rebalance_policy), costes y DN.

Divide el periodo en ventanas train + test; en cada fold corre el backtest
solo en el tramo de test (el lookback del engine aporta historia previa).

Agrega metricas de cada fold y las compara con los benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest
import config as cfg


def run_walkforward(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    train_months: int | None = None,
    test_months: int | None = None,
    output_dir: str = "results",
) -> dict:
    """
    Ejecuta walkforward: para cada ventana de test, corre el backtest completo.

    Returns:
        dict con walkforward DataFrame y directorio de salida
    """
    if train_months is None:
        train_months = cfg.WF_TRAIN_MONTHS
    if test_months is None:
        test_months = cfg.WF_TEST_MONTHS

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    train_start = start
    fold = 0

    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        fold += 1
        fold_dir = output_path / f"walkforward_fold_{fold}"

        print(f"\n{'=' * 60}")
        print(f"WALKFORWARD FOLD {fold}")
        print(f"  Train: {train_start.date()} -> {train_end.date()}")
        print(f"  Test:  {test_start.date()} -> {test_end.date()}")
        print(f"{'=' * 60}")

        try:
            bt_result = run_backtest(
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
                output_dir=str(fold_dir),
            )

            row = {
                "Fold": fold,
                "Train Start": train_start.strftime("%Y-%m-%d"),
                "Train End": train_end.strftime("%Y-%m-%d"),
                "Test Start": test_start.strftime("%Y-%m-%d"),
                "Test End": test_end.strftime("%Y-%m-%d"),
                "Rebalances": bt_result["n_rebalances"],
            }
            for k, v in bt_result["strategy_metrics"].items():
                row[f"Strategy {k}"] = v
            for k, v in bt_result["sp500_metrics"].items():
                row[f"SP500 {k}"] = v

            rows.append(row)

        except Exception as e:
            print(f"  [Fold {fold}] Error: {e}")
            rows.append({
                "Fold": fold,
                "Train Start": train_start.strftime("%Y-%m-%d"),
                "Train End": train_end.strftime("%Y-%m-%d"),
                "Test Start": test_start.strftime("%Y-%m-%d"),
                "Test End": test_end.strftime("%Y-%m-%d"),
                "Error": str(e),
            })

        train_start = train_start + pd.DateOffset(months=test_months)

    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(output_path / "walkforward_resumen.csv", index=False)

    try:
        wf_df.to_excel(output_path / "walkforward_resumen.xlsx", index=False)
    except PermissionError:
        wf_df.to_excel(output_path / "walkforward_resumen_backup.xlsx", index=False)

    print(f"\n{'=' * 60}")
    print("WALKFORWARD RESUMEN")
    print("=" * 60)
    if not wf_df.empty:
        print(wf_df.to_string(index=False))
    else:
        print("No se completaron folds.")

    return {
        "walkforward": wf_df,
        "output_dir": str(output_path),
    }
