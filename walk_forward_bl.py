"""
Walk-Forward validation de la estrategia BL-Omega.
Compatible con el seam confirmado de main_black_litterman.run_backtest(...).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from config import INITIAL_WEALTH, OUTPUT_DIR_BL, OUTPUT_DIR_WF
from main_black_litterman import _load_shared_data, run_backtest
from plots_bl import plot_walk_forward_sharpe, plot_walk_forward_wealth

WF_OUTPUT_DIR = Path(OUTPUT_DIR_WF)
IS_YEARS = 3
OOS_YEARS = 1
PASO_YEARS = 1
PARAM_GRID = {
    "omega_window_fast": [21, 42, 63],
    "gamma": [-1, -2, -3],
    "recalib_freq": [5, 10, 21],
}
METRIC_IS = "sharpe"


@dataclass(frozen=True)
class WalkForwardWindow:
    ventana: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "ventana": self.ventana,
            "is_start": self.is_start,
            "is_end": self.is_end,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
        }


def _coerce_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize() if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)


def generate_windows(data_start, data_end, is_years=IS_YEARS, oos_years=OOS_YEARS, paso=PASO_YEARS) -> list[dict[str, pd.Timestamp]]:
    data_start = _coerce_timestamp(data_start)
    data_end = _coerce_timestamp(data_end)
    windows: list[dict[str, pd.Timestamp]] = []
    current = data_start
    ventana = 1

    while True:
        is_start = current
        is_end = current + pd.DateOffset(years=is_years) - pd.DateOffset(days=1)
        oos_start = is_end + pd.DateOffset(days=1)
        oos_end = oos_start + pd.DateOffset(years=oos_years) - pd.DateOffset(days=1)
        if oos_end > data_end:
            break
        window = WalkForwardWindow(
            ventana=ventana,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
        )
        windows.append(window.as_dict())
        current = current + pd.DateOffset(years=paso)
        ventana += 1
    return windows


def _iter_param_grid(param_grid: dict[str, list[Any]]):
    keys = list(param_grid.keys())
    for combo in itertools.product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, combo))


def optimize_is(is_start, is_end, param_grid, ventana_num, metric_key: str = METRIC_IS):
    best_params: dict[str, Any] | None = None
    best_metric = -np.inf
    all_results: list[dict[str, Any]] = []

    for combo_idx, params in enumerate(_iter_param_grid(param_grid), start=1):
        error = None
        try:
            result = run_backtest(start=is_start, end=is_end, params=params, save_results=False)
            metric_value = float(result[metric_key])
            cagr_value = float(result.get("cagr", np.nan))
            max_dd_value = float(result.get("max_dd", np.nan))
        except Exception as exc:
            metric_value = -np.inf
            cagr_value = np.nan
            max_dd_value = np.nan
            error = str(exc)

        row = {
            "ventana": ventana_num,
            "combo": combo_idx,
            "is_start": pd.Timestamp(is_start).date(),
            "is_end": pd.Timestamp(is_end).date(),
            metric_key: metric_value,
            "cagr": cagr_value,
            "max_dd": max_dd_value,
            "error": error,
            **params,
        }
        all_results.append(row)

        if metric_value > best_metric:
            best_metric = metric_value
            best_params = params.copy()

    if best_params is None:
        best_params = {key: values[0] for key, values in param_grid.items()}
    return best_params, float(best_metric), all_results


def _chain_oos_wealth(series_list: list[pd.Series], initial_wealth: float = INITIAL_WEALTH) -> pd.Series:
    if not series_list:
        return pd.Series(dtype=float, name="wealth")

    chained_parts: list[pd.Series] = []
    running_wealth = float(initial_wealth)
    seen_dates: set[pd.Timestamp] = set()

    for wealth in series_list:
        wealth = pd.Series(wealth).dropna().sort_index()
        if wealth.empty:
            continue
        scaled = running_wealth * (wealth / float(initial_wealth))
        keep_mask = ~scaled.index.isin(seen_dates)
        scaled = scaled.loc[keep_mask]
        if scaled.empty:
            continue
        chained_parts.append(scaled)
        running_wealth = float(scaled.iloc[-1])
        seen_dates.update(pd.Timestamp(idx) for idx in scaled.index)

    if not chained_parts:
        return pd.Series(dtype=float, name="wealth")
    concatenated = pd.concat(chained_parts).sort_index()
    concatenated.name = "wealth"
    return concatenated


def _load_backtest_simple_wealth() -> pd.Series | None:
    wealth_file = Path(OUTPUT_DIR_BL) / "wealth_history.csv"
    if not wealth_file.exists():
        return None
    history_df = pd.read_csv(wealth_file, index_col=0, parse_dates=True)
    if "wealth" not in history_df.columns:
        return None
    return pd.to_numeric(history_df["wealth"], errors="coerce").dropna()


def _save_outputs(summary_df: pd.DataFrame, best_params_df: pd.DataFrame, grid_df: pd.DataFrame, wealth_oos: pd.Series | None):
    WF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(WF_OUTPUT_DIR / "resultados_wf.csv", index=False)
    best_params_df.to_csv(WF_OUTPUT_DIR / "mejores_params_por_ventana.csv", index=False)
    grid_df.to_csv(WF_OUTPUT_DIR / "grid_is_completo.csv", index=False)
    if wealth_oos is not None and not wealth_oos.empty:
        wealth_oos.to_csv(WF_OUTPUT_DIR / "wealth_oos_concatenado.csv", header=["wealth"])


def run_walk_forward(
    is_years: int = IS_YEARS,
    oos_years: int = OOS_YEARS,
    paso_years: int = PASO_YEARS,
    param_grid: dict[str, list[Any]] | None = None,
    metric_is: str = METRIC_IS,
    save_results: bool = True,
):
    param_grid = param_grid or PARAM_GRID
    data, _ = _load_shared_data()
    data_start = pd.Timestamp(data["backtest_start"])
    data_end = pd.Timestamp(data["returns"].index[-1])
    windows = generate_windows(data_start, data_end, is_years=is_years, oos_years=oos_years, paso=paso_years)
    if not windows:
        raise ValueError("No hay suficientes datos para generar ventanas IS/OOS")

    print("=" * 60)
    print("WALK-FORWARD VALIDATION — BL-Omega")
    print("=" * 60)
    print(f"Datos disponibles: {data_start.date()} → {data_end.date()}")
    print(f"IS={is_years} años | OOS={oos_years} año(s) | paso={paso_years} año(s)")
    print(f"Grid: {param_grid}")

    summary_rows: list[dict[str, Any]] = []
    best_params_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    oos_wealth_series: list[pd.Series] = []
    started_at = datetime.now()

    for position, window in enumerate(windows, start=1):
        ventana = int(window["ventana"])
        print(f"\n{'─' * 50}")
        print(
            f"Ventana {ventana}/{len(windows)} | "
            f"IS={window['is_start'].date()}→{window['is_end'].date()} | "
            f"OOS={window['oos_start'].date()}→{window['oos_end'].date()}"
        )

        best_params, best_metric, all_is_results = optimize_is(
            window["is_start"],
            window["is_end"],
            param_grid,
            ventana_num=ventana,
            metric_key=metric_is,
        )
        grid_rows.extend(all_is_results)

        oos_error = None
        sharpe_oos = np.nan
        cagr_oos = np.nan
        max_dd_oos = np.nan
        try:
            oos_result = run_backtest(
                start=window["oos_start"],
                end=window["oos_end"],
                params=best_params,
                save_results=False,
            )
            sharpe_oos = float(oos_result["sharpe"])
            cagr_oos = float(oos_result["cagr"])
            max_dd_oos = float(oos_result["max_dd"])
            oos_wealth_series.append(pd.to_numeric(oos_result["history"]["wealth"], errors="coerce").dropna())
        except Exception as exc:
            oos_error = str(exc)

        summary_rows.append(
            {
                "ventana": ventana,
                "is_start": pd.Timestamp(window["is_start"]).date(),
                "is_end": pd.Timestamp(window["is_end"]).date(),
                "oos_start": pd.Timestamp(window["oos_start"]).date(),
                "oos_end": pd.Timestamp(window["oos_end"]).date(),
                "sharpe_is": round(float(best_metric), 4),
                "sharpe_oos": round(sharpe_oos, 4) if not np.isnan(sharpe_oos) else np.nan,
                "cagr_oos": round(cagr_oos, 4) if not np.isnan(cagr_oos) else np.nan,
                "max_dd_oos": round(max_dd_oos, 4) if not np.isnan(max_dd_oos) else np.nan,
                "oos_error": oos_error,
                **{f"param_{key}": value for key, value in best_params.items()},
            }
        )
        best_params_rows.append(
            {
                "ventana": ventana,
                "is_start": pd.Timestamp(window["is_start"]).date(),
                "is_end": pd.Timestamp(window["is_end"]).date(),
                "best_metric": round(float(best_metric), 4),
                "metric_name": metric_is,
                **best_params,
            }
        )

        elapsed = (datetime.now() - started_at).total_seconds() / 60.0
        avg_window = elapsed / position
        remaining = avg_window * (len(windows) - position)
        print(f"Mejores params: {best_params} | {metric_is} IS={best_metric:.4f}")
        if oos_error:
            print(f"OOS falló: {oos_error}")
        else:
            print(f"OOS Sharpe={sharpe_oos:.4f} | CAGR={cagr_oos:.2%} | MaxDD={max_dd_oos:.2%}")
        print(f"Tiempo transcurrido={elapsed:.1f} min | restante≈{remaining:.1f} min")

    summary_df = pd.DataFrame(summary_rows)
    best_params_df = pd.DataFrame(best_params_rows)
    grid_df = pd.DataFrame(grid_rows)
    wealth_oos = _chain_oos_wealth(oos_wealth_series, initial_wealth=INITIAL_WEALTH)

    sharpe_is_mean = float(summary_df["sharpe_is"].dropna().mean()) if not summary_df.empty else np.nan
    sharpe_oos_valid = summary_df["sharpe_oos"].dropna()
    sharpe_oos_mean = float(sharpe_oos_valid.mean()) if not sharpe_oos_valid.empty else np.nan
    ratio_oos_is = float(sharpe_oos_mean / sharpe_is_mean) if np.isfinite(sharpe_is_mean) and abs(sharpe_is_mean) > 1e-12 else np.nan

    if save_results:
        _save_outputs(summary_df, best_params_df, grid_df, wealth_oos)
        plot_walk_forward_sharpe(summary_df, save_path=WF_OUTPUT_DIR / "grafico_wf_sharpe.png")
        plot_walk_forward_wealth(
            wealth_oos,
            benchmark_wealth=_load_backtest_simple_wealth(),
            save_path=WF_OUTPUT_DIR / "grafico_wf_wealth.png",
        )

    print(f"\nSharpe IS medio:  {sharpe_is_mean:.4f}")
    print(f"Sharpe OOS medio: {sharpe_oos_mean:.4f}")
    print(f"Ratio OOS/IS:     {ratio_oos_is:.4f}")
    if save_results:
        print(f"Outputs guardados en: {WF_OUTPUT_DIR}")

    return {
        "summary": summary_df,
        "best_params": best_params_df,
        "grid": grid_df,
        "wealth_oos": wealth_oos,
        "sharpe_is_mean": sharpe_is_mean,
        "sharpe_oos_mean": sharpe_oos_mean,
        "ratio_oos_is": ratio_oos_is,
        "windows": pd.DataFrame(windows),
        "output_dir": WF_OUTPUT_DIR,
    }


if __name__ == "__main__":
    run_walk_forward()
