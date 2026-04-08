"""
Reconstruccion de la cartera viva a partir de los Excel operaciones_rebalanceo_*.xlsx
y generacion de informes (NAV, metricas, benchmark, costes).

Uso: main_portfolio_backtest.py o run_live_portfolio_report().
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from backtest.engine import _compute_metrics, _format_metrics
from Data.data_loader import download_market_data
from portfolio.positions_io import load_positions_from_excel, portfolio_value_eur

OPERACIONES_RE = re.compile(r"operaciones_rebalanceo_(\d{4}-\d{2}-\d{2})\.xlsx$", re.IGNORECASE)


def discover_operaciones_files(results_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    out: list[tuple[pd.Timestamp, Path]] = []
    for p in sorted(results_dir.glob("operaciones_rebalanceo_*.xlsx")):
        m = OPERACIONES_RE.search(p.name)
        if not m:
            continue
        out.append((pd.Timestamp(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_orders_from_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Ordenes")
    if df.empty:
        return df
    expected = {"ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"}
    cols = set(df.columns)
    if not expected.issubset(cols):
        raise ValueError(f"Hoja Ordenes incompleta en {path}: faltan columnas {expected - cols}")
    return df


def apply_orders_to_positions(positions: dict[str, float], orders_df: pd.DataFrame) -> dict[str, float]:
    out = dict(positions)
    if orders_df is None or orders_df.empty:
        return out
    for _, row in orders_df.iterrows():
        tid = str(row["ID"]).strip()
        dq = float(row["Cantidad"])
        out[tid] = out.get(tid, 0.0) + dq
    return {k: float(v) for k, v in out.items() if abs(float(v)) > 1e-12}


def estimate_orders_cost_eur(orders_df: pd.DataFrame) -> float:
    if orders_df is None or orders_df.empty:
        return 0.0
    total = 0.0
    for _, row in orders_df.iterrows():
        q = abs(float(row["Cantidad"]))
        p = float(row["Precio"])
        ct = float(row.get("CT", 0.0) or 0.0)
        total += q * p * ct
    return total


def estimate_notional_eur(orders_df: pd.DataFrame) -> float:
    if orders_df is None or orders_df.empty:
        return 0.0
    return float(abs(orders_df["Cantidad"]).mul(orders_df["Precio"].astype(float)).sum())


def rebuild_events_from_operaciones(
    files: list[tuple[pd.Timestamp, Path]],
    initial_positions: dict[str, float] | None = None,
) -> tuple[list[tuple[pd.Timestamp, dict[str, float]]], list[dict]]:
    """
    Devuelve lista (fecha, posiciones despues del archivo) y detalle por archivo.
    """
    initial_positions = dict(initial_positions or {})
    pos = dict(initial_positions)
    events: list[tuple[pd.Timestamp, dict[str, float]]] = []
    details: list[dict] = []

    for d, path in files:
        try:
            orders = load_orders_from_excel(path)
        except Exception as exc:
            details.append(
                {
                    "fecha": d,
                    "path": str(path),
                    "error": str(exc),
                    "coste_eur": np.nan,
                    "notional_eur": np.nan,
                    "n_ordenes": 0,
                }
            )
            continue
        cost = estimate_orders_cost_eur(orders)
        notional = estimate_notional_eur(orders)
        pos = apply_orders_to_positions(pos, orders)
        events.append((pd.Timestamp(d), dict(pos)))
        details.append(
            {
                "fecha": d,
                "path": str(path),
                "error": "",
                "coste_eur": cost,
                "notional_eur": notional,
                "n_ordenes": len(orders),
            }
        )

    return events, details


def _position_at_date(
    t: pd.Timestamp,
    event_ts: np.ndarray,
    snapshots: list[dict[str, float]],
    initial: dict[str, float],
) -> dict[str, float]:
    tv = pd.Timestamp(t).value
    idx = int(np.searchsorted(event_ts, tv, side="right") - 1)
    if idx < 0:
        return dict(initial)
    return dict(snapshots[idx])


def build_nav_series(
    prices: pd.DataFrame,
    events: list[tuple[pd.Timestamp, dict[str, float]]],
    initial_positions: dict[str, float],
) -> pd.Series:
    """NAV diario (mark-to-market) con posiciones constantes entre rebalanceos."""
    if not len(prices.index):
        return pd.Series(dtype=float)

    event_ts = np.array([pd.Timestamp(e[0]).value for e in events], dtype=np.int64)
    snapshots = [e[1] for e in events]

    navs = []
    for t in prices.index:
        pos = _position_at_date(t, event_ts, snapshots, initial_positions)
        row = prices.loc[t]
        navs.append(portfolio_value_eur(pos, row))
    return pd.Series(navs, index=prices.index, name="NAV_cartera")


def build_benchmark_series(
    prices_bm: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_value: float,
) -> pd.Series:
    s = prices_bm.loc[start:end].astype(float).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s0 = float(s.iloc[0])
    if s0 <= 0:
        return pd.Series(dtype=float)
    return base_value * (s / s0)


def run_live_portfolio_report(
    results_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_positions_path: str | Path | None = None,
    output_subdir: str | None = None,
) -> dict:
    """
    Lee todos los ``operaciones_rebalanceo_YYYY-MM-DD.xlsx`` en ``results_dir``,
    reconstruye posiciones, descarga precios y escribe informe Excel + CSVs.

    ``initial_positions_path``: Excel opcional con el estado *antes* del primer
    archivo de operaciones (misma hoja que Posiciones).

    ``start_date``: acota inicio (por defecto: primera fecha de operaciones o
    ``PORTFOLIO_LIVE_START_DATE`` en config).
    """
    results_dir = Path(results_dir or getattr(cfg, "PORTFOLIO_LIVE_RESULTS_DIR", "results"))
    output_subdir = output_subdir or getattr(cfg, "PORTFOLIO_LIVE_INFORME_SUBDIR", "informe_cartera_vivo")
    out_dir = results_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    files = discover_operaciones_files(results_dir)
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos operaciones_rebalanceo_YYYY-MM-DD.xlsx en {results_dir}"
        )

    ini_path = initial_positions_path
    if ini_path is None:
        ini_path = getattr(cfg, "PORTFOLIO_LIVE_INITIAL_POSITIONS", None)
    initial: dict[str, float] = {}
    if ini_path:
        initial = load_positions_from_excel(Path(ini_path))

    cfg_start = getattr(cfg, "PORTFOLIO_LIVE_START_DATE", None)
    first_file_date = files[0][0]
    if start_date is None:
        start_date = cfg_start if cfg_start else first_file_date.strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start_date)
    if start_ts > first_file_date and not initial:
        print(
            "[informe] Aviso: start_date es posterior al primer archivo de operaciones "
            "y no hay posiciones iniciales; el NAV puede ser 0 hasta esa fecha."
        )

    events, op_details = rebuild_events_from_operaciones(files, initial_positions=initial)

    all_tickers = set(initial.keys())
    for _, snap in events:
        all_tickers |= set(snap.keys())
    all_tickers = sorted(all_tickers)

    bm_spy = getattr(cfg, "BENCHMARK_TICKER", "SPY")
    bm_msci = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
    tickers_dl = sorted(set(all_tickers) | {bm_spy, bm_msci})

    dl_start = (start_ts - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    market = download_market_data(start_date=dl_start, end_date=end_date, tickers=tickers_dl)
    prices = market["prices"].sort_index()
    prices = prices.loc[prices.index >= start_ts]
    prices = prices.loc[prices.index <= pd.Timestamp(end_date)]

    nav = build_nav_series(prices, events, initial)
    pos = nav[nav > 1e-6]
    if pos.empty:
        print(
            "[informe] Aviso: NAV ~0 en el rango (sin posiciones o sin precios). "
            "Metricas pueden ser triviales."
        )
        wealth = nav
    else:
        # Serie de patrimonio en EUR (mark-to-market); recorte de dias previos al primer NAV > 0
        wealth = nav.loc[pos.index[0] :].copy()

    bench_rows = []
    for name, col in [("Benchmark SPY", bm_spy), ("Benchmark MSCI World (URTH)", bm_msci)]:
        if col not in prices.columns:
            continue
        bm = build_benchmark_series(
            prices[col],
            wealth.index[0],
            wealth.index[-1],
            float(wealth.iloc[0]),
        )
        bm = bm.reindex(wealth.index).ffill()
        bench_rows.append((name, bm))

    metrics_port = _compute_metrics(wealth)
    metrics_fmt = _format_metrics(metrics_port)

    comp = pd.DataFrame(
        {
            "Metrica": list(metrics_fmt.keys()),
            "Cartera": list(metrics_fmt.values()),
        }
    )
    if bench_rows:
        for name, bm in bench_rows:
            if bm is None or bm.dropna().empty or len(bm.dropna()) < 2:
                continue
            m = _compute_metrics(bm.dropna())
            mf = _format_metrics(m)
            comp[name] = list(mf.values())

    total_cost = float(sum(d.get("coste_eur", 0) or 0 for d in op_details if not d.get("error")))
    total_notional = float(sum(d.get("notional_eur", 0) or 0 for d in op_details if not d.get("error")))

    resumen = pd.DataFrame(
        [
            {"Campo": "Directorio operaciones", "Valor": str(results_dir.resolve())},
            {"Campo": "Archivos operaciones", "Valor": len(files)},
            {"Campo": "Coste total estimado (comisiones)", "Valor": f"{total_cost:,.2f} EUR"},
            {"Campo": "Notional acumulado (|orden|)", "Valor": f"{total_notional:,.2f} EUR"},
            {"Campo": "Rango fechas precios", "Valor": f"{wealth.index[0].date()} — {wealth.index[-1].date()}"},
        ]
    )

    serie = pd.DataFrame({"Patrimonio_EUR": wealth})
    for name, bm in bench_rows:
        serie[name] = bm.reindex(wealth.index)

    op_df = pd.DataFrame(op_details)

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    xlsx_path = out_dir / f"informe_cartera_vivo_{today_str}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        comp.to_excel(writer, sheet_name="Metricas", index=False)
        serie.to_excel(writer, sheet_name="Serie_patrimonio")
        op_df.to_excel(writer, sheet_name="Operaciones_archivos", index=False)
        pd.DataFrame(
            [{"fecha": d.isoformat(), "n_tickers": len(s)} for d, s in events],
            columns=["fecha", "n_tickers"],
        ).to_excel(writer, sheet_name="Rebalances", index=False)

    pd.DataFrame({"NAV_EUR": nav}).to_csv(out_dir / "nav_cartera_mtm_completo.csv", index_label="fecha")
    serie.to_csv(out_dir / "patrimonio_vs_benchmark.csv")
    op_df.to_csv(out_dir / "detalle_operaciones_historico.csv", index=False)

    plot_path = None
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(serie.index, serie["Patrimonio_EUR"], label="Cartera", color="C0")
        for c in serie.columns:
            if c == "Patrimonio_cartera":
                continue
            ax.plot(serie.index, serie[c], label=c, alpha=0.75)
        ax.set_title("Patrimonio vivo vs benchmarks")
        ax.set_ylabel("EUR (escala comun)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plot_path = out_dir / f"patrimonio_vs_benchmark_{today_str}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception:
        plot_path = None

    return {
        "output_dir": str(out_dir),
        "excel_path": str(xlsx_path),
        "wealth_series": wealth,
        "metrics": metrics_port,
        "metrics_formatted": metrics_fmt,
        "operaciones_files": len(files),
        "coste_total_estimado_eur": total_cost,
        "notional_acumulado_eur": total_notional,
        "plot_path": str(plot_path) if plot_path else None,
        "events": events,
        "operaciones_detail": op_details,
    }
