"""
Motor de backtesting stateful.

1. Descarga datos una vez para todo el periodo (+ lookback de 18 meses)
2. Para cada fecha de revision (fin de mes):
   a. Corta datos hasta esa fecha (sin look-ahead)
   b. Ejecuta pipeline: senal compuesta -> BL -> Merton -> Davis-Norman
   c. Si aplica politica (portfolio.rebalance_policy.should_apply_rebalance_after_dn):
      actualiza posiciones con dn['final_weights_full'] (+ costes)
3. Calcula wealth diario, pesos diarios, metricas vs SP500 y MSCI World
4. Atribucion EUR aproximada por ETF (peso lag x retorno x patrimonio)
5. Guarda resultados en output_dir/

Misma regla de rebalanceo que main.py / registrador y walkforward (run_backtest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Data.data_loader import download_market_data
from Data.universe import get_defensive_ticker
from portfolio.rebalance_policy import (
    positions_from_weights_full,
    should_apply_rebalance_after_dn,
)
import config as cfg


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _compute_portfolio_value(positions: dict, prices_series: pd.Series) -> float:
    value = 0.0
    for ticker, qty in positions.items():
        price = prices_series.get(ticker, np.nan)
        if not pd.isna(price) and qty != 0:
            value += qty * price
    return value


def _compute_current_weights(positions: dict, prices_series: pd.Series) -> dict:
    total = _compute_portfolio_value(positions, prices_series)
    if total <= 0:
        return {}
    weights = {}
    for ticker, qty in positions.items():
        price = prices_series.get(ticker, np.nan)
        if not pd.isna(price) and qty != 0:
            weights[ticker] = qty * price / total
    return weights


def _turnover_cost_eur(
    old_positions: dict,
    new_positions: dict,
    prices_series: pd.Series,
    transaction_costs: dict[str, float],
) -> float:
    """
    Coste total de transacción al estilo registrador: ct * |Δcantidad| * precio
    (compra paga más, venta cobra menos; en notional equivale a ct por lado).
    """
    all_tickers = set(old_positions) | set(new_positions)
    cost = 0.0
    for ticker in all_tickers:
        oq = float(old_positions.get(ticker, 0.0))
        nq = float(new_positions.get(ticker, 0.0))
        dq = nq - oq
        if abs(dq) < 1e-12:
            continue
        price = prices_series.get(ticker, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        ct = float(transaction_costs.get(ticker, 0.0))
        cost += abs(dq) * price * ct
    return cost


def _execute_rebalance(
    final_weights_full: dict,
    portfolio_value: float,
    prices_series: pd.Series,
    xeon_ticker: str,
) -> dict:
    """Pesos → posiciones. Misma implementacion que portfolio.rebalance_policy."""
    return positions_from_weights_full(
        final_weights_full, portfolio_value, prices_series, xeon_ticker
    )


def _execute_rebalance_with_costs(
    final_weights_full: dict,
    portfolio_value: float,
    prices_series: pd.Series,
    xeon_ticker: str,
    current_positions: dict,
    transaction_costs: dict[str, float] | None,
) -> tuple[dict, float]:
    """
    Misma lógica que _execute_rebalance, pero descuenta el coste de turnover
    del presupuesto (una iteración: V' = V - coste(old, provisión con V)).
    Devuelve (posiciones finales, coste en EUR).
    """
    if not transaction_costs:
        pos = _execute_rebalance(final_weights_full, portfolio_value, prices_series, xeon_ticker)
        return pos, 0.0

    provisional = _execute_rebalance(
        final_weights_full, portfolio_value, prices_series, xeon_ticker
    )
    cost = _turnover_cost_eur(current_positions, provisional, prices_series, transaction_costs)
    net_budget = max(portfolio_value - cost, 0.0)
    final_pos = _execute_rebalance(
        final_weights_full, net_budget, prices_series, xeon_ticker
    )
    return final_pos, cost


def _commission_rates_report(
    transaction_costs: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tabla para Excel: ct por ticker tal como lo usa el motor, y el mismo dato
    en % sobre el nominal y en puntos básicos (1 bp = 0,01%).
    """
    if not transaction_costs:
        empty = pd.DataFrame(
            columns=[
                "Ticker",
                "ct (fraccion motor)",
                "Comision_pct_nominal",
                "Comision_bps",
            ]
        )
        notes = pd.DataFrame(
            {"Nota": ["No hay transaction_costs en market_data (coste 0 en backtest)."]}
        )
        return empty, notes

    rows = []
    for ticker in sorted(transaction_costs.keys()):
        ct = float(transaction_costs[ticker])
        rows.append(
            {
                "Ticker": ticker,
                "ct (fraccion motor)": ct,
                "Comision_pct_nominal": round(ct * 100.0, 6),
                "Comision_bps": round(ct * 10000.0, 4),
            }
        )
    rates = pd.DataFrame(rows)
    notes = pd.DataFrame(
        {
            "Concepto": [
                "Columnas Comision_pct_nominal y Comision_bps",
                "Origen de ct",
                "Uso en backtest",
                "Registrador",
                "Origen tasas",
            ],
            "Detalle": [
                "Comision_pct_nominal = ct*100 (ej. 0.15 => 0,15% del nominal por operacion y lado). "
                "Comision_bps = ct*10000 (1 bp = 0,01%).",
                "Un valor por ticker: download_market_data -> compute_transaction_costs_for_download (Excel + fallback).",
                "Coste EUR = suma(|Delta_cantidad| * Precio * ct) por operacion y activo.",
                "Misma tasa ct: compra ~P*(1+ct), venta ~P*(1-ct); en notional equivale a ct por lado.",
                "Excel tabla Data/comisiones_etfs.xlsx (u horizontal si config); sin fila -> TX_COST_PER_SIDE (MIN/MAX).",
            ],
        }
    )
    return rates, notes


def _compute_attribution_eur(
    all_prices: pd.DataFrame,
    weights_df: pd.DataFrame,
    wealth_strategy: pd.Series,
) -> pd.DataFrame:
    """
    Contribucion diaria en EUR aproximada: w_{t-1,i} * r_{t,i} * V_{t-1}.
    La suma por fila deberia aproximar el cambio de patrimonio del dia.
    """
    cols = [c for c in weights_df.columns if c in all_prices.columns]
    if not cols:
        return pd.DataFrame(index=wealth_strategy.index)

    idx = wealth_strategy.index.intersection(weights_df.index).intersection(all_prices.index)
    W = wealth_strategy.reindex(idx).ffill()
    w = weights_df[cols].reindex(idx).ffill().fillna(0.0)
    pr = all_prices[cols].reindex(idx).ffill()
    rets = pr.pct_change()
    w_lag = w.shift(1)
    V_lag = W.shift(1)
    contrib = w_lag.mul(rets, fill_value=0.0).mul(V_lag, axis=0)
    return contrib


def _compute_metrics(wealth_series: pd.Series) -> dict:
    """Metricas de rendimiento como floats (para comparacion programatica)."""
    if len(wealth_series) < 2:
        return {"Total Return": 0.0, "CAGR": 0.0, "Volatility": 0.0, "Sharpe": 0.0, "Max Drawdown": 0.0}

    returns = wealth_series.pct_change().dropna()
    total_return = wealth_series.iloc[-1] / wealth_series.iloc[0] - 1
    n_days = (wealth_series.index[-1] - wealth_series.index[0]).days
    n_years = max(n_days / 365.25, 1e-6)
    cagr = (1 + total_return) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    avg_rf = np.mean([cfg.get_risk_free_rate(d) for d in wealth_series.index])
    sharpe = (cagr - avg_rf) / vol if vol > 1e-8 else 0.0

    cummax = wealth_series.cummax()
    drawdown = (wealth_series - cummax) / cummax
    max_dd = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }


def _format_metrics(metrics: dict) -> dict:
    """Formatea metricas para display."""
    return {
        "Total Return": f"{metrics['Total Return']:.2%}",
        "CAGR": f"{metrics['CAGR']:.2%}",
        "Volatility": f"{metrics['Volatility']:.2%}",
        "Sharpe": f"{metrics['Sharpe']:.2f}",
        "Max Drawdown": f"{metrics['Max Drawdown']:.2%}",
    }


# ============================================================
# FUNCION PRINCIPAL
# ============================================================

def run_backtest(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    freq: str | None = None,
    initial_capital: float | None = None,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:
    """
    Backtest stateful completo.

    Args:
        verbose: si False, suprime output por fecha (para optimizacion)
    """
    from main import run_pipeline

    if freq is None:
        freq = cfg.REBALANCE_FREQ
    if initial_capital is None:
        initial_capital = cfg.INITIAL_CAPITAL

    xeon_ticker = get_defensive_ticker()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Descarga unica ---
    lookback_start = (pd.Timestamp(start_date) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    if verbose:
        print(f"[Backtest] Descargando datos {lookback_start} -> {end_date} ...")
    market_data = download_market_data(start_date=lookback_start, end_date=end_date)

    spy_prices = None
    try:
        spy_data = download_market_data(
            start_date=lookback_start, end_date=end_date, tickers=[cfg.BENCHMARK_TICKER],
        )
        spy_prices = spy_data["prices"][cfg.BENCHMARK_TICKER]
    except (ValueError, KeyError):
        if verbose:
            print(f"[Backtest] No se pudo descargar {cfg.BENCHMARK_TICKER}")

    msci_ticker = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
    msci_prices = None
    try:
        msci_data = download_market_data(
            start_date=lookback_start, end_date=end_date, tickers=[msci_ticker],
        )
        msci_prices = msci_data["prices"][msci_ticker]
    except (ValueError, KeyError):
        if verbose:
            print(f"[Backtest] No se pudo descargar benchmark MSCI World ({msci_ticker})")

    metadata_by_ticker = {item["ticker"]: item for item in market_data["metadata"]}
    risk_tickers = [
        t for t in market_data["tickers"]
        if metadata_by_ticker.get(t, {}).get("role") != "defensive"
    ]
    all_tickers = risk_tickers + [xeon_ticker]

    all_prices = market_data["prices"].ffill()
    all_returns = market_data["returns"]
    transaction_costs = market_data.get("transaction_costs") or {}

    # --- Review dates → trading dates reales ---
    raw_review = pd.date_range(start=start_date, end=end_date, freq=freq)
    trading_idx = all_prices.index
    actual_review = []
    for rd in raw_review:
        valid = trading_idx[trading_idx <= rd]
        if len(valid) > 0:
            actual_review.append(valid[-1])
    actual_review = sorted(set(actual_review))

    # --- Phase 1: Pipeline en cada fecha de revision ---
    positions: dict = {}
    portfolio_value = float(initial_capital)
    rebalance_events: list[tuple] = []
    decision_rows: list[dict] = []
    n_rebalances = 0
    total_transaction_costs = 0.0

    if verbose:
        print(f"[Backtest] {len(actual_review)} fechas de revision")
        if transaction_costs:
            print("[Backtest] Costes de transacción activos (Excel Data + fallback TX_COST_*)")
        else:
            print("[Backtest] Aviso: sin transaction_costs en market_data")

    for review_date in actual_review:
        trade_cost = 0.0
        cp = all_prices.loc[review_date]

        if positions:
            portfolio_value = _compute_portfolio_value(positions, cp)
            current_weights = _compute_current_weights(positions, cp)
        else:
            current_weights = {}

        returns_slice = all_returns[risk_tickers].loc[:review_date].dropna(how="all")
        prices_slice = all_prices[risk_tickers].loc[:review_date].dropna(how="all")

        if len(returns_slice) < 60:
            if verbose:
                print(f"  [{review_date.date()}] Datos insuficientes ({len(returns_slice)} dias), skip")
            continue

        try:
            result = run_pipeline(
                returns_slice, prices_slice,
                current_weights=current_weights,
                review_date=review_date,
                categoria_por_ticker=cfg.ETF_UNIVERSE,
            )
        except Exception as e:
            if verbose:
                print(f"  [{review_date.date()}] Error en pipeline: {e}")
            continue

        dn = result["dn_result"]
        merton = result["merton_result"]
        rebalanced = False

        if should_apply_rebalance_after_dn(dn, positions if positions else None):
            fw = dict(dn["final_weights_full"])
            positions, trade_cost = _execute_rebalance_with_costs(
                fw,
                portfolio_value,
                cp,
                xeon_ticker,
                current_positions=positions if positions else {},
                transaction_costs=transaction_costs,
            )
            total_transaction_costs += trade_cost
            portfolio_value = _compute_portfolio_value(positions, cp)
            rebalance_events.append((review_date, dict(positions)))
            n_rebalances += 1
            rebalanced = True

        selected = merton.get("selected_etfs", [])
        if verbose:
            tc_show = f" | Coste={trade_cost:,.0f}" if rebalanced and trade_cost > 0 else ""
            print(
                f"  [{review_date.date()}] Value={portfolio_value:,.0f} | "
                f"Rebal={'SI' if rebalanced else 'NO'}{tc_show} | "
                f"ETFs={selected[:5]}{'...' if len(selected) > 5 else ''} | "
                f"XEON={dn['weight_xeon']:.1%} | "
                f"Regime={merton.get('regime', '?')}"
            )

        decision_rows.append({
            "Date": review_date,
            "Portfolio Value": round(portfolio_value, 2),
            "Rebalance": rebalanced,
            "Trade Cost EUR": round(trade_cost, 2),
            "Selected ETFs": ", ".join(selected),
            "N ETFs": len(selected),
            "Weight XEON": round(dn["weight_xeon"], 4),
            "Regime": merton.get("regime", "normal"),
            "Reason": dn.get("reason", ""),
        })

    # --- Phase 2: Wealth diario + pesos diarios ---
    if verbose:
        print("[Backtest] Calculando wealth y pesos diarios...")
    bt_start = pd.Timestamp(start_date)
    bt_trading_dates = trading_idx[trading_idx >= bt_start]

    current_pos: dict = {}
    event_idx = 0
    wealth_rows: list[dict] = []
    weights_rows: list[dict] = []

    for date in bt_trading_dates:
        while event_idx < len(rebalance_events) and rebalance_events[event_idx][0] <= date:
            current_pos = rebalance_events[event_idx][1]
            event_idx += 1

        if current_pos:
            pv = _compute_portfolio_value(current_pos, all_prices.loc[date])
            w = _compute_current_weights(current_pos, all_prices.loc[date])
        else:
            pv = float(initial_capital)
            w = {}

        wealth_rows.append({"Date": date, "Strategy": pv})

        weight_row = {"Date": date}
        for ticker in all_tickers:
            weight_row[ticker] = round(w.get(ticker, 0.0), 6)
        weights_rows.append(weight_row)

    wealth_df = pd.DataFrame(wealth_rows).set_index("Date")
    weights_df = pd.DataFrame(weights_rows).set_index("Date")

    # SP500 benchmark alineado
    if spy_prices is not None and len(spy_prices) > 0:
        spy_bt = spy_prices.loc[bt_start:].ffill()
        if len(spy_bt) > 0:
            spy_base = spy_bt.iloc[0]
            spy_aligned = spy_bt.reindex(wealth_df.index, method="ffill")
            wealth_df["SP500"] = initial_capital * spy_aligned / spy_base
        else:
            wealth_df["SP500"] = float(initial_capital)
    else:
        wealth_df["SP500"] = float(initial_capital)

    msci_col = f"MSCI_World ({msci_ticker})"
    if msci_prices is not None and len(msci_prices) > 0:
        msci_bt = msci_prices.loc[bt_start:].ffill()
        if len(msci_bt) > 0:
            msci_base = msci_bt.iloc[0]
            msci_aligned = msci_bt.reindex(wealth_df.index, method="ffill")
            wealth_df[msci_col] = initial_capital * msci_aligned / msci_base
        else:
            wealth_df[msci_col] = float(initial_capital)
    else:
        wealth_df[msci_col] = float(initial_capital)

    # --- Atribucion EUR por ETF (diaria y acumulada) ---
    attribution_daily = _compute_attribution_eur(all_prices, weights_df, wealth_df["Strategy"])
    attribution_cum = attribution_daily.cumsum()

    # --- Metricas ---
    strategy_metrics = _compute_metrics(wealth_df["Strategy"])
    sp500_metrics = _compute_metrics(wealth_df["SP500"])
    msci_metrics = _compute_metrics(wealth_df[msci_col])
    strategy_fmt = _format_metrics(strategy_metrics)
    sp500_fmt = _format_metrics(sp500_metrics)
    msci_fmt = _format_metrics(msci_metrics)

    cost_pct_initial = (
        total_transaction_costs / initial_capital * 100 if initial_capital > 0 else 0.0
    )
    metrics_comparison = pd.DataFrame({
        "Metric": list(strategy_fmt.keys())
        + ["Rebalances", "Total transaction costs (EUR)", "Costs (% initial capital)"],
        "Strategy": list(strategy_fmt.values())
        + [
            str(n_rebalances),
            f"{total_transaction_costs:,.2f}",
            f"{cost_pct_initial:.2f}%",
        ],
        "SP500 (B&H)": list(sp500_fmt.values()) + ["-", "-", "-"],
        msci_col: list(msci_fmt.values()) + ["-", "-", "-"],
    })

    if verbose:
        print(f"\n{'=' * 60}")
        print("METRICAS FINALES")
        print("=" * 60)
        print(metrics_comparison.to_string(index=False))
        print("=" * 60)

    # --- Guardar resultados ---
    decisions_df = pd.DataFrame(decision_rows)
    commission_rates_df, commission_notes_df = _commission_rates_report(transaction_costs)

    wealth_df.reset_index().to_csv(output_path / "wealth_history.csv", index=False)
    weights_df.reset_index().to_csv(output_path / "weights_history.csv", index=False)
    decisions_df.to_csv(output_path / "backtest_resumen.csv", index=False)
    _ad = attribution_daily.copy()
    _ad.index.name = "Date"
    _ad.to_csv(output_path / "attribution_daily_eur.csv", encoding="utf-8-sig")
    _ac = attribution_cum.copy()
    _ac.index.name = "Date"
    _ac.to_csv(output_path / "attribution_cumulative_eur.csv", encoding="utf-8-sig")
    if not commission_rates_df.empty:
        commission_rates_df.to_csv(
            output_path / "commission_rates_by_etf.csv", index=False, encoding="utf-8-sig"
        )

    def _write_metrics_excel(path: Path) -> None:
        with pd.ExcelWriter(path) as writer:
            metrics_comparison.to_excel(writer, sheet_name="Comparison", index=False)
            decisions_df.to_excel(writer, sheet_name="Decisions", index=False)
            wealth_df.reset_index().to_excel(writer, sheet_name="Wealth", index=False)
            commission_rates_df.to_excel(writer, sheet_name="Commission_pct_by_ETF", index=False)
            commission_notes_df.to_excel(writer, sheet_name="Commission_notes", index=False)

    try:
        _write_metrics_excel(output_path / "Metrics.xlsx")
    except PermissionError:
        if verbose:
            print("[Backtest] Metrics.xlsx bloqueado, guardando como Metrics_backup.xlsx")
        _write_metrics_excel(output_path / "Metrics_backup.xlsx")

    if verbose:
        print(f"[Backtest] Resultados guardados en {output_path}/")
        if not commission_rates_df.empty:
            print(
                "[Backtest] Comisiones por ETF: hoja 'Commission_pct_by_ETF' en Metrics.xlsx "
                f"y {output_path / 'commission_rates_by_etf.csv'}"
            )

    return {
        "wealth": wealth_df,
        "weights": weights_df,
        "decisions": decisions_df,
        "attribution_daily": attribution_daily,
        "attribution_cumulative": attribution_cum,
        "metrics_comparison": metrics_comparison,
        "strategy_metrics": strategy_metrics,
        "sp500_metrics": sp500_metrics,
        "msci_world_metrics": msci_metrics,
        "msci_benchmark_ticker": msci_ticker,
        "msci_wealth_column": msci_col,
        "n_rebalances": n_rebalances,
        "total_transaction_costs": total_transaction_costs,
        "commission_rates": commission_rates_df,
        "output_dir": str(output_path),
    }
