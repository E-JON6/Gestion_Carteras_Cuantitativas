"""
Pipeline principal — filosofia contrarian.

En caidas de mercado:
  1. VIEW_SCALE se amplifica (comprar mas agresivamente)
  2. DN bands se ensanchan (no vender en panico)
  3. Merton siempre al 100% invertido (sin cap de regimen)

run_single() usa la misma politica de rebalanceo que backtest/engine (portfolio.rebalance_policy).

Salidas tipicas en results/: operaciones_rebalanceo_{fecha}.xlsx,
posiciones_post_rebalanceo_{fecha}.xlsx, posiciones_post_rebalanceo_ultimo.xlsx,
historial_ejecuciones.csv (append).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from Data.data_loader import download_market_data
from metricas.composite_signal import compute_composite_signal
from Models.black_litterman import run_black_litterman
from Models.merton import run_merton_v0
from Models.davis_norman_fake import run_davis_norman_fake_v0
from portfolio.audit_log import append_ejecucion_row
from portfolio.email_operaciones import send_operaciones_excel
from portfolio.positions_io import (
    ensure_positions_template,
    load_positions_from_excel,
    portfolio_value_eur,
    save_positions_to_excel,
    weights_from_positions,
)
from portfolio.registrador import run_registrador_v0
import config as cfg


def _get_risk_tickers(market_data: dict) -> list[str]:
    metadata_by_ticker = {item["ticker"]: item for item in market_data["metadata"]}
    return [
        ticker for ticker in market_data["tickers"]
        if metadata_by_ticker.get(ticker, {}).get("role") != "defensive"
    ]


def _detect_regime_early(returns_df) -> str:
    """
    Regimen multi-factor ANTES de BL (VIEW_SCALE y bandas DN).

    - vol_cross: vol media cross-sectional anualizada (como antes).
    - ew_mdd: max drawdown de una cartera equiponderada (proxy de mercado amplio).
    - avg_corr: correlacion media entre pares (estres / risk-on).
    Cualquier disparador fuerte -> crisis; si no, caution; si no, normal.
    """
    raw_vol = float(returns_df.std().mean() * np.sqrt(252))

    lb = min(int(getattr(cfg, "REGIME_LOOKBACK_DD", 252)), len(returns_df))
    ew_mdd = 0.0
    if lb >= 20:
        ew_ret = returns_df.tail(lb).mean(axis=1)
        w = (1.0 + ew_ret).cumprod()
        dd_series = w / w.cummax() - 1.0
        ew_mdd = float(-dd_series.min()) if len(dd_series) else 0.0

    avg_corr = 0.0
    cl = min(int(getattr(cfg, "REGIME_CORR_LOOKBACK", 63)), len(returns_df))
    if cl >= 20:
        cmat = returns_df.tail(cl).corr().values
        m = cmat.shape[0]
        if m > 1:
            tri = np.triu_indices(m, k=1)
            avg_corr = float(np.nanmean(cmat[tri]))

    mdd_c = float(getattr(cfg, "REGIME_EW_MDD_CRISIS", 0.28))
    mdd_a = float(getattr(cfg, "REGIME_EW_MDD_CAUTION", 0.16))
    cr_c = float(getattr(cfg, "REGIME_AVG_CORR_CRISIS", 0.52))
    cr_a = float(getattr(cfg, "REGIME_AVG_CORR_CAUTION", 0.38))

    if raw_vol > cfg.VOL_CRISIS_THR or ew_mdd > mdd_c or avg_corr > cr_c:
        return "crisis"
    if raw_vol > cfg.VOL_CAUTION_THR or ew_mdd > mdd_a or avg_corr > cr_a:
        return "caution"
    return "normal"


def run_pipeline(
    returns_df,
    prices_df,
    current_weights=None,
    review_date=None,
    risk_free_rate=None,
    categoria_por_ticker=None,
):
    if current_weights is None:
        current_weights = {}
    if categoria_por_ticker is None:
        categoria_por_ticker = cfg.ETF_UNIVERSE

    if risk_free_rate is None:
        if review_date is not None:
            risk_free_rate = cfg.get_risk_free_rate(review_date)
        else:
            risk_free_rate = cfg.BL_RF

    valid_tickers = [
        t for t in returns_df.columns
        if returns_df[t].dropna().shape[0] >= 60
    ]
    returns_clean = returns_df[valid_tickers]
    prices_clean = prices_df[[t for t in valid_tickers if t in prices_df.columns]]

    if returns_clean.empty:
        raise ValueError("No hay ETFs con datos suficientes para el pipeline.")

    # --- Regimen anticipado (para ajustes contrarian) ---
    regime_early = _detect_regime_early(returns_clean)

    # Boost VIEW_SCALE en crisis (contrarian: comprar la caida)
    effective_view_scale = cfg.VIEW_SCALE
    if regime_early == "crisis":
        effective_view_scale *= cfg.CRISIS_VIEW_BOOST
    elif regime_early == "caution":
        effective_view_scale *= cfg.CAUTION_VIEW_BOOST

    # 1) Senal compuesta
    scores = compute_composite_signal(
        returns_clean, prices_clean,
        momentum_window=cfg.COMPOSITE_MOMENTUM_WINDOW,
        momentum_skip=cfg.COMPOSITE_MOMENTUM_SKIP,
        reversal_window=cfg.COMPOSITE_REVERSAL_WINDOW,
        trend_window=cfg.COMPOSITE_TREND_WINDOW,
        vol_window=cfg.COMPOSITE_VOL_WINDOW,
        drawdown_window=cfg.COMPOSITE_DRAWDOWN_WINDOW,
        weights=cfg.COMPOSITE_WEIGHTS,
        ticker_categories=cfg.ETF_UNIVERSE,
        category_weights=cfg.CATEGORY_SIGNAL_WEIGHTS,
    )

    # 2) Black-Litterman con VIEW_SCALE ajustado por regimen
    bl_result = run_black_litterman(
        returns_clean, scores,
        rf=risk_free_rate,
        delta=cfg.BL_DELTA,
        tau=cfg.BL_TAU,
        view_scale=effective_view_scale,
        short_window=cfg.COV_SHORT_WINDOW,
        long_window=cfg.COV_LONG_WINDOW,
        blend_alpha=cfg.COV_BLEND_ALPHA,
        ewma_lambda=cfg.EWMA_LAMBDA,
        shrinkage=cfg.COV_SHRINKAGE,
        review_date=review_date,
        prior_weights_mode=getattr(cfg, "BL_PRIOR_WEIGHTS_MODE", "equal"),
    )

    # 3) Merton (siempre 100% invertido)
    sigma_mercado = float(np.sqrt(np.diag(bl_result["Sigma"])).mean())
    merton_result = run_merton_v0(
        bl_result,
        risk_free_rate=risk_free_rate,
        sigma_mercado=sigma_mercado,
        categoria_por_ticker=categoria_por_ticker,
        gamma=cfg.MERTON_GAMMA,
    )

    # 4) Davis-Norman con bandas ajustadas por regimen
    dn_band = cfg.DN_BAND
    if regime_early == "crisis":
        dn_band *= cfg.DN_CRISIS_MULT
    elif regime_early == "caution":
        dn_band *= cfg.DN_CAUTION_MULT

    dn_result = run_davis_norman_fake_v0(
        current_weights,
        merton_result["weights"],
        band=dn_band,
        min_band=cfg.DN_MIN_BAND,
    )

    return {
        "scores": scores,
        "bl_result": bl_result,
        "merton_result": merton_result,
        "dn_result": dn_result,
        "risk_free_rate": risk_free_rate,
        "regime_early": regime_early,
        "effective_view_scale": effective_view_scale,
    }


def run_single(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    current_positions: dict | None = None,
) -> dict:
    """
    Descarga datos hasta end_date (por defecto hoy), ejecuta el pipeline en el
    ultimo dia disponible y genera el Excel de operativa (registrador).

    Posiciones reales (titulos):
      - Si ``current_positions`` es None, se lee ``cfg.POSITIONS_EXCEL_PATH``
        (columnas Ticker + Cantidad). Si no existe y
        ``CREATE_POSITIONS_TEMPLATE_IF_MISSING``, se crea una plantilla vacia.
      - El pipeline (Davis-Norman) recibe **pesos** actuales derivados de esas
        cantidades y precios del ultimo dia; el registrador usa **cantidades**
        y NAV de mercado para los deltas en el Excel de operaciones.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    market_data = download_market_data(start_date=start_date, end_date=end_date)
    last_dt = market_data["prices"].index[-1]
    full_prices_row = market_data["prices"].loc[last_dt]

    risk_tickers = _get_risk_tickers(market_data)
    returns_df = market_data["returns"][risk_tickers]
    prices_df = market_data["prices"][risk_tickers]

    positions_path = Path(getattr(cfg, "POSITIONS_EXCEL_PATH", "results/posiciones_cartera.xlsx"))
    positions_source = "parametro"

    if current_positions is not None:
        cw_qty = dict(current_positions)
    else:
        if positions_path.exists():
            cw_qty = load_positions_from_excel(positions_path)
            positions_source = str(positions_path)
        else:
            cw_qty = {}
            if getattr(cfg, "CREATE_POSITIONS_TEMPLATE_IF_MISSING", True):
                ensure_positions_template(positions_path)
                positions_source = f"plantilla creada en {positions_path}"

    cleaned_qty: dict[str, float] = {}
    for t, q in cw_qty.items():
        p = full_prices_row.get(t, np.nan)
        if pd.notna(p):
            cleaned_qty[t] = float(q)
        else:
            print(f"[posiciones] Aviso: ticker {t!r} sin precio en el ultimo dia — excluido del NAV.")
    cw_qty = cleaned_qty

    nav = portfolio_value_eur(cw_qty, full_prices_row)
    initial = float(getattr(cfg, "INITIAL_CAPITAL", 100_000))
    if nav <= 0 and not cw_qty:
        nav = initial
    elif nav <= 0 and cw_qty:
        print("[posiciones] Aviso: NAV<=0 con posiciones; se usa INITIAL_CAPITAL para el registrador.")
        nav = initial

    cw_weights = weights_from_positions(cw_qty, full_prices_row) if cw_qty else {}

    result = run_pipeline(
        returns_df, prices_df,
        current_weights=cw_weights,
        review_date=last_dt,
    )

    date_str = pd.Timestamp(last_dt).strftime("%Y-%m-%d")
    out_template = getattr(cfg, "REGISTRADOR_OUTPUT_TEMPLATE", "results/operaciones_rebalanceo_{date}.xlsx")
    reg_path = out_template.format(date=date_str)

    reg = run_registrador_v0(
        result["dn_result"],
        market_data,
        current_positions=cw_qty,
        total_value=float(nav),
        output_path=reg_path,
        only_when_engine_would_trade=getattr(
            cfg, "REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE", True
        ),
    )

    post_rebalance_path = None
    ultimo_snapshot_path = None
    if getattr(cfg, "SAVE_SUGGESTED_POSITIONS", True) and reg.get("updated_positions") is not None:
        tpl = (
            getattr(cfg, "POSICIONES_POST_REBALANCEO_TEMPLATE", None)
            or getattr(cfg, "SUGGESTED_POSITIONS_TEMPLATE", "results/posiciones_post_rebalanceo_{date}.xlsx")
        )
        post_rebalance_path = Path(tpl.format(date=date_str))
        save_positions_to_excel(post_rebalance_path, reg["updated_positions"], as_of_date=last_dt)
        snap_tpl = getattr(cfg, "POSICIONES_ULTIMO_SNAPSHOT_PATH", "results/posiciones_post_rebalanceo_ultimo.xlsx")
        ultimo_snapshot_path = Path(snap_tpl)
        save_positions_to_excel(ultimo_snapshot_path, reg["updated_positions"], as_of_date=last_dt)

    if getattr(cfg, "APPEND_EJECUCION_LOG", False):
        log_csv = getattr(cfg, "EJECUCION_LOG_CSV", "results/historial_ejecuciones.csv")
        append_ejecucion_row(
            log_csv,
            {
                "fecha_datos": date_str,
                "nav_previo_eur": float(nav),
                "path_operaciones": reg.get("output_path", ""),
                "path_posiciones_post_rebalanceo": str(post_rebalance_path) if post_rebalance_path else "",
                "path_snapshot_ultimo": str(ultimo_snapshot_path) if ultimo_snapshot_path else "",
                "dn_rebalance": result["dn_result"].get("rebalance"),
                "ordenes_filas": len(reg.get("orders", [])),
                "skipped_due_to_dn": reg.get("skipped_due_to_dn", False),
            },
        )

    email_sent = False
    if getattr(cfg, "EMAIL_OPERACIONES_AFTER_RUN", False):
        email_sent = send_operaciones_excel(reg["output_path"], as_of_label=date_str)

    return {
        "market_data": market_data,
        "as_of_date": last_dt,
        "current_positions_qty": cw_qty,
        "current_weights": cw_weights,
        "portfolio_value_eur": float(nav),
        "positions_source": positions_source,
        "posiciones_post_rebalanceo_path": str(post_rebalance_path) if post_rebalance_path else None,
        "posiciones_ultimo_snapshot_path": str(ultimo_snapshot_path) if ultimo_snapshot_path else None,
        "suggested_positions_path": str(post_rebalance_path) if post_rebalance_path else None,
        "email_operaciones_sent": email_sent,
        "registrador": reg,
        **result,
    }


if __name__ == "__main__":
    result = run_single()
    rf = result["risk_free_rate"]
    print(f"Datos hasta: {result['as_of_date']}")
    print(f"Posiciones: {result['positions_source']} — NAV ~ {result['portfolio_value_eur']:,.0f} EUR")
    if result.get("posiciones_post_rebalanceo_path"):
        print(f"Posiciones post-rebalanceo (fecha): {result['posiciones_post_rebalanceo_path']}")
    if result.get("posiciones_ultimo_snapshot_path"):
        print(f"Ultimo snapshot posiciones: {result['posiciones_ultimo_snapshot_path']}")
    if result.get("email_operaciones_sent"):
        print("Correo de operaciones enviado (SMTP configurado).")
    print(f"Tasa BCE: {rf:.2%}")
    print(f"Regimen: {result['regime_early']}")
    print(f"VIEW_SCALE efectivo: {result['effective_view_scale']:.3f}")
    print(f"\nComposite scores (top 10):\n{result['scores'].head(10)}")
    print(f"\nPesos Merton: {result['merton_result']['weights']}")
    print(f"XEON: {result['merton_result']['weight_xeon']:.2%}")
    dn = result["dn_result"]
    print(f"\nDavis-Norman: rebalance={'SI' if dn['rebalance'] else 'NO'} — {dn.get('reason', '')}")
    reg = result["registrador"]
    print(f"\nOperativa guardada en: {reg['output_path']}")
    if reg.get("skipped_due_to_dn"):
        print("(Sin ordenes: Davis-Norman indica no rebalancear — ver hoja Nota en el Excel.)")
    elif len(reg["orders"]) > 0:
        print(reg["orders"].to_string(index=False))
    else:
        print("(Sin filas de ordenes.)")
