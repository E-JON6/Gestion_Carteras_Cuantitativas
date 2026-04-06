#!/usr/bin/env python3
"""View-only portfolio report — no strategy execution, no trades.

Reads estado.json + Historial_Grupo4.xlsx to show the current portfolio
state with metrics, charts, and the full operation history from E4.

Usage:
    python scripts/view_portfolio.py
    python scripts/view_portfolio.py --date 2026-04-06
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from src.metrics.performance import PerformanceReport

_MADRID_TZ = ZoneInfo("Europe/Madrid")

# ── Palette (same as main report) ───────────────────────────────────────────
BG     = "#0f172a"
CARD   = "#1e293b"
CARD2  = "#162032"
BORDER = "#334155"
TEXT   = "#f1f5f9"
MUTED  = "#94a3b8"
BLUE   = "#3b82f6"
PURPLE = "#a855f7"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
CYAN   = "#06b6d4"
TEAL   = "#14b8a6"
PINK   = "#ec4899"

ACCENT12 = [BLUE, GREEN, AMBER, PURPLE, CYAN, RED,
            TEAL, PINK, "#f97316", "#84cc16", "#38bdf8", "#e879f9"]


# ── Data loading ────────────────────────────────────────────────────────────

def load_estado(state_dir: str) -> dict:
    path = Path(state_dir) / "estado.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    data = json.loads(path.read_text())
    if not data.get("iniciado"):
        print("ERROR: estado.json not initialized")
        sys.exit(1)
    return data


def load_historial(state_dir: str, group_name: str) -> pd.DataFrame:
    path = Path(state_dir) / f"Historial_{group_name}.xlsx"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, header=1)
    df = df[df["Fecha"].notna()].copy()
    if df.empty:
        return df
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def fetch_prices(tickers: list[str], date: pd.Timestamp) -> dict[str, float]:
    """Get today's closing prices for tickers via yfinance."""
    start = date - pd.Timedelta(days=10)
    end = date + pd.Timedelta(days=1)
    prices = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df.empty:
                continue
            df = df.loc[df.index <= date]
            if not df.empty:
                close = df["Close"].iloc[-1]
                prices[t] = float(close.iloc[0]) if hasattr(close, "iloc") else float(close)
        except Exception as e:
            print(f"  Warning: could not fetch {t}: {e}")
    return prices


def build_vl_series(historial_df: pd.DataFrame) -> pd.DataFrame:
    """Build VL DataFrame from Historial — date, total_value."""
    if historial_df.empty:
        return pd.DataFrame(columns=["date", "total_value"])
    df = historial_df[["Fecha", "Valor cartera"]].copy()
    df.columns = ["date", "total_value"]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def extract_operations(historial_df: pd.DataFrame) -> pd.DataFrame:
    """Extract trade-day operations from Historial."""
    if historial_df.empty:
        return pd.DataFrame()

    iuse_ct = 0.00021
    rows = []
    for _, r in historial_df.iterrows():
        decision = str(r.get("Decision", "")).strip().upper()
        if decision == "MANTENER":
            continue

        importe = float(r.get("Importe (EUR)", 0) or 0)
        coste = float(r.get("Coste (EUR)", 0) or 0)
        precio = float(r.get("Precio ETF", 0) or 0)

        if importe <= 0 or precio <= 0:
            continue

        if decision == "COMPRAR":
            shares = importe / (precio * (1 + iuse_ct))
            rows.append({
                "date": r["Fecha"],
                "ticker": "IUSE.L",
                "shares": shares,
                "price": precio,
                "value": importe,
                "cost": coste,
                "direction": "buy",
                "strategy": "E4",
            })
        elif decision == "VENDER":
            shares = importe / (precio * (1 - iuse_ct))
            rows.append({
                "date": r["Fecha"],
                "ticker": "IUSE.L",
                "shares": -shares,
                "price": precio,
                "value": importe,
                "cost": coste,
                "direction": "sell",
                "strategy": "E4",
            })

    return pd.DataFrame(rows)


# ── Plotly helpers ──────────────────────────────────────────────────────────

def _layout(**kw) -> dict:
    base = dict(
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(family="'Inter','Segoe UI',system-ui,sans-serif", color=TEXT, size=11),
        margin=dict(l=50, r=30, t=44, b=40),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                    font=dict(color=MUTED, size=10), orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=9)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=9)),
    )
    base.update(kw)
    return base


def _cfg() -> dict:
    return dict(displayModeBar=True, displaylogo=False, responsive=True,
                modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"],
                toImageButtonOptions=dict(format="svg", scale=2))


def _to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_cfg())


# ── Charts ──────────────────────────────────────────────────────────────────

def chart_vl(vl_df: pd.DataFrame, initial_cash: float) -> str:
    if len(vl_df) < 2:
        return ""
    vl = vl_df.set_index("date")["total_value"]
    rets = vl.pct_change().dropna() * 100
    cummax = vl.cummax()
    dd = (vl - cummax) / cummax * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.04,
                        subplot_titles=["Valor Liquidativo (VL)", "Retorno Diario (%)", "Drawdown (%)"])
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 10

    fig.add_trace(go.Scatter(
        x=vl.index, y=vl.values, mode="lines", name="VL",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>VL: €%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hrect(y0=initial_cash, y1=vl.max()*1.01, fillcolor=GREEN, opacity=0.05,
                  layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=vl.min()*0.99, y1=initial_cash, fillcolor=RED, opacity=0.05,
                  layer="below", line_width=0, row=1, col=1)
    fig.add_hline(y=initial_cash, line_dash="dot", line_color=MUTED, line_width=1,
                  row=1, col=1, annotation_text=f"Capital inicial €{initial_cash/1e6:.1f}M",
                  annotation_font_color=MUTED, annotation_font_size=9)

    bar_colors = [GREEN if r >= 0 else RED for r in rets.values]
    fig.add_trace(go.Bar(
        x=rets.index, y=rets.values, marker_color=bar_colors, marker_opacity=0.85,
        name="Ret. diario",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:+.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="Drawdown",
        line=dict(color=RED, width=1.5), fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>DD: %{y:.2f}%<extra></extra>",
    ), row=3, col=1)

    axes_style = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=9))
    fig.update_layout(**_layout(height=560, showlegend=True,
                                legend=dict(orientation="h", y=1.03, x=0, font=dict(color=MUTED, size=10)),
                                margin=dict(l=60, r=30, t=50, b=40)))
    fig.update_xaxes(**axes_style)
    fig.update_yaxes(**axes_style)
    fig.update_yaxes(tickformat="€,.0f", row=1, col=1)
    fig.update_yaxes(tickformat="+.1f", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(tickformat=".1f", ticksuffix="%", row=3, col=1)
    fig.update_traces(marker_line_width=0, row=2, col=1)
    return _to_div(fig)


def chart_composition(positions: list[dict], total_value: float) -> str:
    if not positions:
        return ""
    df = pd.DataFrame(positions).sort_values("weight")
    colors = [GREEN if w >= 0.15 else BLUE if w >= 0.05 else AMBER for w in df["weight"]]

    fig = go.Figure(go.Bar(
        x=df["weight"] * 100, y=df["ticker"], orientation="h",
        marker=dict(color=colors, opacity=0.9, line=dict(color=BG, width=1.2)),
        text=[f"{w*100:.1f}%" for w in df["weight"]],
        textposition="inside", insidetextanchor="end",
        textfont=dict(color="white", size=10, family="'SF Mono',monospace"),
        hovertemplate="<b>%{y}</b><br>Peso: <b>%{x:.2f}%</b><br>Valor: €%{customdata[0]:,.0f}<extra></extra>",
        customdata=list(zip(df["value"], df["shares"])),
        showlegend=False,
    ))
    total_pct = df["weight"].sum() * 100
    fig.update_layout(**_layout(
        title=dict(text=f"Composición de la Cartera — Total: {total_pct:.1f}% s/NAV",
                   font=dict(color=TEXT, size=12)),
        height=max(200, len(df) * 50 + 90),
        xaxis=dict(title="Peso sobre NAV (%)", range=[0, df["weight"].max()*100*1.18],
                   gridcolor=BORDER, ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT, size=11,
                   family="'SF Mono',monospace")),
    ))
    return _to_div(fig)


def chart_prices(tickers: list[str], date: pd.Timestamp) -> str:
    """Price chart for IUSE.L and XEON.DE over last ~6 months."""
    start = date - pd.Timedelta(days=200)
    end = date + pd.Timedelta(days=1)
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        if df.empty:
            return ""
        close = df["Close"]
    except Exception:
        return ""

    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    ncols = min(len(tickers), 2)
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=tickers[:ncols],
                        horizontal_spacing=0.08)
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 10

    axes = dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=9))

    for idx, t in enumerate(tickers[:ncols]):
        col_n = idx + 1
        if t not in close.columns:
            continue
        data = close[t].dropna()
        if len(data) < 2:
            continue
        chg = (data.iloc[-1] / data.iloc[0] - 1) * 100
        line_color = GREEN if chg >= 0 else RED
        color_idx = ACCENT12[idx % len(ACCENT12)]

        fig.add_trace(go.Scatter(
            x=data.index, y=data.values, mode="lines", name=t,
            line=dict(color=line_color, width=2),
            hovertemplate=f"<b>{t}</b> %{{x|%d/%m/%Y}}<br>%{{y:.2f}}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_n)

        if len(data) >= 21:
            ma21 = data.rolling(21).mean()
            fig.add_trace(go.Scatter(
                x=ma21.index, y=ma21.values, mode="lines", name="MA21",
                line=dict(color=AMBER, width=1.2, dash="dash"),
                showlegend=(idx == 0), opacity=0.85,
            ), row=1, col=col_n)

        fig.add_trace(go.Scatter(
            x=[data.index[-1]], y=[data.iloc[-1]], mode="markers",
            marker=dict(color=line_color, size=7, line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_n)

    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter,sans-serif", size=10),
        height=320, margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x", showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(color=MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return _to_div(fig)


# ── HTML helpers ────────────────────────────────────────────────────────────

def _eur(v: float) -> str:
    return f"€{v:,.0f}"

def _pct(v: float) -> str:
    s = "+" if v >= 0 else ""
    return f"{s}{v:.2%}"

def _kpi(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (
        f'<div class="kpi" style="border-left-color:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{"<div class=\"kpi-sub\">" + sub + "</div>" if sub else ""}'
        f'</div>'
    )

def _badge(ok: bool, label: str) -> str:
    bg = "#14532d" if ok else "#450a0a"
    fg = "#4ade80" if ok else "#f87171"
    ic = "✓" if ok else "✗"
    return f'<span class="badge" style="background:{bg};color:{fg}">{ic} {label}</span>'

def _section(title: str, dot: str = "blue") -> str:
    return f'<h2><span class="dot {dot}"></span>{title}</h2>'

def _table(headers: list[str], rows: str) -> str:
    ths = "".join(
        f'<th style="text-align:{"left" if i==0 else "right"}">{h}</th>'
        for i, h in enumerate(headers)
    )
    return f'<div class="tbl-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{rows}</tbody></table></div>'

_MONO = "font-family:'SF Mono',ui-monospace,monospace;font-size:12px"
_R = "text-align:right"
_L = "text-align:left"


def _strategy_pill(strategy: str) -> str:
    colors = {"E4": (AMBER, "#451a03"), "transicion": (PURPLE, "#2e1065"), "merton_mom": (CYAN, "#083344")}
    fg, bg = colors.get(strategy, (MUTED, CARD2))
    label = {"E4": "E4 · IUSE.L", "transicion": "Transicion", "merton_mom": "merton_mom"}.get(strategy, strategy)
    return (f'<span style="display:inline-block;padding:1px 7px;border-radius:9px;'
            f'font-size:10px;font-weight:600;letter-spacing:.03em;'
            f'background:{bg};color:{fg};white-space:nowrap">{label}</span>')


def tbl_positions(positions: list[dict]) -> str:
    if not positions:
        return '<p class="empty-msg">Sin posiciones abiertas.</p>'
    rows = ""
    for p in sorted(positions, key=lambda x: -x["value"]):
        wt = p["weight"] * 100
        wt_col = GREEN if wt >= 5 else AMBER if wt >= 1 else RED
        rows += (
            f"<tr>"
            f"<td style='{_L};{_MONO};font-weight:700'>{p['ticker']}</td>"
            f"<td style='{_R};{_MONO}'>{p['shares']:,.2f}</td>"
            f"<td style='{_R};{_MONO}'>€{p['price']:,.3f}</td>"
            f"<td style='{_R};{_MONO};font-weight:700'>€{p['value']:,.0f}</td>"
            f"<td style='{_R};font-weight:700;color:{wt_col}'>{wt:.2f}%</td>"
            f"</tr>"
        )
    total = sum(p["weight"] for p in positions) * 100
    rows += (
        f"<tr style='border-top:1px solid {BORDER}'>"
        f"<td style='{_L};color:{MUTED};font-size:11px' colspan='3'>Total peso s/NAV</td>"
        f"<td style='{_R};{_MONO};color:{MUTED}'></td>"
        f"<td style='{_R};font-weight:700;color:{GREEN}'>{total:.1f}%</td></tr>"
    )
    return _table(["Ticker", "Acciones", "Precio", "Valor", "Peso s/NAV"], rows)


def tbl_ops(ops_df: pd.DataFrame) -> str:
    if ops_df.empty:
        return '<p class="empty-msg">Sin historial de operaciones.</p>'
    sorted_df = ops_df.sort_values("date", ascending=False)
    rows = ""
    for _, r in sorted_df.iterrows():
        dir_col = GREEN if r["direction"] == "buy" else RED
        dir_lbl = "COMPRA" if r["direction"] == "buy" else "VENTA"
        pill = _strategy_pill(r.get("strategy", "E4"))
        rows += (
            f"<tr>"
            f"<td style='{_L};{_MONO}'>{pd.Timestamp(r['date']).strftime('%d/%m/%Y')}</td>"
            f"<td style='{_L};{_MONO};font-weight:700'>{r['ticker']}</td>"
            f"<td style='{_L};font-weight:700;color:{dir_col}'>{dir_lbl}</td>"
            f"<td style='{_R};{_MONO}'>{abs(r['shares']):,.2f}</td>"
            f"<td style='{_R};{_MONO}'>€{r['price']:,.3f}</td>"
            f"<td style='{_R};{_MONO}'>€{abs(r['value']):,.0f}</td>"
            f"<td style='{_R};{_MONO}'>€{r['cost']:,.2f}</td>"
            f"<td style='{_L}'>{pill}</td>"
            f"</tr>"
        )
    tc = ops_df["cost"].sum()
    n = len(ops_df)
    rows += (
        f"<tr style='border-top:1px solid {BORDER}'>"
        f"<td colspan='6' style='color:{MUTED};font-size:11px'>Total: {n} operaciones</td>"
        f"<td style='{_R};{_MONO};font-weight:700'>€{tc:,.2f}</td>"
        f"<td></td></tr>"
    )
    return _table(["Fecha", "Ticker", "Dir.", "Acciones", "Precio", "Valor Noc.", "Coste CT", ""], rows)


def tbl_metrics(metrics: dict) -> str:
    if not metrics:
        return '<p class="empty-msg">Historial insuficiente.</p>'
    items = [
        ("total_return",          "Retorno Total",          True,  True),
        ("annualized_return",     "Retorno Anualizado",     True,  True),
        ("annualized_volatility", "Volatilidad Anualizada", True,  False),
        ("sharpe_ratio",          "Sharpe Ratio",           False, True),
        ("max_drawdown",          "Max Drawdown",           True,  False),
        ("calmar_ratio",          "Calmar Ratio",           False, True),
    ]
    rows = ""
    for key, label, is_pct, higher_better in items:
        v = metrics.get(key)
        if v is None:
            continue
        fmt = f"{v:+.2%}" if is_pct else f"{v:+.4f}"
        ok = (v > 0 and higher_better) or (v < 0 and not higher_better and key == "max_drawdown")
        color = GREEN if ok else (MUTED if abs(v) < 1e-6 else RED)
        rows += (
            f"<tr><td style='{_L}'>{label}</td>"
            f"<td style='{_R};{_MONO};color:{color};font-weight:700'>{fmt}</td></tr>"
        )
    return _table(["Metrica", "Valor"], rows)


def tbl_historial(historial_df: pd.DataFrame) -> str:
    """Render the raw Historial table with all daily entries."""
    if historial_df.empty:
        return '<p class="empty-msg">Sin datos en Historial.</p>'
    rows = ""
    for _, r in historial_df.sort_values("Fecha", ascending=False).iterrows():
        fecha = pd.Timestamp(r["Fecha"]).strftime("%d/%m/%Y")
        valor = float(r.get("Valor cartera", 0) or 0)
        ret_acum = float(r.get("Retorno acum.", 0) or 0)
        decision = str(r.get("Decision", "—"))
        coste = float(r.get("Coste (EUR)", 0) or 0)
        n_ops = int(r.get("N operaciones", 0) or 0)

        dec_col = GREEN if "COMPRAR" in decision.upper() else RED if "VENDER" in decision.upper() else MUTED
        ret_col = GREEN if ret_acum > 0 else RED if ret_acum < 0 else MUTED

        rows += (
            f"<tr>"
            f"<td style='{_L};{_MONO}'>{fecha}</td>"
            f"<td style='{_R};{_MONO};font-weight:700'>€{valor:,.0f}</td>"
            f"<td style='{_R};{_MONO};color:{ret_col}'>{ret_acum:+.4f}</td>"
            f"<td style='{_L};font-weight:600;color:{dec_col}'>{decision}</td>"
            f"<td style='{_R};{_MONO}'>€{coste:,.2f}</td>"
            f"<td style='{_R};{_MONO}'>{n_ops}</td>"
            f"</tr>"
        )
    return _table(["Fecha", "Valor Cartera", "Ret. Acum.", "Decision", "Coste", "N Ops Acum."], rows)


# ── CSS ─────────────────────────────────────────────────────────────────────

CSS = f"""
:root{{--bg:{BG};--card:{CARD};--card2:{CARD2};--border:{BORDER};
  --text:{TEXT};--muted:{MUTED};--blue:{BLUE};--green:{GREEN};
  --red:{RED};--amber:{AMBER};--purple:{PURPLE};--cyan:{CYAN}}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1300px;margin:0 auto;padding:20px 24px}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;
  padding:16px 0 14px;border-bottom:1px solid var(--border);margin-bottom:18px}}
.header-title h1{{font-size:22px;font-weight:800;letter-spacing:-.02em}}
.header-title .sub{{color:var(--muted);font-size:12px;margin-top:3px}}
.badge-row{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;margin-top:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin-bottom:16px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:12px 14px;border-left-width:3px;transition:transform .15s}}
.kpi:hover{{transform:translateY(-1px)}}
.kpi-label{{font-size:9px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.08em;font-weight:700}}
.kpi-value{{font-size:21px;font-weight:800;margin-top:5px;font-family:
  'SF Mono',ui-monospace,'Cascadia Mono',monospace;letter-spacing:-.02em}}
.kpi-sub{{font-size:10px;color:var(--muted);margin-top:2px}}
.tab-bar{{display:flex;gap:0;border-bottom:1px solid var(--border);
  margin-bottom:16px;overflow-x:auto}}
.tab-btn{{background:none;border:none;border-bottom:2px solid transparent;
  padding:10px 16px;color:var(--muted);font-size:12px;font-weight:600;
  cursor:pointer;margin-bottom:-1px;white-space:nowrap;
  transition:color .15s,border-color .15s;letter-spacing:.02em}}
.tab-btn:hover{{color:var(--text)}}
.tab-btn.active{{color:var(--blue);border-bottom-color:var(--blue)}}
.tab-content{{display:none;animation:fadeIn .18s ease-in}}
.tab-content.active{{display:block}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:16px 18px;margin-bottom:12px}}
.card h2{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:6px}}
.dot{{width:6px;height:6px;border-radius:50%;background:var(--blue);flex-shrink:0}}
.dot.green{{background:var(--green)}}.dot.red{{background:var(--red)}}
.dot.amber{{background:var(--amber)}}.dot.purple{{background:var(--purple)}}
.dot.cyan{{background:var(--cyan)}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.g3{{display:grid;grid-template-columns:2fr 1fr;gap:12px}}
.g4{{display:grid;grid-template-columns:3fr 1fr;gap:12px}}
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{padding:6px 10px;font-size:9px;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border);
  background:rgba(0,0,0,.2);text-align:left;white-space:nowrap}}
th.right{{text-align:right}}
td{{padding:7px 10px;border-bottom:1px solid rgba(51,65,85,.5);color:var(--text)}}
td.right{{text-align:right}}
td.mono{{font-family:'SF Mono',ui-monospace,monospace;font-size:11.5px}}
td.bold{{font-weight:700}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,.03)}}
.badge{{font-size:10px;font-weight:700;padding:3px 9px;border-radius:20px;
  letter-spacing:.02em}}
.badges{{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px}}
.empty-msg{{color:var(--muted);font-style:italic;font-size:12px;padding:8px 0}}
.section-note{{color:var(--muted);font-size:11px;margin-bottom:10px;line-height:1.7}}
.footer{{text-align:center;color:var(--muted);font-size:10px;padding:18px 0 6px;
  border-top:1px solid var(--border);margin-top:8px}}
@media(max-width:768px){{.g2,.g3,.g4{{grid-template-columns:1fr}}
  .kpis{{grid-template-columns:repeat(2,1fr)}}}}
"""

JS = """
function showTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setTimeout(() => {
        document.querySelectorAll('.js-plotly-plot').forEach(el => {
          Plotly.Plots.resize(el);
        });
      }, 50);
    });
  });
});
"""


# ── Report generation ───────────────────────────────────────────────────────

def generate_report(
    estado: dict,
    historial_df: pd.DataFrame,
    prices: dict[str, float],
    date: pd.Timestamp,
    output_dir: str = "outputs/reports",
    initial_cash: float = 10_000_000.0,
) -> Path:

    # ── Current positions from estado.json ──
    iuse_shares = estado.get("participaciones", 0.0)
    xeon_shares = estado.get("participaciones_rf", 0.0)

    iuse_price = prices.get("IUSE.L", 0.0)
    xeon_price = prices.get("XEON.DE", 0.0)

    positions = []
    total_value = 0.0
    if iuse_shares > 0 and iuse_price > 0:
        val = iuse_shares * iuse_price
        positions.append({"ticker": "IUSE.L", "shares": iuse_shares, "price": iuse_price, "value": val, "weight": 0.0})
        total_value += val
    if xeon_shares > 0 and xeon_price > 0:
        val = xeon_shares * xeon_price
        positions.append({"ticker": "XEON.DE", "shares": xeon_shares, "price": xeon_price, "value": val, "weight": 0.0})
        total_value += val

    # Compute weights
    for p in positions:
        p["weight"] = p["value"] / total_value if total_value > 0 else 0.0

    # ── VL series from Historial ──
    vl_df = build_vl_series(historial_df)

    # Append today's mark-to-market as the latest VL
    if total_value > 0:
        today_entry = pd.DataFrame([{"date": date, "total_value": total_value}])
        vl_df = vl_df[vl_df["date"] != date]
        vl_df = pd.concat([vl_df, today_entry], ignore_index=True).sort_values("date").reset_index(drop=True)

    # ── Operations ──
    ops_df = extract_operations(historial_df)

    # ── Metrics ──
    metrics = {}
    if len(vl_df) >= 2:
        vl_series = pd.Series(vl_df["total_value"].values, index=vl_df["date"])
        report = PerformanceReport(values=vl_series, risk_free_rate=0.0)
        metrics = report.compute()

    # ── Key numbers ──
    last_historial_vl = float(estado.get("valor_cartera", total_value))
    daily_pnl = total_value - last_historial_vl
    total_ret = total_value / initial_cash - 1
    n_ops = int(estado.get("n_operaciones", 0))
    costes_acum = float(estado.get("costes_acumulados", 0))
    peak = float(estado.get("peak_valor", total_value))
    dd_from_peak = (total_value - peak) / peak if peak > 0 else 0.0

    pnl_col = GREEN if daily_pnl >= 0 else RED
    ret_col = GREEN if total_ret >= 0 else RED

    # ── Charts ──
    ch_vl = chart_vl(vl_df, initial_cash)
    ch_comp = chart_composition(positions, total_value)
    ch_prices = chart_prices(["IUSE.L", "XEON.DE"], date)

    # ── HTML assembly ──
    p: list[str] = []
    a = p.append

    a(f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio E4 · {date.strftime('%d/%m/%Y')}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">""")

    # Header
    a(f"""<div class="header">
  <div class="header-title">
    <h1>Portfolio Dashboard — Vista Actual</h1>
    <div class="sub">{date.strftime('%A, %d de %B de %Y')} &nbsp;·&nbsp;
    <b>Estrategia E4 (IUSE.L + XEON.DE)</b> &nbsp;·&nbsp; Afi MFC Gestion Cuantitativa &nbsp;·&nbsp; Grupo 4</div>
  </div>
  <div class="badge-row">
    {_badge(True, "Solo lectura")}
    {_strategy_pill("E4")}
  </div>
</div>""")

    # KPIs
    pnl_sign = "+" if daily_pnl >= 0 else ""
    a('<div class="kpis">')
    a(_kpi("Valor Liquidativo", _eur(total_value), f"Inicial: {_eur(initial_cash)}", BLUE))
    a(_kpi("P&L vs Ultimo Cierre", f"{pnl_sign}{_eur(daily_pnl)}", f"vs estado.json ({estado.get('fecha_ultima_ejecucion', '—')})", pnl_col))
    a(_kpi("Retorno Total", _pct(total_ret), f"desde {estado.get('fecha_inicio', '—')}", ret_col))
    a(_kpi("Drawdown desde Pico", _pct(dd_from_peak), f"pico: {_eur(peak)}", RED if dd_from_peak < -0.01 else GREEN))
    a(_kpi("Operaciones", str(n_ops), f"Costes acum.: {_eur(costes_acum)}", AMBER))
    if metrics:
        sr = metrics.get("sharpe_ratio", 0)
        md = metrics.get("max_drawdown", 0)
        a(_kpi("Sharpe", f"{sr:+.3f}", "riesgo/retorno", GREEN if sr > 0 else RED))
        a(_kpi("Max Drawdown", _pct(md), "caida max. desde pico", RED))
    a('</div>')

    # Tabs
    a("""<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('t-dash',this)">Dashboard</button>
  <button class="tab-btn" onclick="showTab('t-perf',this)">Rendimiento</button>
  <button class="tab-btn" onclick="showTab('t-precios',this)">Precios</button>
  <button class="tab-btn" onclick="showTab('t-ops',this)">Operaciones</button>
  <button class="tab-btn" onclick="showTab('t-historial',this)">Historial</button>
</div>""")

    # ════ TAB: DASHBOARD ════
    a('<div id="t-dash" class="tab-content active">')

    a('<div class="card">')
    a(_section("Posiciones Actuales", "cyan"))
    a(f'<p class="section-note">Mark-to-market con precios del {date.strftime("%d/%m/%Y")}. '
      f'Valor total cartera: <b>{_eur(total_value)}</b></p>')
    a(tbl_positions(positions))
    a('</div>')

    if ch_comp:
        a('<div class="card">')
        a(_section("Composicion por Peso"))
        a(ch_comp)
        a('</div>')

    a('<div class="card">')
    a(_section("Estado de la Cartera", "green"))
    info_rows = (
        f"<tr><td style='{_L}'>Capital inicial</td><td style='{_R};{_MONO}'>{_eur(initial_cash)}</td></tr>"
        f"<tr><td style='{_L}'>Valor cartera (MTM)</td><td style='{_R};{_MONO};font-weight:700'>{_eur(total_value)}</td></tr>"
        f"<tr><td style='{_L}'>Valor cartera (ultimo cierre)</td><td style='{_R};{_MONO}'>{_eur(last_historial_vl)}</td></tr>"
        f"<tr><td style='{_L}'>Fecha inicio</td><td style='{_R};{_MONO}'>{estado.get('fecha_inicio', '—')}</td></tr>"
        f"<tr><td style='{_L}'>Ultima ejecucion</td><td style='{_R};{_MONO}'>{estado.get('fecha_ultima_ejecucion', '—')}</td></tr>"
        f"<tr><td style='{_L}'>Alpha actual</td><td style='{_R};{_MONO};color:{CYAN}'>{estado.get('alpha_actual', '—')}</td></tr>"
        f"<tr><td style='{_L}'>Modo defensivo</td><td style='{_R};{_MONO}'>"
        f"{'SI' if estado.get('en_modo_defensivo') else 'NO'}</td></tr>"
        f"<tr><td style='{_L}'>Peak VL</td><td style='{_R};{_MONO}'>{_eur(peak)}</td></tr>"
        f"<tr><td style='{_L}'>N operaciones</td><td style='{_R};{_MONO}'>{n_ops}</td></tr>"
        f"<tr><td style='{_L}'>Costes acumulados</td><td style='{_R};{_MONO}'>{_eur(costes_acum)}</td></tr>"
    )
    a(_table(["Parametro", "Valor"], info_rows))
    a('</div>')

    a('</div>')  # /t-dash

    # ════ TAB: RENDIMIENTO ════
    a('<div id="t-perf" class="tab-content">')
    a('<div class="g4">')
    a('<div>')
    if ch_vl:
        a('<div class="card">')
        a(_section("Evolucion del VL · Retornos Diarios · Drawdown", "green"))
        a(ch_vl)
        a('</div>')
    else:
        a('<div class="card">')
        a(_section("Evolucion del VL", "green"))
        a('<p class="empty-msg">Se necesitan >= 2 dias de historial.</p>')
        a('</div>')

    a('</div>')  # /left col
    a('<div>')  # right col

    a('<div class="card">')
    a(_section("Metricas de Rendimiento", "purple"))
    a(tbl_metrics(metrics))
    a('</div>')

    if len(vl_df) >= 2:
        vl_s = vl_df.set_index("date")["total_value"]
        rets = vl_s.pct_change().dropna() * 100
        win_rate = (rets > 0).mean() * 100
        avg_pos = rets[rets > 0].mean() if (rets > 0).any() else 0
        avg_neg = rets[rets < 0].mean() if (rets < 0).any() else 0
        rows_stat = (
            f"<tr><td>Dias positivos</td>"
            f"<td class='right mono' style='color:{GREEN}'>{win_rate:.1f}%</td></tr>"
            f"<tr><td>Media dia +</td>"
            f"<td class='right mono' style='color:{GREEN}'>{avg_pos:+.2f}%</td></tr>"
            f"<tr><td>Media dia -</td>"
            f"<td class='right mono' style='color:{RED}'>{avg_neg:+.2f}%</td></tr>"
            f"<tr><td>Mejor dia</td>"
            f"<td class='right mono' style='color:{GREEN}'>{rets.max():+.2f}%</td></tr>"
            f"<tr><td>Peor dia</td>"
            f"<td class='right mono' style='color:{RED}'>{rets.min():+.2f}%</td></tr>"
        )
        a('<div class="card">')
        a(_section("Estadisticos Diarios", "red"))
        a(_table(["Estadistico", "Valor"], rows_stat))
        a('</div>')

    a('</div>')  # /right col
    a('</div>')  # /g4
    a('</div>')  # /t-perf

    # ════ TAB: PRECIOS ════
    a('<div id="t-precios" class="tab-content">')
    a('<p class="section-note">Precios de cierre ajustados · ultimos ~6 meses · '
      f'<span style="color:{AMBER}">-- MA21</span> · Zoom y hover interactivos.</p>')
    if ch_prices:
        a('<div class="card">')
        a(_section("Precios Historicos — IUSE.L y XEON.DE"))
        a(ch_prices)
        a('</div>')
    else:
        a('<div class="card">')
        a('<p class="empty-msg">No se pudieron descargar precios historicos.</p>')
        a('</div>')
    a('</div>')  # /t-precios

    # ════ TAB: OPERACIONES ════
    a('<div id="t-ops" class="tab-content">')
    a('<div class="card">')
    a(_section("Historial de Operaciones — Estrategia E4", "amber"))
    a(f'<p class="section-note">Operaciones extraidas del Historial_Grupo4.xlsx. '
      f'Total: <b>{len(ops_df)} operaciones</b> · Costes acumulados: <b>{_eur(costes_acum)}</b></p>')
    a(tbl_ops(ops_df))
    a('</div>')
    a('</div>')  # /t-ops

    # ════ TAB: HISTORIAL ════
    a('<div id="t-historial" class="tab-content">')
    a('<div class="card">')
    a(_section("Historial Diario Completo", "green"))
    a(f'<p class="section-note">Datos del archivo Historial_Grupo4.xlsx — '
      f'{len(historial_df)} dias registrados (desde {estado.get("fecha_inicio", "—")})</p>')
    a(tbl_historial(historial_df))
    a('</div>')
    a('</div>')  # /t-historial

    # Footer
    now_str = datetime.now(_MADRID_TZ).strftime("%d/%m/%Y %H:%M (Madrid)")
    a(f'<div class="footer">Generado el {now_str} &nbsp;·&nbsp; '
      f'Afi MFC Gestion Cuantitativa &nbsp;·&nbsp; Grupo 4 &nbsp;·&nbsp; '
      f'<span style="color:{AMBER}">Vista de solo lectura — sin transacciones</span></div>')
    a(f'<script>{JS}</script>')
    a('</div></body></html>')

    # ── Write ──
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = "\n".join(p)
    dated = out_dir / f"{date.strftime('%Y-%m-%d')}_portfolio_view.html"
    dated.write_text(html, encoding="utf-8")
    latest = out_dir / "latest_view.html"
    latest.write_text(html, encoding="utf-8")
    return dated


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View-only portfolio report (no trades)")
    parser.add_argument("--date", default=None, help="Date for MTM (YYYY-MM-DD). Default: today")
    parser.add_argument("--state-dir", default="outputs/state", help="State directory")
    parser.add_argument("--output-dir", default="outputs/reports", help="Output directory")
    parser.add_argument("--cash", type=float, default=10_000_000, help="Initial capital")
    parser.add_argument("--group", default="Grupo4", help="Group name")
    args = parser.parse_args()

    if args.date:
        date = pd.Timestamp(args.date)
    else:
        date = pd.Timestamp(datetime.now(_MADRID_TZ).strftime("%Y-%m-%d"))

    print(f"Portfolio View Report — {date.strftime('%Y-%m-%d')}")
    print("-" * 50)

    # 1. Load estado.json
    estado = load_estado(args.state_dir)
    print(f"Estado cargado: VL={estado.get('valor_cartera', 0):,.0f} EUR "
          f"(fecha: {estado.get('fecha_ultima_ejecucion', '?')})")
    print(f"  IUSE.L: {estado.get('participaciones', 0):,.2f} shares")
    print(f"  XEON.DE: {estado.get('participaciones_rf', 0):,.2f} shares")

    # 2. Load Historial
    historial_df = load_historial(args.state_dir, args.group)
    print(f"Historial: {len(historial_df)} dias")

    # 3. Fetch today's prices
    print("\nFetching prices...")
    prices = fetch_prices(["IUSE.L", "XEON.DE"], date)
    for t, p in prices.items():
        print(f"  {t}: €{p:.4f}")

    if not prices:
        print("ERROR: Could not fetch any prices")
        sys.exit(1)

    # 4. Mark-to-market
    iuse_val = estado.get("participaciones", 0) * prices.get("IUSE.L", 0)
    xeon_val = estado.get("participaciones_rf", 0) * prices.get("XEON.DE", 0)
    total_mtm = iuse_val + xeon_val
    print(f"\nMark-to-market: €{total_mtm:,.0f}")
    print(f"  IUSE.L: €{iuse_val:,.0f} ({iuse_val/total_mtm*100:.1f}%)")
    print(f"  XEON.DE: €{xeon_val:,.0f} ({xeon_val/total_mtm*100:.1f}%)")

    ret = total_mtm / args.cash - 1
    print(f"  Retorno total: {ret:+.2%}")

    # 5. Generate report
    print("\nGenerating report...")
    report_path = generate_report(
        estado=estado,
        historial_df=historial_df,
        prices=prices,
        date=date,
        output_dir=args.output_dir,
        initial_cash=args.cash,
    )
    print(f"Report: {report_path}")
    print(f"Latest: {args.output_dir}/latest_view.html")
    print("\nDone. No transactions executed.")


if __name__ == "__main__":
    main()
