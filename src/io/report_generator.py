"""Daily portfolio report generator.

Produces:
  - outputs/reports/YYYY-MM-DD_report.html  — self-contained HTML dashboard
  - outputs/reports/latest.html             — always points to today's report
  - outputs/reports/seguimiento.xlsx        — multi-sheet tracking Excel
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from src.domain.asset import Universe
from src.io.vl_tracker import VLTracker
from src.io.portfolio_state import PortfolioStateManager


# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "blue":   "#2563eb",
    "purple": "#7c3aed",
    "green":  "#16a34a",
    "red":    "#dc2626",
    "amber":  "#f59e0b",
    "gray":   "#6b7280",
    "light":  "#f3f4f6",
    "dark":   "#111827",
    "white":  "#ffffff",
}

STYLE = {
    "figure.facecolor": "#fafafa",
    "axes.facecolor":   "#fafafa",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight": "bold",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style_ax(ax, title: str = "", ylabel: str = "") -> None:
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _eur(v: float) -> str:
    return f"€{v:,.0f}"


def _pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2%}"


def _badge(ok: bool, label: str) -> str:
    color = "#dcfce7" if ok else "#fee2e2"
    text_color = "#166534" if ok else "#991b1b"
    icon = "✓" if ok else "✗"
    return (
        f'<span style="background:{color};color:{text_color};'
        f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">'
        f'{icon} {label}</span>'
    )


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_composition(positions_df: pd.DataFrame, total_value: float) -> str:
    """Horizontal bar chart of current position weights."""
    df = positions_df.copy().sort_values("weight")
    n = len(df)
    if n == 0:
        return ""

    fig_h = max(3.5, n * 0.38)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, fig_h))

    colors = [C["blue"] if w >= 0.05 else C["purple"] if w >= 0.02 else C["amber"]
              for w in df["weight"]]

    bars = ax.barh(df["ticker"], df["weight"] * 100, color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.4, height=0.65)

    # 1% minimum line
    ax.axvline(1.0, color=C["red"], linewidth=1.2, linestyle="--", alpha=0.7,
               label="Mínimo 1%")
    # 5% reference
    ax.axvline(5.0, color=C["gray"], linewidth=0.7, linestyle=":", alpha=0.5)

    # Value labels
    for bar, row in zip(bars, df.itertuples()):
        w = row.weight * 100
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%  (€{row.value:,.0f})", va="center", fontsize=8)

    ax.set_xlabel("Peso (%)", fontsize=9)
    ax.set_xlim(0, max(df["weight"].max() * 100 * 1.35, 5))
    _style_ax(ax, title="Composición de la Cartera")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


def _plot_vl_history(vl_df: pd.DataFrame, initial_cash: float) -> str:
    """Equity curve with initial capital reference."""
    if len(vl_df) < 2:
        return ""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))

    vl = vl_df.set_index("date")["total_value"]
    ax.fill_between(vl.index, vl.values, initial_cash,
                    where=vl.values >= initial_cash, color=C["green"], alpha=0.12)
    ax.fill_between(vl.index, vl.values, initial_cash,
                    where=vl.values < initial_cash, color=C["red"], alpha=0.12)
    ax.plot(vl.index, vl.values, color=C["blue"], linewidth=2.0, zorder=3)
    ax.axhline(initial_cash, color=C["gray"], linewidth=1.0, linestyle="--",
               alpha=0.6, label=f"Capital inicial {_eur(initial_cash)}")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1e6:.2f}M"))
    _style_ax(ax, title="Valor Liquidativo (VL)", ylabel="VL (EUR)")
    ax.legend(fontsize=8, frameon=False)
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


def _plot_daily_returns(vl_df: pd.DataFrame) -> str:
    """Bar chart of daily returns."""
    if len(vl_df) < 3:
        return ""

    df = vl_df.copy().set_index("date")
    rets = df["total_value"].pct_change().dropna()
    if len(rets) == 0:
        return ""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3))

    colors = [C["green"] if r >= 0 else C["red"] for r in rets.values]
    ax.bar(range(len(rets)), rets.values * 100, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.3, width=0.7)
    ax.axhline(0, color=C["dark"], linewidth=0.5, alpha=0.4)

    if len(rets) <= 30:
        ax.set_xticks(range(len(rets)))
        ax.set_xticklabels([d.strftime("%d/%m") for d in rets.index],
                           rotation=45, ha="right", fontsize=7)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _style_ax(ax, title="Retorno Diario", ylabel="Retorno (%)")
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


def _plot_drawdown(vl_df: pd.DataFrame) -> str:
    """Drawdown chart."""
    if len(vl_df) < 3:
        return ""

    df = vl_df.set_index("date")["total_value"]
    cummax = df.cummax()
    dd = (df - cummax) / cummax * 100
    if dd.max() == 0 and dd.min() == 0:
        return ""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.fill_between(dd.index, dd.values, 0, color=C["red"], alpha=0.25)
    ax.plot(dd.index, dd.values, color=C["red"], linewidth=1.2, alpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _style_ax(ax, title="Drawdown", ylabel="DD (%)")
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


def _plot_leverage_gauge(leverage: float, max_leverage: float = 2.0) -> str:
    """Horizontal bar showing current leverage vs limit."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 1.1))

    pct = min(leverage / max_leverage, 1.0)
    color = C["green"] if pct < 0.8 else C["amber"] if pct < 0.95 else C["red"]

    ax.barh([0], [max_leverage * 100], color=C["light"], height=0.6,
            edgecolor=C["gray"], linewidth=0.5)
    ax.barh([0], [leverage * 100], color=color, height=0.6,
            alpha=0.88, edgecolor="white", linewidth=0.3)

    ax.axvline(max_leverage * 100, color=C["red"], linewidth=1.5,
               linestyle="--", label=f"Límite {max_leverage*100:.0f}%")
    ax.text(leverage * 100 + 1, 0, f"{leverage*100:.1f}%",
            va="center", fontsize=10, fontweight="bold", color=C["dark"])

    ax.set_xlim(0, max_leverage * 100 * 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Apalancamiento (%)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


def _plot_sector_pie(positions_df: pd.DataFrame, universe: Universe) -> str:
    """Pie chart by sector using universe sector data."""
    if positions_df.empty:
        return ""

    # Build sector map from universe assets
    sector_map = {a.ticker: (a.sector or "Other") for a in universe.assets}
    positions_df = positions_df.copy()
    positions_df["sector"] = positions_df["ticker"].map(sector_map).fillna("Other")

    by_sector = positions_df.groupby("sector")["value"].sum()
    if len(by_sector) == 0:
        return ""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4))

    palette = [C["blue"], C["purple"], C["green"], C["amber"],
               C["red"], "#06b6d4", "#a78bfa", "#34d399", "#fb923c"]
    colors = [palette[i % len(palette)] for i in range(len(by_sector))]

    wedges, texts, autotexts = ax.pie(
        by_sector.values, labels=by_sector.index,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors, startangle=90, pctdistance=0.78,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(7)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Distribución por Sector", fontsize=12, fontweight="bold", pad=10)
    fig.patch.set_facecolor("#fafafa")
    return _fig_to_b64(fig)


# ── HTML builder ───────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f1f5f9; color: #1e293b; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; font-weight: 700; color: #1e293b; }
h2 { font-size: 15px; font-weight: 700; color: #334155; margin-bottom: 12px;
     text-transform: uppercase; letter-spacing: 0.05em; }
.subtitle { color: #64748b; font-size: 13px; margin-top: 4px; }
.header { background: white; border-radius: 12px; padding: 20px 24px;
          margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
        gap: 14px; margin-bottom: 20px; }
.kpi { background: white; border-radius: 10px; padding: 16px 18px;
       box-shadow: 0 1px 3px rgba(0,0,0,.08); border-left: 4px solid #2563eb; }
.kpi.green { border-left-color: #16a34a; }
.kpi.red   { border-left-color: #dc2626; }
.kpi.amber { border-left-color: #f59e0b; }
.kpi-label { font-size: 11px; color: #64748b; text-transform: uppercase;
             letter-spacing: .05em; font-weight: 600; }
.kpi-value { font-size: 22px; font-weight: 700; margin-top: 4px; color: #1e293b; }
.kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.card { background: white; border-radius: 10px; padding: 20px 22px;
        margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3 { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
img.plot { width: 100%; border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: #f8fafc; font-weight: 700; color: #475569; font-size: 11px;
     text-transform: uppercase; letter-spacing: .04em; padding: 8px 12px;
     border-bottom: 2px solid #e2e8f0; text-align: right; }
th:first-child { text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #f1f5f9;
     color: #334155; text-align: right; }
td:first-child { text-align: left; font-weight: 600; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
.positive { color: #16a34a; font-weight: 600; }
.negative { color: #dc2626; font-weight: 600; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.section-title { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #2563eb; }
.dot.green { background: #16a34a; }
.dot.red   { background: #dc2626; }
.dot.amber { background: #f59e0b; }
.no-trades { color: #94a3b8; font-style: italic; font-size: 13px; padding: 12px 0; }
.footer { text-align: center; color: #94a3b8; font-size: 11px; padding: 16px 0; }
@media (max-width: 768px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
"""


def _img(b64: str) -> str:
    return f'<img class="plot" src="data:image/png;base64,{b64}" alt="chart">'


def _color_class(v: float) -> str:
    return "positive" if v > 0 else "negative" if v < 0 else ""


def _positions_table(positions_df: pd.DataFrame) -> str:
    if positions_df.empty:
        return '<p class="no-trades">Sin posiciones abiertas.</p>'
    rows = ""
    for _, r in positions_df.sort_values("value", ascending=False).iterrows():
        wt = r["weight"] * 100
        wt_cls = _color_class(1)  # always positive
        wt_flag = " ⚠" if wt < 1.0 else ""
        rows += (
            f"<tr>"
            f"<td>{r['ticker']}</td>"
            f"<td>{r['shares']:,.2f}</td>"
            f"<td>€{r['price']:,.3f}</td>"
            f"<td>€{r['value']:,.0f}</td>"
            f"<td><b>{wt:.2f}%{wt_flag}</b></td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Ticker</th><th>Acciones</th><th>Precio</th><th>Valor</th><th>Peso</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _trades_table(ops_df: pd.DataFrame, date: pd.Timestamp) -> str:
    today = ops_df[ops_df["date"].dt.date == date.date()] if not ops_df.empty else pd.DataFrame()
    if today.empty:
        return '<p class="no-trades">Sin operaciones hoy.</p>'
    rows = ""
    for _, r in today.iterrows():
        dir_label = "COMPRA" if r["direction"] == "buy" else "VENTA"
        dir_color = "positive" if r["direction"] == "buy" else "negative"
        rows += (
            f"<tr>"
            f"<td>{r['ticker']}</td>"
            f'<td class="{dir_color}">{dir_label}</td>'
            f"<td>{abs(r['shares']):,.2f}</td>"
            f"<td>€{r['price']:,.3f}</td>"
            f"<td>€{abs(r['value']):,.0f}</td>"
            f"<td>€{r['cost']:,.2f}</td>"
            f"</tr>"
        )
    total_cost = today["cost"].sum()
    rows += (
        f"<tr style='font-weight:700;background:#f8fafc'>"
        f"<td colspan='5'>Total costes</td>"
        f"<td>€{total_cost:,.2f}</td></tr>"
    )
    return (
        "<table><thead><tr>"
        "<th>Ticker</th><th>Dirección</th><th>Acciones</th>"
        "<th>Precio</th><th>Valor</th><th>Coste (CT)</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _metrics_table(metrics: dict) -> str:
    if not metrics:
        return '<p class="no-trades">Historial insuficiente para métricas.</p>'
    labels = {
        "total_return":           ("Retorno Total",       True),
        "annualized_return":      ("Retorno Anualizado",  True),
        "annualized_volatility":  ("Volatilidad Anual",   False),
        "sharpe_ratio":           ("Sharpe Ratio",        None),
        "max_drawdown":           ("Max Drawdown",        False),
        "calmar_ratio":           ("Calmar Ratio",        None),
    }
    rows = ""
    for key, (label, pct_positive) in labels.items():
        v = metrics.get(key)
        if v is None:
            continue
        is_pct = "return" in key or "volatility" in key or "drawdown" in key
        fmt = f"{v:+.2%}" if is_pct else f"{v:.4f}"
        cls = ""
        if pct_positive is True:
            cls = _color_class(v)
        elif pct_positive is False:
            cls = _color_class(-v)  # lower is worse for drawdown/vol
        rows += f"<tr><td>{label}</td><td class='{cls}'><b>{fmt}</b></td></tr>"
    return (
        "<table><thead><tr><th>Métrica</th><th>Valor</th></tr></thead>"
        "<tbody>" + rows + "</tbody></table>"
    )


# ── Main class ─────────────────────────────────────────────────────────────────

@dataclass
class ReportGenerator:
    """Generates the daily HTML dashboard and updates the seguimiento Excel."""

    report_dir: str = "outputs/reports"
    initial_cash: float = 10_000_000.0
    max_leverage: float = 2.0

    def generate(
        self,
        result: dict,
        vl_tracker: VLTracker,
        state_mgr: PortfolioStateManager,
        universe: Universe,
    ) -> Path:
        """Build the full HTML report. Returns the path of the generated file."""
        date: pd.Timestamp = result["date"]
        post_vl: float     = result["post_trade_vl"]
        pre_vl: float      = result["pre_trade_vl"]
        cash: float        = result["cash"]
        positions: dict    = result["positions"]
        today_prices       = result["today_prices"]

        pos_value = post_vl - cash
        leverage  = pos_value / post_vl if post_vl else 0.0
        daily_pnl = post_vl - pre_vl
        daily_ret = pre_vl / self.initial_cash - 1 if pre_vl > 0 else 0.0
        total_ret = post_vl / self.initial_cash - 1

        # Build current positions DataFrame
        pos_rows = []
        for ticker, shares in positions.items():
            if shares != 0:
                price = today_prices.get_price(ticker)
                value = shares * price
                weight = value / post_vl if post_vl else 0.0
                pos_rows.append({"ticker": ticker, "shares": shares,
                                  "price": price, "value": value, "weight": weight})
        positions_df = pd.DataFrame(pos_rows)

        # Load history
        vl_df  = vl_tracker.history()
        ops_df = state_mgr.operations_history()
        metrics = vl_tracker.metrics()

        # ── Plots ──────────────────────────────────────────────
        img_composition  = _plot_composition(positions_df, post_vl)
        img_sector_pie   = _plot_sector_pie(positions_df, universe)
        img_vl           = _plot_vl_history(vl_df, self.initial_cash)
        img_returns      = _plot_daily_returns(vl_df)
        img_drawdown     = _plot_drawdown(vl_df)
        img_leverage     = _plot_leverage_gauge(leverage, self.max_leverage)

        # ── Compliance checks ──────────────────────────────────
        n_below_min = sum(1 for r in pos_rows if r["weight"] < 0.01) if pos_rows else 0
        check_min_weight  = n_below_min == 0
        check_leverage    = leverage <= self.max_leverage
        check_invested    = (1 - cash / post_vl) >= 0.98 if post_vl > 0 else False
        check_no_short    = all(s >= 0 for s in positions.values())

        # ── KPI color ──────────────────────────────────────────
        total_ret_cls  = "green" if total_ret >= 0 else "red"
        lev_cls        = "green" if leverage < 1.5 else "amber" if leverage < 1.9 else "red"

        # ── HTML assembly ──────────────────────────────────────
        html_parts: list[str] = []
        a = html_parts.append

        a(f"""<!DOCTYPE html><html lang="es"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Report — {date.strftime('%d/%m/%Y')}</title>
<style>{_CSS}</style></head><body><div class="container">""")

        # Header
        a(f"""<div class="header">
  <h1>Dashboard de Cartera</h1>
  <p class="subtitle">Fecha de liquidación: <b>{date.strftime('%A %d de %B de %Y')}</b>
    &nbsp;·&nbsp; Grupo 4 — Afi MFC Gestión Cuantitativa</p>
</div>""")

        # KPIs row
        daily_pnl_cls = "green" if daily_pnl >= 0 else "red"
        total_ret_cls2 = "green" if total_ret >= 0 else "red"
        a('<div class="kpis">')
        a(f"""<div class="kpi"><div class="kpi-label">Valor Liquidativo</div>
  <div class="kpi-value">{_eur(post_vl)}</div>
  <div class="kpi-sub">Capital inicial: {_eur(self.initial_cash)}</div></div>""")
        a(f"""<div class="kpi {daily_pnl_cls}"><div class="kpi-label">P&amp;L Hoy</div>
  <div class="kpi-value">{_eur(daily_pnl)}</div>
  <div class="kpi-sub">{_pct(daily_pnl / pre_vl) if pre_vl > 0 else '—'}</div></div>""")
        a(f"""<div class="kpi {total_ret_cls2}"><div class="kpi-label">Retorno Total</div>
  <div class="kpi-value">{_pct(total_ret)}</div>
  <div class="kpi-sub">desde inicio</div></div>""")
        a(f"""<div class="kpi {lev_cls}"><div class="kpi-label">Apalancamiento</div>
  <div class="kpi-value">{leverage*100:.1f}%</div>
  <div class="kpi-sub">límite 200%</div></div>""")
        a(f"""<div class="kpi"><div class="kpi-label">Posiciones</div>
  <div class="kpi-value">{len(pos_rows)}</div>
  <div class="kpi-sub">Operaciones hoy: {result['n_trades']}</div></div>""")
        cash_cls = "red" if cash < 0 else "green"
        a(f"""<div class="kpi {cash_cls}"><div class="kpi-label">Caja</div>
  <div class="kpi-value">{_eur(cash)}</div>
  <div class="kpi-sub">{'Apalancado' if cash < 0 else 'En efectivo'}</div></div>""")
        a('</div>')  # /kpis

        # Compliance badges
        a('<div class="card">')
        a('<div class="section-title"><div class="dot"></div><h2>Cumplimiento Restricciones</h2></div>')
        a('<div class="badges">')
        a(_badge(check_min_weight,  f"Peso mínimo 1% ({n_below_min} violaciones)"))
        a(_badge(check_leverage,    f"Apalancamiento ≤ 200% ({leverage*100:.1f}%)"))
        a(_badge(check_invested,    "Cartera ≥ 98% invertida"))
        a(_badge(check_no_short,    "Sin posiciones cortas"))
        a('</div>')

        # Leverage gauge
        if img_leverage:
            a(f'<div style="margin-top:12px">{_img(img_leverage)}</div>')
        a('</div>')  # /compliance card

        # Portfolio composition
        a('<div class="grid-3">')
        a('<div class="card">')
        a('<div class="section-title"><div class="dot"></div><h2>Composición de la Cartera</h2></div>')
        if img_composition:
            a(_img(img_composition))
        else:
            a('<p class="no-trades">Sin posiciones abiertas.</p>')
        a('</div>')  # /composition card

        a('<div class="card">')
        a('<div class="section-title"><div class="dot green"></div><h2>Por Sector</h2></div>')
        if img_sector_pie:
            a(_img(img_sector_pie))
        else:
            a('<p class="no-trades">Sin datos de sector.</p>')
        a('</div>')  # /sector pie card
        a('</div>')  # /grid-3

        # Positions table
        a('<div class="card">')
        a('<div class="section-title"><div class="dot"></div><h2>Detalle de Posiciones</h2></div>')
        a(_positions_table(positions_df))
        a('</div>')

        # Today's operations
        a('<div class="card">')
        a('<div class="section-title"><div class="dot amber"></div><h2>Operativa del Día</h2></div>')
        a(_trades_table(ops_df, date))
        a('</div>')

        # VL History (only if data)
        if img_vl:
            a('<div class="card">')
            a('<div class="section-title"><div class="dot green"></div><h2>Evolución del VL</h2></div>')
            a(_img(img_vl))
            a('</div>')

        if img_returns or img_drawdown:
            a('<div class="grid-2">')
            if img_returns:
                a('<div class="card">')
                a('<div class="section-title"><div class="dot"></div><h2>Retornos Diarios</h2></div>')
                a(_img(img_returns))
                a('</div>')
            if img_drawdown:
                a('<div class="card">')
                a('<div class="section-title"><div class="dot red"></div><h2>Drawdown</h2></div>')
                a(_img(img_drawdown))
                a('</div>')
            a('</div>')  # /grid-2

        # Metrics
        a('<div class="card">')
        a('<div class="section-title"><div class="dot purple"></div><h2>Métricas de Rendimiento</h2></div>')
        a(_metrics_table(metrics))
        a('</div>')

        # Footer
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        now_str = _dt.now(_ZI("Europe/Madrid")).strftime("%d/%m/%Y %H:%M (Madrid)")
        a(f'<div class="footer">Generado el {now_str} · Afi MFC Gestión Cuantitativa · Grupo 4</div>')
        a('</div></body></html>')  # /container

        # ── Write files ────────────────────────────────────────
        out_dir = Path(self.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        html_content = "\n".join(html_parts)
        dated_path = out_dir / f"{date.strftime('%Y-%m-%d')}_report.html"
        dated_path.write_text(html_content, encoding="utf-8")

        # Always overwrite 'latest.html'
        latest_path = out_dir / "latest.html"
        latest_path.write_text(html_content, encoding="utf-8")

        # ── Seguimiento Excel ──────────────────────────────────
        seguimiento_path = vl_tracker.export_seguimiento(
            positions_history=state_mgr.positions_history().to_dict("records") or None,
            operations_history=state_mgr.operations_history().to_dict("records") or None,
            output_path=str(out_dir / "seguimiento.xlsx"),
        )

        return dated_path
