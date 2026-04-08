"""Daily portfolio report generator — dark fintech, interactive Plotly charts.

Outputs:
  outputs/reports/YYYY-MM-DD_report.html  — self-contained interactive HTML
  outputs/reports/latest.html             — always current
  outputs/reports/seguimiento.xlsx        — tracking Excel
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.domain.asset import Universe
from src.io.vl_tracker import VLTracker
from src.io.portfolio_state import PortfolioStateManager

# ── Palette ─────────────────────────────────────────────────────────────────────
BG       = "#0f172a"
CARD     = "#1e293b"
CARD2    = "#162032"
BORDER   = "#334155"
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"
BLUE     = "#3b82f6"
PURPLE   = "#a855f7"
GREEN    = "#22c55e"
RED      = "#ef4444"
AMBER    = "#f59e0b"
CYAN     = "#06b6d4"
TEAL     = "#14b8a6"
PINK     = "#ec4899"

ACCENT12 = [BLUE, GREEN, AMBER, PURPLE, CYAN, RED,
            TEAL, PINK, "#f97316", "#84cc16", "#38bdf8", "#e879f9"]

# ── Plotly base layout ───────────────────────────────────────────────────────────
def _layout(**kw) -> dict:
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(family="'Inter','Segoe UI',system-ui,sans-serif",
                  color=TEXT, size=11),
        margin=dict(l=50, r=30, t=44, b=40),
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER,
                        font=dict(color=TEXT, size=11)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
            font=dict(color=MUTED, size=10),
            orientation="h", yanchor="bottom", y=1.02,
        ),
        hovermode="x unified",
        xaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=MUTED, size=9),
        ),
        yaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(color=MUTED, size=9),
        ),
    )
    base.update(kw)
    return base


def _cfg() -> dict:
    return dict(
        displayModeBar=True,
        modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d",
                                "hoverClosestCartesian", "hoverCompareCartesian"],
        displaylogo=False,
        responsive=True,
        toImageButtonOptions=dict(format="svg", scale=2),
    )


def _to_div(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config=_cfg(), div_id=None,
    )


# ── Charts ───────────────────────────────────────────────────────────────────────

def _chart_composition(positions_df: pd.DataFrame, total_value: float) -> str:
    if positions_df.empty:
        return ""
    df = positions_df.sort_values("weight").copy()
    total_weight_pct = df["weight"].sum() * 100

    # Color by weight level
    colors = [GREEN if w >= 0.15 else BLUE if w >= 0.05 else AMBER
              for w in df["weight"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["weight"] * 100,
        y=df["ticker"],
        orientation="h",
        marker=dict(color=colors, opacity=0.90,
                    line=dict(color=BG, width=1.2)),
        text=[f"{w*100:.1f}%" for w in df["weight"]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=10, family="'SF Mono',monospace"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Peso s/NAV: <b>%{x:.2f}%</b><br>"
            "Valor: €%{customdata[0]:,.0f}<br>"
            "Shares: %{customdata[1]:,.0f}<extra></extra>"
        ),
        customdata=list(zip(df["value"], df["shares"])),
        showlegend=False,
    ))

    # Línea mínimo 1%
    fig.add_vline(x=1.0, line_dash="dot", line_color=RED, line_width=1.5,
                  annotation=dict(text="Mín. 1%", font=dict(color=RED, size=9),
                                  yanchor="top"))

    # Anotación de total (puede superar 100% con leverage)
    leverage_note = f"Total: {total_weight_pct:.1f}% s/NAV" + (
        " (>100% = apalancado)" if total_weight_pct > 105 else "")

    fig.update_layout(
        **_layout(
            title=dict(
                text=f"Composición de la Cartera — {leverage_note}",
                font=dict(color=TEXT, size=12),
            ),
            height=max(300, len(df) * 40 + 90),
            xaxis=dict(
                title="Peso sobre NAV (%)",
                range=[0, df["weight"].max() * 100 * 1.18],
                gridcolor=BORDER, tickfont=dict(color=MUTED, size=9),
                ticksuffix="%",
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color=TEXT, size=11, family="'SF Mono',monospace"),
            ),
        )
    )
    return _to_div(fig)


def _chart_sector_treemap(positions_df: pd.DataFrame, universe: Universe) -> str:
    """Treemap por sector — mucho más claro que un donut con 13 sectores únicos."""
    if positions_df.empty:
        return ""
    sector_map = {a.ticker: (a.sector or "Otros") for a in universe.assets}
    df = positions_df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Otros")
    df["pct"] = df["value"] / df["value"].sum() * 100

    # Nodos: root → sector → ticker
    ids, labels, parents, values, customs = [], [], [], [], []

    # Raíz
    ids.append("root")
    labels.append("Cartera")
    parents.append("")
    values.append(0)
    customs.append("")

    # Sectores
    for sector, grp in df.groupby("sector"):
        ids.append(sector)
        labels.append(sector.replace("_", " ").title())
        parents.append("root")
        values.append(grp["value"].sum())
        customs.append(f"{grp['value'].sum()/df['value'].sum()*100:.1f}%")

    # Activos
    for _, row in df.iterrows():
        ids.append(row["ticker"])
        labels.append(row["ticker"])
        parents.append(row["sector"])
        values.append(row["value"])
        customs.append(f"{row['pct']:.1f}%  €{row['value']:,.0f}")

    # Asignar colores consistentes por sector
    unique_sectors = df["sector"].unique()
    sector_color_map = {s: ACCENT12[i % len(ACCENT12)] for i, s in enumerate(unique_sectors)}
    colors = []
    for i in ids:
        if i == "root":
            colors.append(CARD)
        elif i in sector_color_map:
            colors.append(sector_color_map[i])
        else:
            sector = df.loc[df["ticker"] == i, "sector"].values
            colors.append(sector_color_map[sector[0]] if len(sector) else BLUE)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        customdata=customs,
        texttemplate="<b>%{label}</b><br>%{customdata}",
        hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{customdata}<extra></extra>",
        marker=dict(
            colors=colors,
            line=dict(color=BG, width=1.5),
        ),
        textfont=dict(color="white", size=11),
        root_color=BG,
        maxdepth=2,
        branchvalues="total",
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter,sans-serif"),
        title=dict(text="Distribución Sectorial (Treemap)",
                   font=dict(color=TEXT, size=12)),
        height=340,
        margin=dict(l=10, r=10, t=44, b=10),
    )
    return _to_div(fig)


def _chart_vl(vl_df: pd.DataFrame, initial_cash: float) -> str:
    if len(vl_df) < 2:
        return ""
    vl = vl_df.set_index("date")["total_value"]
    rets = vl.pct_change().dropna() * 100
    cummax = vl.cummax()
    dd = (vl - cummax) / cummax * 100

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["Valor Liquidativo (VL)", "Retorno Diario (%)", "Drawdown (%)"],
    )
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 10

    # ── VL line with fill ──
    fig.add_trace(go.Scatter(
        x=vl.index, y=vl.values,
        mode="lines", name="VL",
        line=dict(color=BLUE, width=2.5),
        fill="tonexty" if False else None,
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>VL: €%{y:,.0f}<extra></extra>",
        showlegend=True,
    ), row=1, col=1)
    # Shade above/below initial capital
    fig.add_hrect(y0=initial_cash, y1=vl.max() * 1.01,
                  fillcolor=GREEN, opacity=0.05, layer="below",
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=vl.min() * 0.99, y1=initial_cash,
                  fillcolor=RED, opacity=0.05, layer="below",
                  line_width=0, row=1, col=1)
    fig.add_hline(y=initial_cash, line_dash="dot", line_color=MUTED,
                  line_width=1, row=1, col=1,
                  annotation_text=f"Capital inicial €{initial_cash/1e6:.1f}M",
                  annotation_font_color=MUTED, annotation_font_size=9)

    # ── Daily returns ──
    bar_colors = [GREEN if r >= 0 else RED for r in rets.values]
    fig.add_trace(go.Bar(
        x=rets.index, y=rets.values,
        marker_color=bar_colors, marker_opacity=0.85,
        name="Ret. diario",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:+.2f}%<extra></extra>",
        showlegend=True,
    ), row=2, col=1)

    # ── Drawdown ──
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        line=dict(color=RED, width=1.5),
        fill="tozeroy", fillcolor=f"rgba(239,68,68,0.15)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>DD: %{y:.2f}%<extra></extra>",
        showlegend=True,
    ), row=3, col=1)

    axes_style = dict(gridcolor=BORDER, zerolinecolor=BORDER,
                      tickfont=dict(color=MUTED, size=9))
    fig.update_layout(
        **_layout(
            height=560,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", y=1.03, x=0,
                        font=dict(color=MUTED, size=10)),
            margin=dict(l=60, r=30, t=50, b=40),
        )
    )
    fig.update_xaxes(**axes_style)
    fig.update_yaxes(**axes_style)
    fig.update_yaxes(tickformat="€,.0f", row=1, col=1)
    fig.update_yaxes(tickformat="+.1f", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(tickformat=".1f", ticksuffix="%", row=3, col=1)
    fig.update_traces(marker_line_width=0, row=2, col=1)
    return _to_div(fig)


def _chart_leverage_gauge(leverage: float, max_lev: float = 2.0) -> str:
    fill_color = GREEN if leverage < 1.5 else AMBER if leverage < 1.9 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=leverage * 100,
        number=dict(suffix="%", font=dict(color=fill_color, size=28)),
        delta=dict(reference=100, valueformat=".1f", suffix="%",
                   font=dict(size=12)),
        gauge=dict(
            axis=dict(range=[0, max_lev * 100],
                      tickvals=[0, 50, 100, 150, 200],
                      ticktext=["0%", "50%", "100%", "150%", "200%"],
                      tickfont=dict(color=MUTED, size=9),
                      tickcolor=MUTED),
            bar=dict(color=fill_color, thickness=0.6),
            bgcolor=CARD,
            borderwidth=1, bordercolor=BORDER,
            steps=[
                dict(range=[0, 100],   color="rgba(34,197,94,0.08)"),
                dict(range=[100, 150], color="rgba(251,191,36,0.08)"),
                dict(range=[150, 190], color="rgba(245,158,11,0.12)"),
                dict(range=[190, 200], color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color=RED, width=2),
                           thickness=0.8, value=max_lev * 100),
        ),
        title=dict(text="Apalancamiento actual", font=dict(color=MUTED, size=11)),
    ))
    fig.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter,sans-serif"),
        height=210,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return _to_div(fig)


def _chart_prices_subplots(history_df: pd.DataFrame, positions: dict,
                            universe: Universe, lookback: int = 252) -> str:
    """Grid de subplots con precios absolutos + MA21 + MA63 por cada posición."""
    held = [t for t, s in positions.items() if s != 0]
    if not held or history_df is None or history_df.empty:
        return ""
    valid = [t for t in held if t in history_df.columns]
    if not valid:
        return ""

    sector_map = {a.ticker: (a.sector or "").replace("_", " ").title()
                  for a in universe.assets}

    ncols = 3
    nrows = (len(valid) + ncols - 1) // ncols

    # Subplot titles: ticker + cambio %
    subplot_titles = []
    for t in valid:
        col = history_df[t].dropna().iloc[-lookback:]
        if len(col) >= 2:
            chg = (col.iloc[-1] / col.iloc[0] - 1) * 100
            sign = "+" if chg >= 0 else ""
            subplot_titles.append(f"{t}  {sign}{chg:.1f}%")
        else:
            subplot_titles.append(t)
    # pad to fill grid
    while len(subplot_titles) < nrows * ncols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.07,
    )

    # Style subtitle annotations
    for ann in fig.layout.annotations:
        ann.font.update(color=MUTED, size=10)
        ann.font.family = "Inter,sans-serif"

    axes_style = dict(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont=dict(color=MUTED, size=8),
        showgrid=True, linecolor=BORDER,
    )

    for idx, ticker in enumerate(valid):
        row = idx // ncols + 1
        col_n = idx % ncols + 1
        data = history_df[ticker].dropna().iloc[-lookback:]
        if len(data) < 2:
            continue
        color = ACCENT12[idx % len(ACCENT12)]
        chg = (data.iloc[-1] / data.iloc[0] - 1) * 100
        line_color = GREEN if chg >= 0 else RED

        show_leg = idx == 0  # only show legend items once

        # ── Price line ──
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            mode="lines",
            name="Precio",
            line=dict(color=line_color, width=2),
            showlegend=show_leg,
            legendgroup="price",
            hovertemplate=f"<b>{ticker}</b> %{{x|%d/%m/%Y}}<br>%{{y:.2f}}<extra></extra>",
            fill="tonexty" if False else None,
        ), row=row, col=col_n)

        # ── Área de relleno bajo la línea (sutil) ──
        fill_color = "rgba(34,197,94,0.06)" if chg >= 0 else "rgba(239,68,68,0.06)"
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[data.min() * 0.995] * len(data),
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col_n)
        fig.update_traces(
            selector=dict(name="Precio", legendgroup="price"),
            fill="tonexty", fillcolor=fill_color,
        )

        # ── MA21 ──
        if len(data) >= 21:
            ma21 = data.rolling(21).mean()
            fig.add_trace(go.Scatter(
                x=ma21.index, y=ma21.values,
                mode="lines", name="MA21",
                line=dict(color=AMBER, width=1.2, dash="dash"),
                showlegend=show_leg,
                legendgroup="ma21",
                opacity=0.85,
                hovertemplate=f"MA21 %{{y:.2f}}<extra></extra>",
            ), row=row, col=col_n)

        # ── MA63 ──
        if len(data) >= 63:
            ma63 = data.rolling(63).mean()
            fig.add_trace(go.Scatter(
                x=ma63.index, y=ma63.values,
                mode="lines", name="MA63",
                line=dict(color=PURPLE, width=1.2, dash="dot"),
                showlegend=show_leg,
                legendgroup="ma63",
                opacity=0.85,
                hovertemplate=f"MA63 %{{y:.2f}}<extra></extra>",
            ), row=row, col=col_n)

        # ── Marcador precio actual ──
        fig.add_trace(go.Scatter(
            x=[data.index[-1]], y=[data.iloc[-1]],
            mode="markers",
            marker=dict(color=line_color, size=6, symbol="circle",
                        line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col_n)

    # Hide empty axes
    for i in range(len(valid), nrows * ncols):
        r = i // ncols + 1
        c = i % ncols + 1
        xax = f"xaxis{i + 1}" if i > 0 else "xaxis"
        yax = f"yaxis{i + 1}" if i > 0 else "yaxis"

    fig.update_xaxes(**axes_style)
    fig.update_yaxes(**axes_style)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter,sans-serif", size=9),
        height=max(340, nrows * 240),
        margin=dict(l=40, r=20, t=30, b=60),
        hovermode="x",
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=-0.06,
            font=dict(color=MUTED, size=10),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
    )
    return _to_div(fig)


def _chart_price_grid(history_df: pd.DataFrame, positions: dict,
                      universe: Universe, lookback: int = 126) -> str:
    held = [t for t, s in positions.items() if s != 0]
    if not held or history_df is None or history_df.empty:
        return ""
    valid = [t for t in held if t in history_df.columns]
    if not valid:
        return ""

    ncols = 3
    nrows = (len(valid) + ncols - 1) // ncols
    titles = []
    for t in valid:
        col = history_df[t].dropna().iloc[-lookback:]
        if len(col) >= 2:
            chg = (col.iloc[-1] / col.iloc[0] - 1) * 100
            titles.append(f"<b>{t}</b>  {col.iloc[-1]:.2f}  ({chg:+.1f}%)")
        else:
            titles.append(f"<b>{t}</b>")
    # Pad to fill grid
    while len(titles) < nrows * ncols:
        titles.append("")

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        vertical_spacing=0.12, horizontal_spacing=0.06)

    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 9

    for idx, ticker in enumerate(valid):
        row = idx // ncols + 1
        col_n = idx % ncols + 1
        data = history_df[ticker].dropna().iloc[-lookback:]
        if len(data) < 2:
            continue
        color = ACCENT12[idx % len(ACCENT12)]
        chg = (data.iloc[-1] / data.iloc[0] - 1) * 100
        line_col = GREEN if chg >= 0 else RED

        # Price line
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values, mode="lines", name=ticker,
            line=dict(color=line_col, width=1.8),
            hovertemplate=f"<b>{ticker}</b> %{{x|%d/%m/%Y}}<br>%{{y:.2f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col_n)

        # MA21
        if len(data) >= 21:
            ma21 = data.rolling(21).mean()
            fig.add_trace(go.Scatter(
                x=ma21.index, y=ma21.values, mode="lines",
                line=dict(color=AMBER, width=1, dash="dash"),
                name="MA21", showlegend=(idx == 0),
                hovertemplate=f"MA21 %{{y:.2f}}<extra></extra>",
                opacity=0.8,
            ), row=row, col=col_n)

        # MA63
        if len(data) >= 63:
            ma63 = data.rolling(63).mean()
            fig.add_trace(go.Scatter(
                x=ma63.index, y=ma63.values, mode="lines",
                line=dict(color=PURPLE, width=1, dash="dot"),
                name="MA63", showlegend=(idx == 0),
                hovertemplate=f"MA63 %{{y:.2f}}<extra></extra>",
                opacity=0.8,
            ), row=row, col=col_n)

    axes_style = dict(gridcolor=BORDER, zerolinecolor=BORDER,
                      tickfont=dict(color=MUTED, size=7), showgrid=True)
    fig.update_xaxes(**axes_style)
    fig.update_yaxes(**axes_style)
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter,sans-serif", size=10),
        height=max(320, nrows * 220),
        margin=dict(l=40, r=20, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="h", y=-0.06, x=0,
                    font=dict(color=MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x",
    )
    return _to_div(fig)


def _chart_monthly_heatmap(vl_df: pd.DataFrame) -> str:
    if len(vl_df) < 20:
        return ""
    vl = vl_df.set_index("date")["total_value"]
    monthly = vl.resample("ME").apply(
        lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else np.nan
    ).dropna()
    if monthly.empty:
        return ""
    rows_data, years = [], []
    month_lbls = ["Ene","Feb","Mar","Abr","May","Jun",
                  "Jul","Ago","Sep","Oct","Nov","Dic"]
    unique_years = sorted(set(d.year for d in monthly.index))
    matrix, text_matrix = [], []
    for yr in unique_years:
        row_vals = [np.nan] * 12
        for d, v in monthly.items():
            if d.year == yr:
                row_vals[d.month - 1] = v * 100
        matrix.append(row_vals)
        text_matrix.append([f"{v:.1f}%" if not np.isnan(v) else ""
                             for v in row_vals])

    vmax = max(4.0, np.nanmax(np.abs(matrix)))

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=month_lbls,
        y=[str(yr) for yr in unique_years],
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT),
        colorscale=[
            [0.00, "#7f1d1d"], [0.30, "#dc2626"],
            [0.50, CARD],
            [0.70, "#166534"], [1.00, "#15803d"],
        ],
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(
            tickformat=".0f", ticksuffix="%",
            tickfont=dict(color=MUTED, size=9),
            thickness=12, len=0.8,
            outlinecolor=BORDER, outlinewidth=1,
        ),
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
        showscale=True,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter,sans-serif", size=10),
        title=dict(text="Retornos Mensuales (%)",
                   font=dict(color=TEXT, size=13)),
        height=max(200, len(unique_years) * 44 + 80),
        margin=dict(l=50, r=60, t=50, b=40),
        xaxis=dict(tickfont=dict(color=MUTED, size=10), side="top"),
        yaxis=dict(tickfont=dict(color=MUTED, size=10), autorange="reversed"),
    )
    return _to_div(fig)


def _chart_metrics_radar(metrics: dict) -> str:
    """Spider chart of key normalized metrics."""
    if not metrics:
        return ""
    keys = ["sharpe_ratio", "calmar_ratio", "total_return", "annualized_return"]
    labels = ["Sharpe", "Calmar", "Ret. Total", "Ret. Anual."]
    vals_raw = [metrics.get(k, 0) for k in keys]
    # Normalize to 0-1 range for radar (clip at reasonable bounds)
    bounds = [(0, 3), (0, 3), (0, 5), (0, 0.5)]
    vals_norm = [max(0, min(1, (v - lo) / (hi - lo) if hi > lo else 0))
                 for v, (lo, hi) in zip(vals_raw, bounds)]
    vals_norm.append(vals_norm[0])  # close the shape
    labels_r = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_norm,
        theta=labels_r,
        fill="toself",
        fillcolor=f"rgba(59,130,246,0.18)",
        line=dict(color=BLUE, width=2),
        marker=dict(size=7, color=BLUE),
        hovertemplate="<b>%{theta}</b><br>%{customdata}<extra></extra>",
        customdata=[f"{v:.4f}" for v in vals_raw] + [f"{vals_raw[0]:.4f}"],
        name="Portfolio",
    ))
    fig.update_layout(
        paper_bgcolor=CARD,
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color=MUTED, size=8),
                            gridcolor=BORDER, linecolor=BORDER),
            angularaxis=dict(tickfont=dict(color=TEXT, size=10),
                             gridcolor=BORDER, linecolor=BORDER),
        ),
        showlegend=False,
        title=dict(text="Perfil de Rendimiento", font=dict(color=TEXT, size=12)),
        height=300,
        margin=dict(l=40, r=40, t=50, b=20),
        font=dict(color=TEXT, family="Inter,sans-serif"),
    )
    return _to_div(fig)


# ── HTML primitives ──────────────────────────────────────────────────────────────

def _eur(v: float) -> str:
    return f"€{v:,.0f}"

def _pct(v: float, sign: bool = True) -> str:
    s = "+" if sign and v >= 0 else ""
    return f"{s}{v:.2%}"

def _badge(ok: bool, label: str) -> str:
    bg = "#14532d" if ok else "#450a0a"
    fg = "#4ade80" if ok else "#f87171"
    ic = "✓" if ok else "✗"
    return f'<span class="badge" style="background:{bg};color:{fg}">{ic} {label}</span>'

def _kpi(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (
        f'<div class="kpi" style="border-left-color:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{"<div class=\"kpi-sub\">" + sub + "</div>" if sub else ""}'
        f'</div>'
    )

def _empty(msg: str) -> str:
    return f'<p class="empty-msg">{msg}</p>'

def _table(headers: list[str], rows: str) -> str:
    """Render a table. First column left-aligned, rest right-aligned."""
    ths = "".join(
        f'<th style="text-align:{"left" if i == 0 else "right"}">{h}</th>'
        for i, h in enumerate(headers)
    )
    return (
        f'<div class="tbl-wrap"><table>'
        f'<thead><tr>{ths}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )

def _section(title: str, dot_color: str = "blue") -> str:
    return (
        f'<h2><span class="dot {dot_color}"></span>{title}</h2>'
    )


# ── Table renderers ──────────────────────────────────────────────────────────────

_MONO = "font-family:'SF Mono',ui-monospace,monospace;font-size:12px"
_R    = "text-align:right"
_L    = "text-align:left"

def _tbl_positions(positions_df: pd.DataFrame) -> str:
    if positions_df.empty:
        return _empty("Sin posiciones abiertas.")
    rows = ""
    total_pct = positions_df["weight"].sum() * 100
    for _, r in positions_df.sort_values("value", ascending=False).iterrows():
        wt = r["weight"] * 100
        wt_col = GREEN if wt >= 5 else AMBER if wt >= 1 else RED
        flag = " ⚠" if wt < 1.0 else ""
        rows += (
            f"<tr>"
            f"<td style='{_L};{_MONO};font-weight:700'>{r['ticker']}</td>"
            f"<td style='{_R};{_MONO}'>{r['shares']:,.2f}</td>"
            f"<td style='{_R};{_MONO}'>€{r['price']:,.3f}</td>"
            f"<td style='{_R};{_MONO};font-weight:700'>€{r['value']:,.0f}</td>"
            f"<td style='{_R};font-weight:700;color:{wt_col}'>{wt:.2f}%{flag}</td>"
            f"</tr>"
        )
    rows += (
        f"<tr style='border-top:1px solid {BORDER}'>"
        f"<td style='{_L};color:{MUTED};font-size:11px' colspan='3'>Total peso s/NAV</td>"
        f"<td style='{_R};{_MONO};color:{MUTED}'></td>"
        f"<td style='{_R};font-weight:700;color:{AMBER if total_pct > 105 else GREEN}'>"
        f"{total_pct:.1f}%</td></tr>"
    )
    return _table(["Ticker", "Acciones", "Precio", "Valor", "Peso s/NAV"], rows)


def _strategy_pill(strategy: str) -> str:
    """Small pill badge for strategy name."""
    colors = {
        "E4":              (AMBER,  "#451a03"),
        "transicion":      (PURPLE, "#2e1065"),
        "merton_mom":      (CYAN,   "#083344"),
        "merton_custom":   (CYAN,   "#083344"),
        "merton_full":     (BLUE,   "#1e3a5f"),
        "merton":          (TEAL,   "#134e4a"),
        "merton_dual_mom": (GREEN,  "#14532d"),
        "buy_and_hold":    (MUTED,  CARD2),
    }
    fg, bg = colors.get(strategy, (MUTED, CARD2))
    label = {
        "E4": "E4 · IUSE.L",
        "transicion": "Transición",
    }.get(strategy, strategy)
    return (
        f'<span style="display:inline-block;padding:1px 7px;border-radius:9px;'
        f'font-size:10px;font-weight:600;letter-spacing:.03em;'
        f'background:{bg};color:{fg};white-space:nowrap">{label}</span>'
    )


def _tbl_trades_today(ops_df: pd.DataFrame, date: pd.Timestamp) -> str:
    today = (ops_df[ops_df["date"].dt.date == date.date()]
             if not ops_df.empty else pd.DataFrame())
    if today.empty:
        return _empty("Sin operaciones ejecutadas hoy.")

    has_strategy = "strategy" in today.columns
    rows = ""

    # Group: transition trades first, then strategy trades
    if has_strategy:
        transition = today[today["strategy"].isin(["transicion", "E4"])]
        new_trades = today[~today["strategy"].isin(["transicion", "E4"])]
        sections = []
        if not transition.empty:
            sections.append(("Liquidación estrategia anterior", transition))
        if not new_trades.empty:
            sections.append(("Nuevas posiciones", new_trades))
    else:
        sections = [("", today)]

    for label, sub_df in sections:
        if label:
            rows += (
                f"<tr><td colspan='7' style='padding:10px 0 4px;color:{MUTED};"
                f"font-size:11px;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:.05em;border:none'>{label}</td></tr>"
            )
        for _, r in sub_df.iterrows():
            dir_lbl = "COMPRA" if r["direction"] == "buy" else "VENTA"
            dir_col = GREEN if r["direction"] == "buy" else RED
            strat = _strategy_pill(r["strategy"]) if has_strategy else ""
            rows += (
                f"<tr>"
                f"<td style='{_L};{_MONO};font-weight:700'>{r['ticker']}</td>"
                f"<td style='{_L};font-weight:700;color:{dir_col}'>{dir_lbl}</td>"
                f"<td style='{_R};{_MONO}'>{abs(r['shares']):,.2f}</td>"
                f"<td style='{_R};{_MONO}'>€{r['price']:,.3f}</td>"
                f"<td style='{_R};{_MONO}'>€{abs(r['value']):,.0f}</td>"
                f"<td style='{_R};{_MONO}'>€{r['cost']:,.2f}</td>"
                f"<td style='{_L}'>{strat}</td>"
                f"</tr>"
            )

    tc = today["cost"].sum()
    rows += (
        f"<tr style='border-top:1px solid {BORDER}'>"
        f"<td colspan='5' style='color:{MUTED};font-size:11px'>Total costes del día</td>"
        f"<td style='{_R};{_MONO};font-weight:700'>€{tc:,.2f}</td>"
        f"<td></td></tr>"
    )
    return _table(["Ticker", "Dir.", "Acciones", "Precio", "Valor Noc.", "Coste CT", ""], rows)


def _tbl_ops_history(ops_df: pd.DataFrame) -> str:
    if ops_df.empty:
        return _empty("Sin historial de operaciones.")

    has_strategy = "strategy" in ops_df.columns
    sorted_df = ops_df.sort_values("date", ascending=False)

    # Group by strategy for clear visual separation
    rows = ""
    current_strat = None
    for _, r in sorted_df.iterrows():
        strat = r.get("strategy", "—") if has_strategy else "—"

        # Section header when strategy changes (chronologically descending)
        if has_strategy and strat != current_strat:
            current_strat = strat
            strat_label = {
                "merton_mom":      "Merton Momentum (multi-asset)",
                "merton_custom":   "Merton Custom (top-N + momentum)",
                "merton_full":     "Merton Full (vol regime + frozen)",
                "merton":          "Merton (base)",
                "merton_dual_mom": "Merton Dual Momentum",
                "buy_and_hold":    "Buy & Hold",
                "transicion":      "Transición de estrategia",
                "E4":              "Estrategia anterior — E4 (IUSE.L)",
            }.get(strat, strat)
            strat_dot = {
                "merton_mom": CYAN, "merton_custom": CYAN,
                "merton_full": BLUE, "merton": TEAL,
                "merton_dual_mom": GREEN, "buy_and_hold": MUTED,
                "transicion": PURPLE, "E4": AMBER,
            }.get(strat, MUTED)
            rows += (
                f"<tr><td colspan='8' style='padding:14px 0 6px;border:none'>"
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:{strat_dot};margin-right:8px;"
                f"vertical-align:middle'></span>"
                f"<span style='color:{TEXT};font-size:12px;font-weight:700;"
                f"letter-spacing:.02em'>{strat_label}</span>"
                f"</td></tr>"
            )

        dir_col = GREEN if r["direction"] == "buy" else RED
        dir_lbl = "COMPRA" if r["direction"] == "buy" else "VENTA"
        pill = _strategy_pill(strat) if has_strategy else ""
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
        f"<td colspan='6' style='color:{MUTED};font-size:11px'>"
        f"Total: {n} operaciones</td>"
        f"<td style='{_R};{_MONO};font-weight:700'>€{tc:,.2f}</td>"
        f"<td></td></tr>"
    )
    return _table(["Fecha", "Ticker", "Dir.", "Acciones", "Precio",
                   "Valor Noc.", "Coste CT", ""], rows)


def _tbl_metrics(metrics: dict) -> str:
    if not metrics:
        return _empty("Historial insuficiente.")
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
            f"<tr>"
            f"<td style='{_L}'>{label}</td>"
            f"<td style='{_R};{_MONO};color:{color};font-weight:700'>{fmt}</td>"
            f"</tr>"
        )
    return _table(["Métrica", "Valor"], rows)


def _tbl_universe(universe: Universe) -> str:
    rows = ""
    for a in sorted(universe.assets, key=lambda x: x.ticker):
        ct_bp = (a.transaction_cost or 0) * 10_000
        rows += (
            f"<tr><td class='mono bold'>{a.ticker}</td>"
            f"<td>{a.name or '—'}</td>"
            f"<td>{a.sector or '—'}</td>"
            f"<td>{a.asset_type or '—'}</td>"
            f"<td class='right mono'>{ct_bp:.2f} bp</td>"
            f"</tr>"
        )
    return _table(["Ticker", "Nombre", "Sector", "Tipo", "CT (bp)"], rows)


def _tbl_strategy_params(strategy) -> str:
    if strategy is None:
        return _empty("Estrategia no disponible.")
    rows = ""
    skip = {"universe", "name"}
    for f in dataclasses.fields(strategy):
        if f.name.startswith("_") or f.name in skip:
            continue
        v = getattr(strategy, f.name, None)
        if callable(v):
            v = type(v).__name__
        elif hasattr(v, "__class__") and "estimator" in type(v).__name__.lower():
            v = type(v).__name__
        color = CYAN
        rows += (
            f"<tr><td class='mono' style='color:{MUTED}'>{f.name}</td>"
            f"<td class='mono' style='color:{color};font-weight:700'>{v}</td></tr>"
        )
    return _table(["Parámetro", "Valor"], rows)


# ── CSS + JS ─────────────────────────────────────────────────────────────────────

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
/* header */
.header{{display:flex;justify-content:space-between;align-items:flex-start;
  padding:16px 0 14px;border-bottom:1px solid var(--border);margin-bottom:18px}}
.header-title h1{{font-size:22px;font-weight:800;letter-spacing:-.02em}}
.header-title .sub{{color:var(--muted);font-size:12px;margin-top:3px}}
.badge-row{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;margin-top:4px}}
/* kpis */
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
/* tabs */
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
/* cards */
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:16px 18px;margin-bottom:12px}}
.card h2{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:6px}}
.dot{{width:6px;height:6px;border-radius:50%;background:var(--blue);flex-shrink:0}}
.dot.green{{background:var(--green)}}.dot.red{{background:var(--red)}}
.dot.amber{{background:var(--amber)}}.dot.purple{{background:var(--purple)}}
.dot.cyan{{background:var(--cyan)}}
/* grid */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.g3{{display:grid;grid-template-columns:2fr 1fr;gap:12px}}
.g4{{display:grid;grid-template-columns:3fr 1fr;gap:12px}}
.g4r{{display:grid;grid-template-columns:1fr 3fr;gap:12px}}
/* plot container — needed for Plotly */
.plotly-div{{width:100%;min-height:80px}}
/* tables */
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
tr.total-row td{{font-weight:700;color:var(--muted);font-size:11px;
  border-top:1px solid var(--border)}}
/* badges */
.badge{{font-size:10px;font-weight:700;padding:3px 9px;border-radius:20px;
  letter-spacing:.02em}}
.badges{{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px}}
/* misc */
.empty-msg{{color:var(--muted);font-style:italic;font-size:12px;padding:8px 0}}
.section-note{{color:var(--muted);font-size:11px;margin-bottom:10px;line-height:1.7}}
.footer{{text-align:center;color:var(--muted);font-size:10px;padding:18px 0 6px;
  border-top:1px solid var(--border);margin-top:8px}}
/* responsive */
@media(max-width:768px){{.g2,.g3,.g4,.g4r{{grid-template-columns:1fr}}
  .kpis{{grid-template-columns:repeat(2,1fr)}}}}
"""

JS = """
function showTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
// Responsive Plotly resize on tab show
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


# ── Main class ────────────────────────────────────────────────────────────────────

@dataclass
class ReportGenerator:
    report_dir: str = "outputs/reports"
    initial_cash: float = 10_000_000.0
    max_leverage: float = 2.0

    def generate(
        self,
        result: dict,
        vl_tracker: VLTracker,
        state_mgr: PortfolioStateManager,
        universe: Universe,
        strategy=None,
    ) -> Path:
        date: pd.Timestamp      = result["date"]
        post_vl: float          = result["post_trade_vl"]
        pre_vl: float           = result["pre_trade_vl"]
        cash: float             = result["cash"]
        positions: dict         = result["positions"]
        today_prices            = result["today_prices"]
        history_df: pd.DataFrame = result.get("history_df")

        pos_value = post_vl - cash
        leverage  = pos_value / post_vl if post_vl else 0.0
        daily_pnl = post_vl - pre_vl
        total_ret = post_vl / self.initial_cash - 1

        pos_rows = []
        for ticker, shares in positions.items():
            if shares != 0:
                price  = today_prices.get_price(ticker)
                value  = shares * price
                weight = value / post_vl if post_vl else 0.0
                pos_rows.append({"ticker": ticker, "shares": shares,
                                  "price": price, "value": value, "weight": weight})
        positions_df = pd.DataFrame(pos_rows)

        vl_df   = vl_tracker.history()
        ops_df  = state_mgr.operations_history()
        metrics = vl_tracker.metrics()

        # ── Charts ──────────────────────────────────────────
        ch_comp      = _chart_composition(positions_df, post_vl)
        ch_sector    = _chart_sector_treemap(positions_df, universe)
        ch_vl        = _chart_vl(vl_df, self.initial_cash)
        ch_gauge     = _chart_leverage_gauge(leverage, self.max_leverage)
        ch_heatmap   = _chart_monthly_heatmap(vl_df)
        ch_radar     = _chart_metrics_radar(metrics)
        ch_prices    = _chart_prices_subplots(history_df, positions, universe)

        # ── Compliance ───────────────────────────────────────
        n_below = sum(1 for r in pos_rows if r["weight"] < 0.01)
        ok_min  = n_below == 0
        ok_lev  = leverage <= self.max_leverage
        ok_inv  = (1 - cash / post_vl) >= 0.98 if post_vl > 0 else False
        ok_srt  = all(s >= 0 for s in positions.values())

        pnl_col = GREEN if daily_pnl >= 0 else RED
        ret_col = GREEN if total_ret >= 0 else RED
        lev_col = GREEN if leverage < 1.5 else AMBER if leverage < 1.9 else RED

        # ── HTML ─────────────────────────────────────────────
        p: list[str] = []
        a = p.append

        a(f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio · {date.strftime('%d/%m/%Y')}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">""")

        # Header
        strat_name = strategy.name if strategy else "—"
        a(f"""<div class="header">
  <div class="header-title">
    <h1>Portfolio Dashboard</h1>
    <div class="sub">{date.strftime('%A, %d de %B de %Y')} &nbsp;·&nbsp;
    <b>{strat_name}</b> &nbsp;·&nbsp; Afi MFC Gestión Cuantitativa &nbsp;·&nbsp; Grupo 4</div>
  </div>
  <div class="badge-row">
    {_badge(ok_min,  "Peso mín. 1%")}
    {_badge(ok_lev,  f"Leverage ≤200%")}
    {_badge(ok_inv,  "Cartera invertida")}
    {_badge(ok_srt,  "Sin cortos")}
  </div>
</div>""")

        # KPIs
        pnl_sign = "+" if daily_pnl >= 0 else ""
        a('<div class="kpis">')
        a(_kpi("Valor Liquidativo", _eur(post_vl),  f"Inicial: {_eur(self.initial_cash)}", BLUE))
        a(_kpi("P&L Hoy",    f"{pnl_sign}{_eur(daily_pnl)}", _pct(daily_pnl/pre_vl) if pre_vl else "—", pnl_col))
        a(_kpi("Retorno Total",     _pct(total_ret),   "desde inicio", ret_col))
        a(_kpi("Apalancamiento",    f"{leverage*100:.1f}%", f"límite {self.max_leverage*100:.0f}%", lev_col))
        a(_kpi("Posiciones",        str(len(pos_rows)), f"Trades hoy: {result['n_trades']}", CYAN))
        cash_col = RED if cash < 0 else GREEN
        a(_kpi("Caja",             _eur(cash), "apalancado" if cash < 0 else "libre", cash_col))
        if metrics:
            sr = metrics.get("sharpe_ratio", 0)
            md = metrics.get("max_drawdown", 0)
            a(_kpi("Sharpe",        f"{sr:+.3f}", "riesgo/retorno", GREEN if sr > 0 else RED))
            a(_kpi("Max Drawdown",  _pct(md), "caída máx. desde pico", RED))
        a('</div>')

        # Tabs
        a("""<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('t-dash',this)">📊 Dashboard</button>
  <button class="tab-btn" onclick="showTab('t-perf',this)">📈 Rendimiento</button>
  <button class="tab-btn" onclick="showTab('t-precios',this)">💹 Precios</button>
  <button class="tab-btn" onclick="showTab('t-cartera',this)">🗂 Cartera</button>
  <button class="tab-btn" onclick="showTab('t-ops',this)">⚡ Operativa</button>
  <button class="tab-btn" onclick="showTab('t-strat',this)">⚙️ Estrategia</button>
</div>""")

        # ════ TAB: DASHBOARD ════════════════════════════════
        a('<div id="t-dash" class="tab-content active">')

        # Compliance + gauge
        a('<div class="card">')
        a(_section("Cumplimiento de Restricciones"))
        a('<div class="badges">')
        a(_badge(ok_min,  f"Peso mínimo 1%  ({n_below} violaciones)"))
        a(_badge(ok_lev,  f"Apalancamiento ≤ 200%  ({leverage*100:.1f}%)"))
        a(_badge(ok_inv,  f"Cartera ≥ 98% invertida  ({(1-cash/post_vl)*100:.1f}% actual)"))
        a(_badge(ok_srt,  "Sin posiciones cortas"))
        a('</div>')
        if ch_gauge:
            a(f'<div style="margin-top:4px">{ch_gauge}</div>')
        a('</div>')

        a('<div class="g3">')
        a('<div class="card">')
        a(_section("Composición por Peso"))
        a(ch_comp if ch_comp else _empty("Sin posiciones."))
        a('</div>')
        a('<div class="card">')
        a(_section("Distribución Sectorial", "green"))
        a(ch_sector if ch_sector else _empty("Sin datos."))
        a('</div>')
        a('</div>')

        a('<div class="card">')
        a(_section("Posiciones Actuales", "cyan"))
        a(_tbl_positions(positions_df))
        a('</div>')

        a('</div>')  # /t-dash

        # ════ TAB: RENDIMIENTO ════════════════════════════════
        a('<div id="t-perf" class="tab-content">')

        a('<div class="g4">')
        a('<div>')  # left col

        if ch_vl:
            a('<div class="card">')
            a(_section("Evolución del VL · Retornos Diarios · Drawdown", "green"))
            a(ch_vl)
            a('</div>')
        else:
            a('<div class="card">')
            a(_section("Evolución del VL", "green"))
            a(_empty("Se necesitan ≥ 2 días de historial para mostrar esta gráfica."))
            a('</div>')

        if ch_heatmap:
            a('<div class="card">')
            a(_section("Retornos Mensuales", "amber"))
            a(ch_heatmap)
            a('</div>')

        a('</div>')  # /left col

        a('<div>')  # right col
        a('<div class="card">')
        a(_section("Métricas", "purple"))
        a(_tbl_metrics(metrics))
        a('</div>')

        if ch_radar:
            a('<div class="card">')
            a(_section("Perfil de Rendimiento", "blue"))
            a(ch_radar)
            a('</div>')

        if len(vl_df) >= 2:
            vl_s = vl_df.set_index("date")["total_value"]
            rets = vl_s.pct_change().dropna() * 100
            win_rate = (rets > 0).mean() * 100
            avg_pos  = rets[rets > 0].mean() if (rets > 0).any() else 0
            avg_neg  = rets[rets < 0].mean() if (rets < 0).any() else 0
            rows_stat = (
                f"<tr><td>Días positivos</td>"
                f"<td class='right mono' style='color:{GREEN}'>{win_rate:.1f}%</td></tr>"
                f"<tr><td>Media día +</td>"
                f"<td class='right mono' style='color:{GREEN}'>{avg_pos*100:+.2f}%</td></tr>"
                f"<tr><td>Media día −</td>"
                f"<td class='right mono' style='color:{RED}'>{avg_neg*100:+.2f}%</td></tr>"
                f"<tr><td>Mejor día</td>"
                f"<td class='right mono' style='color:{GREEN}'>{rets.max():+.2f}%</td></tr>"
                f"<tr><td>Peor día</td>"
                f"<td class='right mono' style='color:{RED}'>{rets.min():+.2f}%</td></tr>"
            )
            a('<div class="card">')
            a(_section("Estadísticos Diarios", "red"))
            a(_table(["Estadístico", "Valor"], rows_stat))
            a('</div>')

        a('</div>')  # /right col
        a('</div>')  # /g4
        a('</div>')  # /t-perf

        # ════ TAB: PRECIOS ═════════════════════════════════════
        a('<div id="t-precios" class="tab-content">')
        a('<p class="section-note">Precios de cierre ajustados · últimos 252 días hábiles · '
          '<span style="color:' + AMBER + '">── MA21</span> · '
          '<span style="color:' + PURPLE + '">·· MA63</span> · '
          'El marcador indica el precio actual. Zoom y hover interactivos.</p>')

        if ch_prices:
            a('<div class="card">')
            a(_section("Precios Históricos de las Posiciones Actuales"))
            a(ch_prices)
            a('</div>')
        else:
            a('<div class="card">')
            a(_empty("Sin datos de precios históricos disponibles."))
            a('</div>')

        a('</div>')  # /t-precios

        # ════ TAB: CARTERA ══════════════════════════════════════
        a('<div id="t-cartera" class="tab-content">')

        a('<div class="card">')
        a(_section("Detalle de Posiciones", "cyan"))
        a(_tbl_positions(positions_df))
        a('</div>')

        a('<div class="g2">')
        a('<div class="card">')
        a(_section("Composición por Peso"))
        a(ch_comp if ch_comp else _empty("Sin posiciones."))
        a('</div>')
        a('<div class="card">')
        a(_section("Distribución Sectorial", "green"))
        a(ch_sector if ch_sector else _empty("Sin datos."))
        a('</div>')
        a('</div>')

        a('</div>')  # /t-cartera

        # ════ TAB: OPERATIVA ════════════════════════════════════
        a('<div id="t-ops" class="tab-content">')

        a('<div class="card">')
        a(_section("Operaciones de Hoy", "amber"))
        a(_tbl_trades_today(ops_df, date))
        a('</div>')

        a('<div class="card">')
        a(_section("Historial Completo de Operaciones"))
        a(_tbl_ops_history(ops_df))
        a('</div>')

        a('</div>')  # /t-ops

        # ════ TAB: ESTRATEGIA ═══════════════════════════════════
        a('<div id="t-strat" class="tab-content">')

        a('<div class="g2">')

        a('<div class="card">')
        a(_section("Parámetros de la Estrategia", "cyan"))
        a(_tbl_strategy_params(strategy))
        a('</div>')

        a('<div class="card">')
        a(_section("Pipeline de Señales", "purple"))
        strat_gamma = getattr(strategy, "gamma", "—") if strategy else "—"
        rebal = getattr(strategy, "rebalance_every", 21) if strategy else 21
        band  = getattr(strategy, "band_scale", 2.0) if strategy else 2.0
        lkbk  = getattr(strategy, "momentum_lookback", 252) if strategy else 252
        skip  = getattr(strategy, "momentum_skip", 21) if strategy else 21
        mrf   = getattr(strategy, "max_risky_fraction", 1.0) if strategy else 1.0
        a(f"""<p class="section-note" style="font-size:12px;color:var(--text);line-height:2">
<b>Merton-Davis-Norman + Momentum</b> — maximización utilidad CRRA (γ={strat_gamma})<br>
<span style='color:{MUTED};font-size:11px'>
  Rebalancea cada <b>{rebal} días hábiles</b> ó cuando el apalancamiento supera el límite ó
  en el primer día (cartera vacía). Fracción máxima en activos de riesgo: <b>{mrf*100:.0f}%</b>.
</span><br>
<b style='color:{MUTED};font-size:10px;text-transform:uppercase;letter-spacing:.06em'>
  Condiciones de disparo (OR)
</b><br>
<span style='color:{AMBER}'>⚡</span> Cada {rebal} días hábiles (rebalanceo periódico)<br>
<span style='color:{RED}'>⚡</span> LeverageGuard dispara si el apalancamiento supera max_leverage<br>
<span style='color:{GREEN}'>⚡</span> Primer día (cartera vacía → inversión inicial)<br>
<br>
<b style='color:{MUTED};font-size:10px;text-transform:uppercase;letter-spacing:.06em'>
  Pipeline de señales (cuando se dispara)
</b><br>
<span style='color:{CYAN}'>①</span> <b>MertonSignaler</b> — resuelve el problema de Merton: w* = (1/γ)·Σ⁻¹·(μ−r)
  con estimadores JamesStein (μ̂) y Ledoit-Wolf (Σ̂), cap en {mrf*100:.0f}% activos de riesgo<br>
<span style='color:{AMBER}'>②</span> <b>MomentumSignaler</b> — escala w_i por α_i ∈ [{getattr(strategy,"momentum_alpha_min",0.3)}, 1]
  según retorno relativo de los últimos {lkbk} días (skip {skip}d para evitar reversión de corto plazo)<br>
<span style='color:{PURPLE}'>③</span> <b>NoTradeBandFilter</b> (band_scale={band}) — elimina señales donde |w_actual − w*| &lt; banda DN,
  reduciendo turnover sin coste de oportunidad significativo<br>
<span style='color:{GREEN}'>④</span> <b>fill_defensive</b> — asigna el peso residual (1 − Σw_riesgo) al activo monetario XEON.DE,
  garantizando cartera 100% invertida<br>
<span style='color:{RED}'>⑤</span> <b>MinWeightFilter</b> — descarta posiciones con w &lt; 1% (restricción explícita AFI)<br>
<span style='color:{MUTED}'>⑥</span> <b>CostAwareFilter</b> — cancela el trade si el beneficio esperado del rebalanceo
  no cubre el coste de transacción CT_i<br>
</p>""")
        a('</div>')

        a('</div>')  # /g2

        a('<div class="card">')
        a(_section("Universo de Activos — jaime.yaml", "green"))
        a(_tbl_universe(universe))
        a('</div>')

        a('</div>')  # /t-strat

        # Footer
        now_str = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%d/%m/%Y %H:%M (Madrid)")
        a(f'<div class="footer">Generado el {now_str} &nbsp;·&nbsp; Afi MFC Gestión Cuantitativa &nbsp;·&nbsp; Grupo 4</div>')
        a(f'<script>{JS}</script>')
        a('</div></body></html>')

        # ── Write ────────────────────────────────────────────
        out_dir = Path(self.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        html = "\n".join(p)
        dated = out_dir / f"{date.strftime('%Y-%m-%d')}_report.html"
        dated.write_text(html, encoding="utf-8")
        (out_dir / "latest.html").write_text(html, encoding="utf-8")

        vl_tracker.export_seguimiento(
            positions_history=state_mgr.positions_history().to_dict("records") or None,
            operations_history=state_mgr.operations_history().to_dict("records") or None,
            output_path=str(out_dir / "seguimiento.xlsx"),
        )
        return dated
