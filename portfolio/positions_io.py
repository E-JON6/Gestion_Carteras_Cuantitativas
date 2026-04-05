"""
Entrada/salida de posiciones en Excel (cartera real en titulos).

Flujo diario:
  1. Mantener results/posiciones_cartera.xlsx con columnas Ticker + Cantidad.
  2. Ejecutar main.py: carga posiciones, calcula NAV y pesos, pipeline + registrador.
  3. Enviar operaciones_rebalanceo_{fecha}.xlsx al bróker.
  4. Tras la ejecucion, actualizar a mano posiciones_cartera.xlsx (o copiar desde
     posiciones_post_rebalanceo_{fecha}.xlsx o posiciones_post_rebalanceo_ultimo.xlsx).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def portfolio_value_eur(positions: dict, prices_row: pd.Series) -> float:
    """Valor de mercado con precios del ultimo dia (Serie por ticker)."""
    v = 0.0
    for ticker, qty in positions.items():
        if qty is None or float(qty) == 0:
            continue
        p = prices_row.get(ticker, np.nan)
        if pd.notna(p):
            v += float(qty) * float(p)
    return v


def weights_from_positions(positions: dict, prices_row: pd.Series) -> dict:
    """Misma idea que backtest/engine._compute_current_weights (pesos, no titulos)."""
    total = portfolio_value_eur(positions, prices_row)
    if total <= 0:
        return {}
    out = {}
    for ticker, qty in positions.items():
        if qty is None or float(qty) == 0:
            continue
        p = prices_row.get(ticker, np.nan)
        if pd.notna(p):
            out[ticker] = float(qty) * float(p) / total
    return out


def _normalize_positions_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("ticker", "id", "isin", "activo"):
            col_map[c] = "Ticker"
        elif cl in ("cantidad", "qty", "quantity", "titulos", "n"):
            col_map[c] = "Cantidad"
    df = df.rename(columns=col_map)
    if "Ticker" not in df.columns or "Cantidad" not in df.columns:
        raise ValueError(
            "El Excel de posiciones necesita columnas reconocibles: "
            "Ticker (o ID) y Cantidad (o Qty)."
        )
    return df[["Ticker", "Cantidad"]].dropna(subset=["Ticker"])


def load_positions_from_excel(path: str | Path) -> dict[str, float]:
    """
    Lee hoja primera o 'Posiciones'. Devuelve {ticker: cantidad}.
    Filas con cantidad 0 se omiten.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        df = pd.read_excel(path, sheet_name="Posiciones")
    except ValueError:
        df = pd.read_excel(path, sheet_name=0)

    df = _normalize_positions_columns(df)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        t = str(row["Ticker"]).strip()
        if not t or t.lower() == "nan":
            continue
        q = row["Cantidad"]
        if pd.isna(q):
            continue
        out[t] = float(q)
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def save_positions_to_excel(
    path: str | Path,
    positions: dict,
    as_of_date=None,
    sheet_name: str = "Posiciones",
) -> Path:
    """Guarda Ticker + Cantidad. Opcional columna Fecha en una segunda hoja Meta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"Ticker": k, "Cantidad": float(v)}
        for k, v in sorted(positions.items())
        if abs(float(v)) > 1e-12
    ]
    df = pd.DataFrame(rows)
    meta = pd.DataFrame()
    if as_of_date is not None:
        meta = pd.DataFrame(
            [{"Campo": "Fecha referencia", "Valor": str(pd.Timestamp(as_of_date))}]
        )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        if not meta.empty:
            meta.to_excel(writer, sheet_name="Meta", index=False)

    return path


def ensure_positions_template(path: str | Path) -> Path:
    """Crea Excel vacio con cabeceras y nota en Meta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=["Ticker", "Cantidad"])
    meta = pd.DataFrame(
        {
            "Instruccion": [
                "Rellena una fila por ticker (ISIN/ticker Yahoo). Cantidad = titulos "
                "(decimales permitidos en XEON.DE). Tras ejecutar ordenes en bróker, "
                "actualiza este archivo o copia desde posiciones_sugeridas_*.xlsx.",
            ]
        }
    )
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Posiciones", index=False)
        meta.to_excel(writer, sheet_name="Meta", index=False)
    return path
