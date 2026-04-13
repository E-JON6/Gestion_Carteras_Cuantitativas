"""
Regenera Data/comisiones_etfs.xlsx (formato tabla: Ticker, Comision).

- Universo: ETF_UNIVERSE + XEON_TICKER (config).
- Prioridad por ticker: valor ya en el Excel existente; semillas; TX_COST_PER_SIDE acotado.

Ejecutar desde la raiz del proyecto:
    python Data/build_etf_comisiones_excel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg
from Data.data_loader import (
    _commission_match_key,
    _resolve_commissions_excel_path,
    load_etf_commissions_excel,
)

_SEEDED_COMMISSIONS: dict[str, float] = {
    "IWDA.L": 0.00025299,
    "IEMA.L": 0.00075451,
    "IUSN.DE": 0.00093855,
    "XLK": 0.00013465,
    "XLF": 0.00019102,
    "XLV": 0.00016431,
    "XLU": 0.00021575,
    "SOXX": 0.00016372,
    "CIBR": 0.00052463,
    "BOTZ": 0.00023622,
    "LIT": 0.00047705,
    "ITB": 0.00020254,
    "GLD": 0.00017532,
    "SLV": 0.00017602,
    "DBA": 0.00029566,
    "TLT": 0.00015743,
    "IEF": 0.00015203,
    "VGK": 0.00021826,
    "EWJ": 0.00016148,
    "MCHI": 0.00018031,
    "EWZ": 0.0002554,
    "INDA": 0.00019162,
}


def _default_rate_clamped() -> float:
    per = float(getattr(cfg, "TX_COST_PER_SIDE", 0.0008))
    mn = float(getattr(cfg, "TX_COST_PER_SIDE_MIN", 0.0005))
    mx = float(getattr(cfg, "TX_COST_PER_SIDE_MAX", 0.0012))
    return max(mn, min(mx, per))


def _universe_tickers_ordered() -> list[str]:
    risk = sorted(cfg.ETF_UNIVERSE.keys(), key=lambda s: s.upper())
    x = cfg.XEON_TICKER
    if x in risk:
        return risk
    return risk + [x]


def _load_existing_by_match_key(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        tbl = load_etf_commissions_excel(path)
        return {_commission_match_key(k): float(v) for k, v in tbl.items()}
    except Exception:
        return {}


def _resolve_rate(
    ticker: str,
    existing_mk: dict[str, float],
    seeded_mk: dict[str, float],
    default: float,
) -> tuple[float, str]:
    mk = _commission_match_key(ticker)
    if mk in existing_mk:
        return existing_mk[mk], "excel_previo"
    if mk in seeded_mk:
        return seeded_mk[mk], "semilla"
    return default, "TX_COST_*"


def build_excel(overwrite: bool = True) -> Path:
    out_path = _resolve_commissions_excel_path(
        str(getattr(cfg, "ETF_COMMISSIONS_EXCEL_PATH", "Data/comisiones_etfs.xlsx") or "").strip()
        or None
    )
    if out_path is None:
        out_path = _ROOT / "Data" / "comisiones_etfs.xlsx"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path

    existing_mk = _load_existing_by_match_key(out_path)
    seeded_mk = {_commission_match_key(k): v for k, v in _SEEDED_COMMISSIONS.items()}
    default = _default_rate_clamped()

    tickers = _universe_tickers_ordered()
    stats: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for t in tickers:
        rate, src = _resolve_rate(t, existing_mk, seeded_mk, default)
        stats[src] = stats.get(src, 0) + 1
        rows.append({"Ticker": t, "Comision": rate})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_path.exists() and not overwrite:
        raise FileExistsError(str(out_path))
    sheet = getattr(cfg, "ETF_COMMISSIONS_SHEET", 0)
    sheet_name = sheet if isinstance(sheet, str) else "comisiones_etfs"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except PermissionError:
        alt = out_path.with_name(f"{out_path.stem}.generado{out_path.suffix}")
        with pd.ExcelWriter(alt, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(
            f"Aviso: no se pudo sobrescribir {out_path}. Guardado en {alt}",
            file=sys.stderr,
        )
        print(f"Escrito: {alt}")
        print(f"ETFs: {len(tickers)} | por origen: {stats}")
        return alt
    print(f"Escrito: {out_path}")
    print(f"ETFs: {len(tickers)} | por origen: {stats}")
    return out_path


if __name__ == "__main__":
    build_excel()
