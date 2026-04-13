"""
Carga datos reales de mercado para la v0.

Interfaz v0:
- download_market_data() recibe tickers y rango temporal.
- Devuelve un diccionario con claves estables:
  'tickers', 'prices', 'returns', 'metadata' y 'transaction_costs'.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
import io
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from Data.universe import get_universe, get_universe_tickers
except ModuleNotFoundError:
    from universe import get_universe, get_universe_tickers

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRICE_FIELDS = ("Close", "Adj Close")
EXECUTION_LOOKAHEAD_DAYS = 5


def _download_from_yfinance(**kwargs) -> tuple[pd.DataFrame, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        raw_data = yf.download(
            progress=False,
            group_by="column",
            threads=False,
            **kwargs,
        )
    return raw_data, buffer.getvalue()



def _provider_message_snippet(provider_output: str, max_lines: int = 3, max_chars: int = 320) -> str:
    lines = [line.strip() for line in str(provider_output).splitlines() if line.strip()]
    if not lines:
        return ""

    snippet = " | yfinance: " + " || ".join(lines[-max_lines:])
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet



def _with_provider_context(message: str, provider_output: str) -> str:
    return f"{message}{_provider_message_snippet(provider_output)}"



def _normalize_tickers(tickers: str | Iterable[str] | None) -> list[str]:
    if tickers is None:
        return get_universe_tickers()

    if isinstance(tickers, str):
        tickers = [tickers]

    normalized_tickers = []
    for ticker in tickers:
        ticker_str = str(ticker).strip()
        if ticker_str and ticker_str not in normalized_tickers:
            normalized_tickers.append(ticker_str)

    return normalized_tickers



def _extract_from_multiindex(raw_data: pd.DataFrame) -> pd.DataFrame:
    first_level = set(raw_data.columns.get_level_values(0))
    second_level = set(raw_data.columns.get_level_values(1))

    for field in PRICE_FIELDS:
        if field in first_level:
            return raw_data[field].copy()
        if field in second_level:
            return raw_data.xs(field, axis=1, level=1).copy()

    raise ValueError("No se encontro una columna Close o Adj Close en la descarga de yfinance.")



def _extract_price_frame(
    raw_data: pd.DataFrame,
    requested_tickers: list[str],
    provider_output: str = "",
) -> pd.DataFrame:
    """Extrae un DataFrame de precios de cierre desde la salida de yfinance."""
    if raw_data.empty:
        raise ValueError(
            _with_provider_context(
                "La descarga de yfinance vino vacia para los tickers solicitados; no hay tickers validos con datos en el rango pedido.",
                provider_output,
            )
        )

    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            prices = _extract_from_multiindex(raw_data)
        except ValueError as exc:
            raise ValueError(_with_provider_context(str(exc), provider_output)) from exc
    else:
        price_column = next((field for field in PRICE_FIELDS if field in raw_data.columns), None)
        if price_column is None:
            raise ValueError(
                _with_provider_context(
                    "No se encontro una columna Close o Adj Close en la descarga de yfinance.",
                    provider_output,
                )
            )
        prices = raw_data[[price_column]].copy()
        if len(requested_tickers) == 1:
            prices.columns = [requested_tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.columns = [str(column) for column in prices.columns]
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.ffill().dropna(how="all")

    if requested_tickers:
        available_tickers = [ticker for ticker in requested_tickers if ticker in prices.columns]
        if available_tickers:
            prices = prices[available_tickers]

    if prices.empty:
        raise ValueError(
            _with_provider_context(
                "La descarga no contiene precios utilizables despues de la limpieza.",
                provider_output,
            )
        )

    return prices



def _build_metadata(available_tickers: list[str]) -> list[dict[str, str]]:
    universe_by_ticker = {etf["ticker"]: dict(etf) for etf in get_universe()}
    metadata = []

    for ticker in available_tickers:
        metadata.append(
            universe_by_ticker.get(
                ticker,
                {
                    "ticker": ticker,
                    "name": ticker,
                    "asset_class": "unknown",
                    "region": "unknown",
                    "role": "risk",
                },
            )
        )

    return metadata



def _normalize_ticker_cell(value: object) -> str:
    s = str(value).strip().strip('"').strip("'")
    return s


def _commission_match_key(ticker: str) -> str:
    """Clave estable para cruzar tickers Excel ↔ yfinance (mayusculas, sin comillas)."""
    return _normalize_ticker_cell(ticker).upper()


def _parse_commission_cell(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Celda de comision vacia o no numerica")
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    s = str(value).strip().replace(" ", "").replace("%", "")
    if not s:
        raise ValueError("Celda de comision vacia")
    s = s.replace(",", ".")
    return float(s)


def _is_commission_table_header(cell0: object, cell1: object) -> bool:
    a = str(cell0).strip().lower()
    if a in ("ticker", "id", "symbol", "activo", "isin", "etf"):
        return True
    b = str(cell1).strip().lower()
    if b in ("comision", "commission", "ct", "tasa", "fee", "coste"):
        return True
    return False


def load_etf_commissions_table_excel(
    path: str | Path,
    sheet_name: int | str = 0,
) -> dict[str, float]:
    """
    Tabla vertical: columnas ticker + comision (cualquier nombre en cabecera opcional).
    Filas vacias o no numericas en comision se ignoran. Coma o punto decimal.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el Excel de comisiones: {p.resolve()}")

    try:
        df = pd.read_excel(p, sheet_name=sheet_name, header=None, engine="openpyxl")
    except ValueError:
        df = pd.read_excel(p, sheet_name=0, header=None, engine="openpyxl")

    if df.shape[1] < 2:
        raise ValueError(f"Se esperan al menos 2 columnas en {p.name}")

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        raw_t = row.iloc[0]
        raw_c = row.iloc[1]
        if pd.isna(raw_t) or str(raw_t).strip() == "":
            continue
        if _is_commission_table_header(raw_t, raw_c):
            continue
        ticker = _normalize_ticker_cell(raw_t)
        if not ticker or ticker.lower() in ("nan", "none"):
            continue
        try:
            out[ticker] = _parse_commission_cell(raw_c)
        except ValueError:
            continue
    if not out:
        raise ValueError(
            f"No se leyeron filas validas (ticker + comision) en {p.name} (hoja {sheet_name!r})."
        )
    return out


def load_etf_commissions_horizontal_excel(path: str | Path) -> dict[str, float]:
    """
    Lee el Excel de comisiones en layout horizontal:
    - Fila 1: ticker (a menudo en pares de columnas con el mismo simbolo).
    - Fila 2: descripcion (ignorada).
    - Fila 3: comision por lado (fraccion del nominal); suele estar solo en la 1ª columna de cada par.
    Acepta coma o punto como separador decimal.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el Excel de comisiones: {p.resolve()}")

    df = pd.read_excel(p, header=None, engine="openpyxl")
    if df.shape[0] < 3:
        raise ValueError(f"El Excel de comisiones debe tener al menos 3 filas: {p}")

    row0 = df.iloc[0]
    row2 = df.iloc[2]
    ncols = int(df.shape[1])
    out: dict[str, float] = {}
    c = 0
    while c < ncols:
        raw_t = row0.iloc[c]
        if pd.isna(raw_t) or str(raw_t).strip() == "":
            c += 1
            continue
        ticker = _normalize_ticker_cell(raw_t)
        raw_cost = row2.iloc[c]
        if pd.isna(raw_cost) and c + 1 < ncols:
            raw_cost = row2.iloc[c + 1]
        if pd.isna(raw_cost):
            raise ValueError(
                f"Fila 3 sin comision para ticker {ticker!r} (columna Excel {c + 1}) en {p.name}"
            )
        out[ticker] = _parse_commission_cell(raw_cost)
        if c + 1 < ncols:
            raw_t2 = row0.iloc[c + 1]
            if not pd.isna(raw_t2) and _normalize_ticker_cell(raw_t2) == ticker:
                c += 2
                continue
        c += 1
    return out


def _resolve_commissions_excel_path(cfg_path: str | None) -> Path | None:
    if cfg_path is None:
        return None
    s = str(cfg_path).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def load_etf_commissions_excel(path: str | Path) -> dict[str, float]:
    """
    Carga comisiones segun config ETF_COMMISSIONS_FORMAT: 'table' o 'horizontal'.
    """
    p = Path(path)
    fmt = "table"
    sheet: int | str = 0
    try:
        import config as _cfg

        fmt = str(getattr(_cfg, "ETF_COMMISSIONS_FORMAT", "table")).strip().lower()
        sheet = getattr(_cfg, "ETF_COMMISSIONS_SHEET", 0)
    except ImportError:
        pass
    if fmt == "horizontal":
        return load_etf_commissions_horizontal_excel(p)
    return load_etf_commissions_table_excel(p, sheet_name=sheet)


def compute_transaction_costs_for_download(prices_df: pd.DataFrame) -> dict[str, float]:
    """
    Comision por ticker: prioridad al Excel en config (tabla o horizontal); si falta ticker,
    usa tasa homogenea TX_COST_* salvo ETF_COMMISSION_REQUIRE_EXCEL_FOR_ALL.
    """
    try:
        import config as _cfg

        excel_attr = getattr(_cfg, "ETF_COMMISSIONS_EXCEL_PATH", None)
        strict = bool(getattr(_cfg, "ETF_COMMISSION_REQUIRE_EXCEL_FOR_ALL", False))
    except ImportError:
        excel_attr = None
        strict = False

    resolved = _resolve_commissions_excel_path(
        str(excel_attr).strip() if excel_attr is not None else None
    )
    if resolved is None:
        return compute_transaction_costs_v0(prices_df)
    if not resolved.is_file():
        if strict:
            raise FileNotFoundError(
                f"ETF_COMMISSION_REQUIRE_EXCEL_FOR_ALL pero no hay archivo: {resolved}"
            )
        warnings.warn(
            f"No se encuentra {resolved}; se usan comisiones homogeneas (TX_COST_*).",
            UserWarning,
            stacklevel=2,
        )
        return compute_transaction_costs_v0(prices_df)

    table = load_etf_commissions_excel(resolved)
    norm_table = {_commission_match_key(k): float(v) for k, v in table.items()}
    homogeneous = compute_transaction_costs_v0(prices_df)
    fallback_rate = float(next(iter(homogeneous.values()))) if homogeneous else 0.0008

    out: dict[str, float] = {}
    missing: list[str] = []
    for col in prices_df.columns:
        t = str(col)
        mk = _commission_match_key(t)
        if mk in norm_table:
            out[t] = norm_table[mk]
        elif t in table:
            out[t] = float(table[t])
        else:
            missing.append(t)
            out[t] = fallback_rate

    if strict and missing:
        raise ValueError(
            "Tickers sin comision en el Excel (ETF_COMMISSION_REQUIRE_EXCEL_FOR_ALL=True): "
            + ", ".join(sorted(missing))
        )
    if missing:
        warnings.warn(
            f"{len(missing)} ticker(s) sin entrada en {resolved.name}; "
            f"usando TX_COST_PER_SIDE acotado ({fallback_rate:.6f}) para: {sorted(missing)}",
            UserWarning,
            stacklevel=2,
        )
    return out


def resolve_transaction_cost_for_ticker(
    ticker: str,
    transaction_costs: dict[str, float] | None,
) -> float:
    """
    ct por lado para un ticker: 1) Excel comisiones (config) si hay fila;
    2) clave en transaction_costs (p. ej. de download_market_data);
    3) TX_COST_PER_SIDE acotado a MIN/MAX.
    """
    tc = transaction_costs or {}
    raw = str(ticker)
    mk = _commission_match_key(raw)
    try:
        import config as _cfg

        excel_attr = getattr(_cfg, "ETF_COMMISSIONS_EXCEL_PATH", None)
        resolved = _resolve_commissions_excel_path(
            str(excel_attr).strip() if excel_attr is not None else None
        )
        if resolved is not None and resolved.is_file():
            table = load_etf_commissions_excel(resolved)
            norm_table = {_commission_match_key(k): float(v) for k, v in table.items()}
            if mk in norm_table:
                return norm_table[mk]
    except Exception:
        pass
    if raw in tc:
        return float(tc[raw])
    for k, v in tc.items():
        if _commission_match_key(str(k)) == mk:
            return float(v)
    try:
        import config as _cfg

        per = float(getattr(_cfg, "TX_COST_PER_SIDE", 0.0008))
        mn = float(getattr(_cfg, "TX_COST_PER_SIDE_MIN", 0.0005))
        mx = float(getattr(_cfg, "TX_COST_PER_SIDE_MAX", 0.0012))
        return max(mn, min(mx, per))
    except Exception:
        return 0.0008


def download_market_data(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    tickers: list[str] | str | None = None,
    auto_adjust: bool = True,
) -> dict:
    """
    Descarga precios reales y calcula retornos simples.

    Devuelve un diccionario listo para los siguientes modulos:
    - tickers: lista final descargada
    - prices: DataFrame de precios
    - returns: DataFrame de retornos porcentuales
    - metadata: lista del universo disponible
    - transaction_costs: ct por ticker (fraccion del nominal por lado); ver Data/comisiones_etfs.xlsx
      (o ruta en config) y fallback TX_COST_*.
    """
    selected_tickers = _normalize_tickers(tickers)
    if not selected_tickers:
        raise ValueError("No se recibio ningun ticker valido para descargar datos.")

    raw_data, provider_output = _download_from_yfinance(
        tickers=selected_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
    )

    prices = _extract_price_frame(raw_data, selected_tickers, provider_output=provider_output)
    prices = prices.dropna(axis=1, how="all")
    available_tickers = [ticker for ticker in selected_tickers if ticker in prices.columns]

    if not available_tickers:
        raise ValueError(
            _with_provider_context(
                "No hay tickers validos con precios descargados en el rango solicitado.",
                provider_output,
            )
        )

    prices = prices[available_tickers]
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    metadata = _build_metadata(available_tickers)
    transaction_costs = compute_transaction_costs_for_download(prices)

    return {
        "tickers": available_tickers,
        "prices": prices,
        "returns": returns,
        "metadata": metadata,
        "transaction_costs": transaction_costs,
    }



def compute_transaction_costs_v0(
    prices_df: pd.DataFrame,
    per_side_pct: float | None = None,
    min_pct: float | None = None,
    max_pct: float | None = None,
) -> dict[str, float]:
    """
    Coste por operacion y lado (ct) como fraccion del nominal, homogeneo por ETF.

    Reserva para fallback cuando un ticker no aparece en el Excel de comisiones.
    Valores desde config: TX_COST_PER_SIDE y limites MIN/MAX.
    """
    try:
        import config as _cfg

        if per_side_pct is None:
            per_side_pct = float(getattr(_cfg, "TX_COST_PER_SIDE", 0.0008))
        if min_pct is None:
            min_pct = float(getattr(_cfg, "TX_COST_PER_SIDE_MIN", 0.0005))
        if max_pct is None:
            max_pct = float(getattr(_cfg, "TX_COST_PER_SIDE_MAX", 0.0012))
    except ImportError:
        per_side_pct = per_side_pct if per_side_pct is not None else 0.0008
        min_pct = min_pct if min_pct is not None else 0.0005
        max_pct = max_pct if max_pct is not None else 0.0012

    rate = max(min_pct, min(max_pct, float(per_side_pct)))

    if prices_df.empty:
        return {}
    return {ticker: rate for ticker in prices_df.columns}



def _build_execution_fallback(clean_prices: pd.DataFrame, last_prices: dict[str, float]) -> dict:
    return {
        "date": clean_prices.index[-1],
        "prices": last_prices,
        "used_next_day": False,
        "execution_source": "last_available",
    }



def get_execution_data_v0(
    tickers: list[str] | str,
    prices_df: pd.DataFrame,
    auto_adjust: bool = True,
) -> dict:
    selected_tickers = _normalize_tickers(tickers)
    if not selected_tickers:
        raise ValueError("No hay tickers para resolver el precio de ejecucion.")

    if prices_df.empty:
        raise ValueError("No hay precios historicos para resolver el precio de ejecucion.")

    clean_prices = prices_df.ffill().dropna(how="all")
    if clean_prices.empty:
        raise ValueError("Los precios historicos no contienen datos utilizables.")

    last_timestamp = pd.Timestamp(clean_prices.index[-1])
    last_date = last_timestamp.normalize()
    next_date = last_date + pd.Timedelta(days=1)
    window_end = next_date + pd.Timedelta(days=EXECUTION_LOOKAHEAD_DAYS)
    last_prices = clean_prices.iloc[-1].to_dict()
    fallback = _build_execution_fallback(clean_prices, last_prices)

    raw_data, _provider_output = _download_from_yfinance(
        tickers=selected_tickers,
        start=next_date.strftime("%Y-%m-%d"),
        end=window_end.strftime("%Y-%m-%d"),
        auto_adjust=auto_adjust,
    )

    if raw_data.empty:
        return fallback

    try:
        future_prices = _extract_price_frame(raw_data, selected_tickers).dropna(axis=1, how="all")
    except ValueError:
        return fallback

    if future_prices.empty:
        return fallback

    normalized_index = pd.Index(pd.to_datetime(future_prices.index).normalize())
    max_execution_date = next_date + pd.Timedelta(days=EXECUTION_LOOKAHEAD_DAYS - 1)
    matching_rows = future_prices.loc[
        (normalized_index >= next_date) & (normalized_index <= max_execution_date)
    ].dropna(how="all")

    if matching_rows.empty:
        return fallback

    execution_prices = last_prices.copy()
    execution_prices.update(matching_rows.iloc[0].dropna().to_dict())
    return {
        "date": matching_rows.index[0],
        "prices": execution_prices,
        "used_next_day": True,
        "execution_source": "future_session",
    }


if __name__ == "__main__":
    market_data = download_market_data()
    print("Tickers descargados:", market_data["tickers"])
    print("Precios shape:", market_data["prices"].shape)
    print("Retornos shape:", market_data["returns"].shape)
