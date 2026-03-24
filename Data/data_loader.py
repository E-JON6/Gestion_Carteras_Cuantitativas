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

import pandas as pd
import yfinance as yf

try:
    from Data.universe import get_universe, get_universe_tickers
except ModuleNotFoundError:
    from universe import get_universe, get_universe_tickers

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
    - transaction_costs: diccionario simple por ticker
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
    transaction_costs = compute_transaction_costs_v0(prices)

    return {
        "tickers": available_tickers,
        "prices": prices,
        "returns": returns,
        "metadata": metadata,
        "transaction_costs": transaction_costs,
    }



def compute_transaction_costs_v0(
    prices_df: pd.DataFrame,
    spread: float = 0.5,
    extra_fee: float = 0.0001,
    window: int = 21,
) -> dict[str, float]:
    if prices_df.empty:
        return {}

    recent_prices = prices_df.ffill().tail(max(window, 1))
    if recent_prices.empty:
        return {ticker: float(extra_fee) for ticker in prices_df.columns}

    daily_costs = 0.5 * spread / recent_prices.replace(0, pd.NA) + extra_fee
    daily_costs = daily_costs.fillna(extra_fee)
    return {ticker: float(cost) for ticker, cost in daily_costs.mean().to_dict().items()}



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
