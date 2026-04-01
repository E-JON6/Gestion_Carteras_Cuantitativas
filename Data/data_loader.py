"""
Carga y preparación de datos de mercado para el pipeline BL-Omega.

El módulo mantiene las funciones v0 usadas por el repo actual, pero además expone
helpers más estrictos para el nuevo motor multiactivo:
- `load_universe_data()` detecta fecha de inicio válida y construye calendario limpio.
- `build_lambda_dict()` devuelve lambdas defensibles por ticker.
- `get_execution_data_v0()` busca la próxima sesión disponible para operativa.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import redirect_stderr, redirect_stdout
import io

import pandas as pd
import yfinance as yf

try:
    from Data.universe import get_defensive_ticker, get_universe, get_universe_tickers
except ModuleNotFoundError:
    from universe import get_defensive_ticker, get_universe, get_universe_tickers

PRICE_FIELDS = ("Close", "Adj Close")
EXECUTION_LOOKAHEAD_DAYS = 5
DEFAULT_RISK_LAMBDA = 0.0020
DEFAULT_DEFENSIVE_LAMBDA = 0.0005
DEFAULT_MAX_FFILL_DAYS = 1
DEFAULT_MIN_HISTORY_DAYS = 252
DEFAULT_MIN_RISK_ASSETS = 15


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

    normalized_tickers: list[str] = []
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

    raise ValueError("No se encontró una columna Close o Adj Close en la descarga de yfinance.")



def _coerce_price_frame(prices: pd.DataFrame, ffill_limit: int | None = None) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.columns = [str(column) for column in prices.columns]
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    if ffill_limit == 0:
        return prices.dropna(how="all")
    if ffill_limit is not None:
        prices = prices.ffill(limit=max(ffill_limit, 0))

    return prices.dropna(how="all")



def _extract_price_frame(
    raw_data: pd.DataFrame,
    requested_tickers: Sequence[str],
    provider_output: str = "",
    *,
    ffill_limit: int | None = DEFAULT_MAX_FFILL_DAYS,
) -> pd.DataFrame:
    """Extrae un DataFrame de precios de cierre desde la salida de yfinance."""
    if raw_data.empty:
        raise ValueError(
            _with_provider_context(
                "La descarga de yfinance vino vacía para los tickers solicitados; no hay tickers válidos con datos en el rango pedido.",
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
                    "No se encontró una columna Close o Adj Close en la descarga de yfinance.",
                    provider_output,
                )
            )
        prices = raw_data[[price_column]].copy()
        if len(requested_tickers) == 1:
            prices.columns = [requested_tickers[0]]

    prices = _coerce_price_frame(prices, ffill_limit=ffill_limit)

    if requested_tickers:
        available_tickers = [ticker for ticker in requested_tickers if ticker in prices.columns]
        if available_tickers:
            prices = prices[available_tickers]

    if prices.empty:
        raise ValueError(
            _with_provider_context(
                "La descarga no contiene precios utilizables después de la limpieza.",
                provider_output,
            )
        )

    return prices



def _build_metadata(available_tickers: Sequence[str]) -> list[dict[str, str]]:
    universe_by_ticker = {etf["ticker"]: dict(etf) for etf in get_universe()}
    metadata: list[dict[str, str]] = []

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



def _filter_tickers_by_history(
    prices: pd.DataFrame,
    xeon_ticker: str,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    eligible: list[str] = []
    dropped: list[str] = []

    for ticker in prices.columns:
        n_obs = int(prices[ticker].dropna().shape[0])
        if n_obs >= min_history_days or ticker == xeon_ticker:
            eligible.append(ticker)
        else:
            dropped.append(f"{ticker} ({n_obs} obs)")

    if xeon_ticker not in eligible:
        raise ValueError(
            f"{xeon_ticker} no tiene historia suficiente para el pipeline BL-Omega."
        )

    filtered = prices[eligible]
    if dropped:
        print("[data_loader] Tickers eliminados por historia insuficiente:", ", ".join(dropped))
    return filtered



def detect_backtest_start(
    prices_raw: pd.DataFrame,
    xeon_ticker: str,
    min_risk_assets: int = DEFAULT_MIN_RISK_ASSETS,
) -> pd.Timestamp:
    risk_cols = [ticker for ticker in prices_raw.columns if ticker != xeon_ticker]
    if not risk_cols:
        raise ValueError("No hay columnas de riesgo para detectar la fecha de inicio del backtest.")

    for date, row in prices_raw[risk_cols].iterrows():
        if int(row.notna().sum()) >= int(min_risk_assets):
            return pd.Timestamp(date)

    raise ValueError(
        f"Nunca coinciden {min_risk_assets} ETFs de riesgo con datos simultáneos."
    )



def build_trading_calendar(
    prices_raw: pd.DataFrame,
    backtest_start: str | pd.Timestamp,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
) -> pd.DataFrame:
    prices = prices_raw.loc[pd.Timestamp(backtest_start) :].copy()
    prices = prices.ffill(limit=max_ffill_days)
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("El calendario limpio quedó vacío tras aplicar forward-fill y dropna.")
    return prices



def build_lambda_dict(
    tickers: Sequence[str],
    *,
    default_lambda: float = DEFAULT_RISK_LAMBDA,
    defensive_lambda: float = DEFAULT_DEFENSIVE_LAMBDA,
    xeon_ticker: str | None = None,
) -> dict[str, float]:
    xeon_ticker = xeon_ticker or get_defensive_ticker()
    lambdas: dict[str, float] = {}
    for ticker in tickers:
        lambdas[str(ticker)] = float(defensive_lambda if ticker == xeon_ticker else default_lambda)
    return lambdas



def load_universe_data(
    *,
    start: str = "2014-01-01",
    end: str | None = None,
    tickers_riesgo: Sequence[str] | None = None,
    xeon_ticker: str | None = None,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    min_risk_assets: int = DEFAULT_MIN_RISK_ASSETS,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
) -> dict:
    """Carga el universo completo y devuelve datos listos para el motor multiactivo."""
    xeon_ticker = xeon_ticker or get_defensive_ticker()
    risk_tickers = list(tickers_riesgo or get_universe_tickers(include_defensive=False))
    requested_tickers = risk_tickers + [xeon_ticker]

    raw_data, provider_output = _download_from_yfinance(
        tickers=requested_tickers,
        start=start,
        end=end,
        auto_adjust=True,
    )
    prices_raw = _extract_price_frame(
        raw_data,
        requested_tickers,
        provider_output=provider_output,
        ffill_limit=None,
    )
    prices_raw = _filter_tickers_by_history(prices_raw, xeon_ticker, min_history_days=min_history_days)
    backtest_start = detect_backtest_start(
        prices_raw,
        xeon_ticker=xeon_ticker,
        min_risk_assets=min_risk_assets,
    )
    prices = build_trading_calendar(
        prices_raw,
        backtest_start=backtest_start,
        max_ffill_days=max_ffill_days,
    )
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise ValueError("No se pudieron calcular retornos para el universo limpio.")

    tickers = list(prices.columns)
    metadata = _build_metadata(tickers)
    transaction_costs = build_lambda_dict(tickers, xeon_ticker=xeon_ticker)

    return {
        "prices_raw": prices_raw,
        "prices": prices,
        "returns": returns,
        "metadata": metadata,
        "transaction_costs": transaction_costs,
        "todos_tickers": tickers,
        "tickers": tickers,
        "xeon_ticker": xeon_ticker,
        "backtest_start": returns.index[0],
    }



def download_market_data(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    tickers: list[str] | str | None = None,
    auto_adjust: bool = True,
) -> dict:
    """
    Descarga precios reales y calcula retornos simples.

    Devuelve un diccionario listo para los siguientes módulos:
    - tickers: lista final descargada
    - prices: DataFrame de precios
    - returns: DataFrame de retornos porcentuales
    - metadata: lista del universo disponible
    - transaction_costs: diccionario simple por ticker
    """
    selected_tickers = _normalize_tickers(tickers)
    if not selected_tickers:
        raise ValueError("No se recibió ningún ticker válido para descargar datos.")

    raw_data, provider_output = _download_from_yfinance(
        tickers=selected_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
    )

    prices = _extract_price_frame(
        raw_data,
        selected_tickers,
        provider_output=provider_output,
        ffill_limit=DEFAULT_MAX_FFILL_DAYS,
    )
    prices = prices.dropna(axis=1, how="all")
    available_tickers = [ticker for ticker in selected_tickers if ticker in prices.columns]

    if not available_tickers:
        raise ValueError(
            _with_provider_context(
                "No hay tickers válidos con precios descargados en el rango solicitado.",
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
    spread: float | None = None,
    extra_fee: float | None = None,
    window: int = 21,
) -> dict[str, float]:
    """
    Mantiene la firma v0 pero devuelve lambdas defensibles por ticker.

    `spread` y `extra_fee` se aceptan sólo por compatibilidad. Ya no se usa la
    heurística absurda basada en `0.5 / precio` que inflaba costes en activos baratos.
    """
    if prices_df.empty:
        return {}

    _ = spread, extra_fee, window  # compatibilidad explícita
    return build_lambda_dict(prices_df.columns)



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
        raise ValueError("No hay tickers para resolver el precio de ejecución.")

    if prices_df.empty:
        raise ValueError("No hay precios históricos para resolver el precio de ejecución.")

    clean_prices = prices_df.ffill(limit=DEFAULT_MAX_FFILL_DAYS).dropna(how="all")
    if clean_prices.empty:
        raise ValueError("Los precios históricos no contienen datos utilizables.")

    last_timestamp = pd.Timestamp(clean_prices.index[-1])
    last_date = last_timestamp.normalize()
    next_date = last_date + pd.Timedelta(days=1)
    window_end = next_date + pd.Timedelta(days=EXECUTION_LOOKAHEAD_DAYS)
    last_prices = clean_prices.iloc[-1].to_dict()
    fallback = _build_execution_fallback(clean_prices, last_prices)

    raw_data, provider_output = _download_from_yfinance(
        tickers=selected_tickers,
        start=next_date.strftime("%Y-%m-%d"),
        end=window_end.strftime("%Y-%m-%d"),
        auto_adjust=auto_adjust,
    )

    if raw_data.empty:
        return fallback

    try:
        future_prices = _extract_price_frame(
            raw_data,
            selected_tickers,
            provider_output=provider_output,
            ffill_limit=0,
        ).dropna(axis=1, how="all")
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
