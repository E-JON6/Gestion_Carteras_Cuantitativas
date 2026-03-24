"""
Carga datos reales de mercado para la v0.

Interfaz v0:
- download_market_data() recibe tickers y rango temporal.
- Devuelve un diccionario con claves estables:
  'tickers', 'prices', 'returns' y 'metadata'.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

try:
    from Data.universe import get_universe, get_universe_tickers
except ModuleNotFoundError:
    from universe import get_universe, get_universe_tickers


def _extract_price_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extrae un DataFrame de precios de cierre desde la salida de yfinance."""
    if raw_data.empty:
        raise ValueError("No se han descargado datos de yfinance.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        first_level = raw_data.columns.get_level_values(0)

        if "Close" in first_level:
            prices = raw_data["Close"].copy()
        elif "Adj Close" in first_level:
            prices = raw_data["Adj Close"].copy()
        else:
            raise ValueError("No se encontro una columna Close o Adj Close en la descarga.")
    else:
        price_column = "Close" if "Close" in raw_data.columns else "Adj Close"
        prices = raw_data[[price_column]].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices.sort_index().ffill().dropna(how="all")


def download_market_data(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    tickers: list[str] | None = None,
    auto_adjust: bool = True,
) -> dict:
    """
    Descarga precios reales y calcula retornos simples.

    Devuelve un diccionario listo para los siguientes modulos:
    - tickers: lista final descargada
    - prices: DataFrame de precios
    - returns: DataFrame de retornos porcentuales
    - metadata: lista base del universo
    """
    selected_tickers = tickers or get_universe_tickers()

    raw_data = yf.download(
        tickers=selected_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=False,
    )

    prices = _extract_price_frame(raw_data)
    prices = prices.dropna(axis=1, how="all")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    available_tickers = list(prices.columns)
    metadata = [etf for etf in get_universe() if etf["ticker"] in available_tickers]
    transaction_costs = compute_transaction_costs_v0(prices)

    return {
        "tickers": available_tickers,
        "prices": prices,
        "returns": returns,
        "metadata": metadata,
        "transaction_costs": transaction_costs,
    }


def compute_transaction_costs_v0(prices_df, spread=0.5, extra_fee=0.0001, window=21):
    recent_prices = prices_df.tail(window)
    daily_costs = 0.5 * spread / recent_prices + extra_fee
    return daily_costs.mean().to_dict()


def get_execution_data_v0(tickers, prices_df, auto_adjust=True):
    last_date = pd.Timestamp(prices_df.index[-1]).normalize()
    next_date = last_date + pd.Timedelta(days=1)
    today = pd.Timestamp.today().normalize()
    last_prices = prices_df.ffill().iloc[-1].to_dict()

    if next_date <= today:
        raw_data = yf.download(
            tickers=tickers,
            start=next_date.strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=False,
        )

        if not raw_data.empty:
            future_prices = _extract_price_frame(raw_data).dropna(axis=1, how="all")
            if not future_prices.empty:
                execution_prices = last_prices.copy()
                execution_prices.update(future_prices.iloc[0].to_dict())
                return {
                    "date": future_prices.index[0],
                    "prices": execution_prices,
                    "used_next_day": True,
                }

    return {
        "date": prices_df.index[-1],
        "prices": last_prices,
        "used_next_day": False,
    }


if __name__ == "__main__":
    market_data = download_market_data()
    print("Tickers descargados:", market_data["tickers"])
    print("Precios shape:", market_data["prices"].shape)
    print("Retornos shape:", market_data["returns"].shape)