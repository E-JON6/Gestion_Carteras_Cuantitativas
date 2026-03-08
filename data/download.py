"""
Módulo de descarga de datos.
Yahoo Finance para precios de ETFs, FRED para tasas de interés.
"""

import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from gestion_cuantitativa.config import (
    ETF_CONFIGS, DEFAULT_ETF_TICKER,
    RISK_FREE_FRED, VIX_TICKER,
    DATA_START, SIMULATION_END
)


def download_risky_asset(ticker=None, start=None, end=None):
    """
    Descarga datos OHLCV diarios del ETF.

    Returns:
        pd.DataFrame con columnas Open, High, Low, Close, Volume
    """
    ticker = ticker or DEFAULT_ETF_TICKER
    start = start or DATA_START
    end = end or SIMULATION_END

    data = yf.download(ticker, start=start, end=end, progress=False)

    # yfinance puede devolver MultiIndex si hay un solo ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel('Ticker')

    # Asegurar que el índice es DatetimeIndex sin timezone
    data.index = pd.to_datetime(data.index).tz_localize(None)

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def download_risky_asset_with_fallback(ticker, start=None, end=None):
    """
    Descarga datos del ETF, con fallback automático si el ticker principal falla
    o tiene datos insuficientes.

    Args:
        ticker: str, ticker principal
        start: fecha inicio
        end: fecha fin

    Returns:
        tuple (pd.DataFrame, str) con datos y ticker real utilizado
    """
    start = start or DATA_START
    end = end or SIMULATION_END

    try:
        data = download_risky_asset(ticker=ticker, start=start, end=end)
        if len(data) < 252:
            raise ValueError(f"Datos insuficientes para {ticker}: {len(data)} filas")
        return data, ticker
    except Exception as e:
        config = ETF_CONFIGS.get(ticker, {})
        fallbacks = config.get('fallback_tickers', [])
        for fb_ticker in fallbacks:
            try:
                print(f"  ⚠ {ticker} falló ({e}), probando fallback: {fb_ticker}")
                data = download_risky_asset(ticker=fb_ticker, start=start, end=end)
                if len(data) >= 252:
                    return data, fb_ticker
            except Exception:
                continue
        raise RuntimeError(
            f"No se pudo descargar datos para {ticker} ni sus fallbacks {fallbacks}"
        )


def download_risk_free_rate(series_id=None, start=None, end=None):
    """
    Descarga la tasa libre de riesgo desde FRED.
    ECBDFR = ECB Deposit Facility Rate (en %).

    Returns:
        pd.Series con la tasa anualizada (en decimal, no porcentaje)
    """
    series_id = series_id or RISK_FREE_FRED
    start = start or DATA_START
    end = end or SIMULATION_END

    data = pdr.get_data_fred(series_id, start=start, end=end)
    rate = data.iloc[:, 0] / 100.0  # Convertir de % a decimal
    rate.name = 'risk_free_rate'

    return rate


def download_vix(start=None, end=None):
    """
    Descarga el VIX para E3 (Volatility Targeting).

    Returns:
        pd.Series con el cierre diario del VIX
    """
    start = start or DATA_START
    end = end or SIMULATION_END

    data = yf.download(VIX_TICKER, start=start, end=end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel('Ticker')

    data.index = pd.to_datetime(data.index).tz_localize(None)

    return data['Close'].dropna().rename('VIX')


def download_all_data(ticker=None, start=None, end=None):
    """
    Descarga todos los datos necesarios para un ETF dado.

    Args:
        ticker: str, ticker del ETF (default: DEFAULT_ETF_TICKER)
        start: fecha inicio
        end: fecha fin

    Returns:
        dict con keys: 'risky', 'risk_free', 'vix', 'actual_ticker'
    """
    ticker = ticker or DEFAULT_ETF_TICKER
    start = start or DATA_START
    end = end or SIMULATION_END

    print(f"Descargando {ticker}...")
    risky, actual_ticker = download_risky_asset_with_fallback(
        ticker, start=start, end=end
    )
    print(f"  → {len(risky)} filas desde {risky.index[0].date()} hasta {risky.index[-1].date()}")
    if actual_ticker != ticker:
        print(f"  → Usando ticker alternativo: {actual_ticker}")

    print(f"Descargando {RISK_FREE_FRED} desde FRED...")
    risk_free = download_risk_free_rate(start=start, end=end)
    print(f"  → {len(risk_free)} filas, último valor: {risk_free.iloc[-1]:.4f}")

    print(f"Descargando {VIX_TICKER}...")
    vix = download_vix(start=start, end=end)
    print(f"  → {len(vix)} filas")

    return {
        'risky': risky,
        'risk_free': risk_free,
        'vix': vix,
        'actual_ticker': actual_ticker,
    }


if __name__ == "__main__":
    data = download_all_data()
    print("\n=== Resumen ===")
    print(f"Activo de riesgo: {len(data['risky'])} obs ({data['actual_ticker']})")
    print(f"Tasa libre riesgo: {len(data['risk_free'])} obs")
    print(f"VIX: {len(data['vix'])} obs")
