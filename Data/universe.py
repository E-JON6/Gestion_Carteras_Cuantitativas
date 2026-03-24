"""
Universo base de ETFs para la v0.

Interfaz v0:
- get_universe() devuelve la lista completa con metadatos minimos.
- get_universe_tickers() devuelve los tickers que usara data_loader.py.
- XEON.DE se mantiene como activo defensivo/monetario del pipeline.

Fuente de verdad:
- config.py define el universo de riesgo y el ticker defensivo.
- Este modulo solo adapta esa configuracion al contrato actual del runtime.
"""

from __future__ import annotations

from config import ETF_UNIVERSE as CONFIG_ETF_UNIVERSE
from config import XEON_TICKER as CONFIG_XEON_TICKER

DEFENSIVE_TICKER = CONFIG_XEON_TICKER


def _build_runtime_universe() -> list[dict[str, str]]:
    risk_universe = [
        {
            "ticker": ticker,
            "name": ticker,
            "asset_class": "risk_etf",
            "region": "unknown",
            "role": "risk",
            "category": category,
        }
        for ticker, category in CONFIG_ETF_UNIVERSE.items()
    ]

    defensive_entry = {
        "ticker": CONFIG_XEON_TICKER,
        "name": CONFIG_XEON_TICKER,
        "asset_class": "cash_proxy",
        "region": "unknown",
        "role": "defensive",
        "category": "cash_proxy",
    }

    return risk_universe + [defensive_entry]


ETF_UNIVERSE = _build_runtime_universe()


def _validate_runtime_universe() -> None:
    if not CONFIG_ETF_UNIVERSE:
        raise ValueError("config.ETF_UNIVERSE no puede estar vacio.")

    if CONFIG_XEON_TICKER in CONFIG_ETF_UNIVERSE:
        raise ValueError("config.XEON_TICKER no puede formar parte de config.ETF_UNIVERSE.")

    expected_with_defensive = list(CONFIG_ETF_UNIVERSE.keys()) + [CONFIG_XEON_TICKER]
    runtime_with_defensive = [etf["ticker"] for etf in _build_runtime_universe()]

    if runtime_with_defensive != expected_with_defensive:
        raise ValueError(
            "El universo runtime no coincide con config.py; revisa ETF_UNIVERSE y XEON_TICKER."
        )


def get_universe() -> list[dict[str, str]]:
    """Devuelve la lista completa de ETFs con copias de sus metadatos minimos."""
    _validate_runtime_universe()
    return [dict(etf) for etf in _build_runtime_universe()]


def get_universe_tickers(include_defensive: bool = True) -> list[str]:
    """Devuelve los tickers del universo respetando el orden canonico de config.py."""
    _validate_runtime_universe()
    tickers = [etf["ticker"] for etf in _build_runtime_universe()]
    if include_defensive:
        return tickers
    return [ticker for ticker in tickers if ticker != CONFIG_XEON_TICKER]


def get_defensive_ticker() -> str:
    """Devuelve el ticker defensivo/monetario usado por el pipeline v0."""
    _validate_runtime_universe()
    return CONFIG_XEON_TICKER
