"""
Universo base de ETFs para la v0.

Interfaz v0:
- get_universe() devuelve la lista completa con metadatos minimos.
- get_universe_tickers() devuelve los tickers que usara data_loader.py.
- XEON.DE se mantiene como activo defensivo/monetario del pipeline.
"""

from __future__ import annotations

DEFENSIVE_TICKER = "XEON.DE"

ETF_UNIVERSE = [
    {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "equity",
        "region": "US",
        "role": "risk",
    },
    {
        "ticker": "VGK",
        "name": "Vanguard FTSE Europe ETF",
        "asset_class": "equity",
        "region": "Europe",
        "role": "risk",
    },
    {
        "ticker": "EWJ",
        "name": "iShares MSCI Japan ETF",
        "asset_class": "equity",
        "region": "Japan",
        "role": "risk",
    },
    {
        "ticker": "EEM",
        "name": "iShares MSCI Emerging Markets ETF",
        "asset_class": "equity",
        "region": "Emerging",
        "role": "risk",
    },
    {
        "ticker": "VNQ",
        "name": "Vanguard Real Estate ETF",
        "asset_class": "real_estate",
        "region": "Global",
        "role": "risk",
    },
    {
        "ticker": "GLD",
        "name": "SPDR Gold Shares",
        "asset_class": "commodity",
        "region": "Global",
        "role": "risk",
    },
    {
        "ticker": "TLT",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "asset_class": "fixed_income",
        "region": "US",
        "role": "risk",
    },
    {
        "ticker": DEFENSIVE_TICKER,
        "name": "Xtrackers II EUR Overnight Rate Swap UCITS ETF",
        "asset_class": "cash_proxy",
        "region": "Europe",
        "role": "defensive",
    },
]


def get_universe() -> list[dict[str, str]]:
    """Devuelve la lista completa de ETFs con copias de sus metadatos minimos."""
    return [dict(etf) for etf in ETF_UNIVERSE]


def get_universe_tickers(include_defensive: bool = True) -> list[str]:
    """Devuelve los tickers del universo respetando el orden canónico del repo."""
    return [
        etf["ticker"]
        for etf in ETF_UNIVERSE
        if include_defensive or etf["role"] != "defensive"
    ]


def get_defensive_ticker() -> str:
    """Devuelve el ticker defensivo/monetario usado por el pipeline v0."""
    return DEFENSIVE_TICKER
