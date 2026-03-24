"""
Universo base de ETFs para la v0.

Interfaz v0:
- get_universe() devuelve la lista completa con metadatos minimos.
- get_universe_tickers() devuelve los tickers que usara data_loader.py.
"""

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
        "ticker": "XEON.DE",
        "name": "Xtrackers II EUR Overnight Rate Swap UCITS ETF",
        "asset_class": "cash_proxy",
        "region": "Europe",
        "role": "defensive",
    },
]


def get_universe():
    """Devuelve la lista completa de ETFs con sus metadatos minimos."""
    return list(ETF_UNIVERSE)


def get_universe_tickers(include_defensive=True):
    """Devuelve los tickers que se usaran para descargar datos reales."""
    if include_defensive:
        return [etf["ticker"] for etf in ETF_UNIVERSE]

    return [etf["ticker"] for etf in ETF_UNIVERSE if etf["role"] != "defensive"]