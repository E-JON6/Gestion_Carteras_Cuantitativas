from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """Represents a tradeable financial asset."""

    ticker: str
    name: str = ""
    asset_type: str = "equity"
    sector: str = ""

    transaction_cost: float | None = None
