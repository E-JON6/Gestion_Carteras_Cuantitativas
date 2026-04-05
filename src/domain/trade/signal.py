from __future__ import annotations

import pandas as pd

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .order import PendingOrder, InvalidOrder


@dataclass(frozen=True)
class Signal:
    """A strategy's intention to hold a specific weight in an asset.

    This is a desired allocation, not an executed trade. It may be
    filtered or discarded before reaching the Broker.
    """

    date: pd.Timestamp
    ticker: str
    target_weight: float
    reason: str | None = None
    info: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "date": self.date,
            "ticker": self.ticker,
            "target_weight": self.target_weight,
            "reason": self.reason,
        }
        if self.info:
            d.update(self.info)
        return d

    def invalidate(self, reason: str) -> InvalidOrder:
        """Returns Order with status invalid."""
        from .order import InvalidOrder
        return InvalidOrder(
            signal=self,
            shares=0,
            price=0,
            reason=reason,
        )

    def order(self, shares: float, price: float) -> PendingOrder:
        """Returns Order with status pending."""
        from .order import PendingOrder
        return PendingOrder(
            signal=self,
            shares=shares,
            price=price,
        )
