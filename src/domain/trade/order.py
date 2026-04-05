from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from typing import ClassVar

import pandas as pd

from .signal import Signal


@dataclass(frozen=True)
class Order(ABC):
    """A concrete instruction to buy or sell shares.

    Created by the Broker from a Signal. Can be accepted or rejected.

    Lifecycle: Signal -> Order (pending) -> accepted -> Trade
                                         -> rejected (no Trade)
    """

    status: ClassVar[str]

    signal: Signal
    shares: float
    price: float

    @property
    def ticker(self) -> str:
        return self.signal.ticker
    
    @property
    def date(self) -> pd.Timestamp:
        return self.signal.date

    @property
    def direction(self) -> str:
        return "buy" if self.shares > 0 else "sell"

    @property
    def abs_value_before_costs(self) -> float:
        return abs(self.shares) * self.price
    
    @property
    def value_before_costs(self) -> float:
        return self.shares * self.price

    

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "ticker": self.ticker,
            "direction": self.direction,
            "shares": self.shares,
            "price": self.price,
            "value": self.abs_value_before_costs,
            "status": self.status,
            "signal_reason": self.signal.reason,
            "reason": getattr(self, "reason", None),
        }


@dataclass(frozen=True)
class PendingOrder(Order):
    status: ClassVar[str] = "pending"

    def accept(self) -> Order:
        """Return a new Order with status accepted."""
        return AcceptedOrder(
            signal=self.signal,
            shares=self.shares,
            price=self.price,
        )
    
    def reject(self, reason: str) -> Order:
        """Return a new Order with status rejected."""
        return RejectedOrder(
            signal=self.signal,
            shares=self.shares,
            price=self.price,
            reason=reason
        )
    

@dataclass(frozen=True)
class AcceptedOrder(Order):
    status: ClassVar[str] = "accepted"

@dataclass(frozen=True)
class RejectedOrder(Order):
    status: ClassVar[str] = "rejected"
    reason: str

@dataclass(frozen=True)
class InvalidOrder(Order):
    status: ClassVar[str] = "invalid"
    reason: str