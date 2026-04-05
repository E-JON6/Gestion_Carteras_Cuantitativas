from __future__ import annotations

import pandas as pd

from dataclasses import dataclass

from .order import AcceptedOrder


@dataclass(frozen=True)
class Trade:
    """Represents a single executed trade operation."""

    order: AcceptedOrder
    cost: float

    @property
    def ticker(self) -> str:
        return self.order.ticker

    @property
    def date(self) -> pd.Timestamp:
        return self.order.date

    @property
    def shares(self) -> float:
        return self.order.shares

    @property
    def price(self) -> float:
        return self.order.price

    @property
    def direction(self) -> str:
        return self.order.direction

    @property
    def value_before_costs(self) -> float:
        return self.order.value_before_costs

    @property
    def abs_value_before_costs(self) -> float:
        return self.order.abs_value_before_costs
    
    @property
    def value_after_costs(self) -> float:
        return self.order.value_before_costs + self.cost

    @property
    def abs_value_after_costs(self) -> float:
        return self.order.abs_value_before_costs + self.cost

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "ticker": self.ticker,
            "direction": self.direction,
            "shares": self.shares,
            "price": self.price,
            "value": self.abs_value_before_costs,
            "cost": self.cost,
        }

    @property
    def cash_delta(self) -> float:
        """Net cash change. Negative when buying, positive when selling."""
        return - self.abs_value_before_costs - self.cost
