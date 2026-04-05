from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Immutable capture of full portfolio state at a single point in time."""

    date: pd.Timestamp

    total_value: float
    cash: float
    total_positions_value: float
    alpha: float

    positions: dict[str, float]
    positions_values: dict[str, float]
    positions_weights: dict[str, float]


    @property
    def values_dict(self) -> dict:
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "total_positions_value": self.total_positions_value,
            **{f"{asset}_value": v for asset, v in self.positions_values.items()},
        }

    @property
    def weights_dict(self) -> dict[str, float]:
        return {
            **{f"{asset}_weight": w for asset, w in self.positions_weights.items()},
        }

    @property
    def positions_dict(self) -> dict[str, float]:
        return {
            **{f"{asset}_shares": s for asset, s in self.positions.items()},
        }

    @property
    def full_dict(self) -> dict:
        """Everything in one flat dict."""
        return {
            "date": self.date,
            **self.values_dict,
            **self.weights_dict,
            **self.positions_dict,
            "alpha": self.alpha,
        }
