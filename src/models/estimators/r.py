from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain.asset import PriceHistory


class RiskFreeRateEstimator(ABC):
    """Estimator for the risk-free rate.

    All estimators accept a PriceHistory and return an
    annualised risk-free rate as a float.
    """

    @abstractmethod
    def estimate(self, history: PriceHistory) -> float: ...


@dataclass
class DefensiveRfrEstimator(RiskFreeRateEstimator):
    """Estimates annualised risk-free rate from a defensive ticker's price history.

    Uses geometric annualisation: (last / first) ^ (252 / n_days) - 1.
    """

    defensive_ticker: str
    window: int = 252

    def estimate(self, history: PriceHistory) -> float:
        if self.defensive_ticker not in history.tickers:
            raise ValueError(f"defensive ticker {self.defensive_ticker} not in history")

        series = history.df[self.defensive_ticker].dropna()
        if len(series) < 2:
            raise ValueError(
                f"not enough data to estimate rfr from {self.defensive_ticker}"
            )

        series = series.iloc[-self.window:]
        total_return = series.iloc[-1] / series.iloc[0]
        n_days = len(series)
        return float(total_return ** (252 / n_days) - 1)


@dataclass
class FixedRiskFreeRate(RiskFreeRateEstimator):
    """Returns a fixed risk-free rate (useful for testing or known rates)."""

    rate: float = 0.0

    def estimate(self, history: PriceHistory) -> float:
        return self.rate
