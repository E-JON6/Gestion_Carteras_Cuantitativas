from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceReport:
    """Computes performance metrics from the portfolio value series."""

    TRADING_DAYS_PER_YEAR: int = field(default=252, init=False, repr=False)

    values: pd.Series
    risk_free_rate: float
    
    returns: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "returns", self.values.pct_change().dropna())

    def compute(self) -> dict:
        """Compute all metrics and return them as a dict."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
        }

    @property
    def total_return(self) -> float:
        return self.values.iloc[-1] / self.values.iloc[0] - 1

    @property
    def annualized_return(self) -> float:
        total = self.total_return
        n_days = len(self.values)
        years = n_days / self.TRADING_DAYS_PER_YEAR
        if years <= 0:
            return 0.0
        return (1 + total) ** (1 / years) - 1

    @property
    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    @property
    def sharpe_ratio(self) -> float:
        ann_ret = self.annualized_return
        ann_vol = self.annualized_volatility
        if ann_vol == 0:
            return 0.0
        return (ann_ret - self.risk_free_rate) / ann_vol

    @property
    def max_drawdown(self) -> float:
        cummax = self.values.cummax()
        drawdowns = (self.values - cummax) / cummax
        return drawdowns.min()

    @property
    def calmar_ratio(self) -> float:
        mdd = self.max_drawdown
        if mdd == 0:
            return 0.0
        return self.annualized_return / abs(mdd)