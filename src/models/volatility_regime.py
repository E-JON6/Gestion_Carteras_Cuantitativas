from dataclasses import dataclass

import numpy as np

from src.domain.asset import PriceHistory


@dataclass
class VolatilityRegimeDetector:
    """Detects market volatility regime from price history.

    Classifies the market as normal, caution, or crisis based on
    the average annualized volatility of the given tickers.
    """

    caution_threshold: float = 0.20
    crisis_threshold: float = 0.30
    caution_cap: float = 0.70
    crisis_cap: float = 0.40

    def detect(self, history: PriceHistory, tickers: list[str] | None = None) -> tuple[str, float]:
        """Detect regime from price history.

        Returns:
            Tuple (regime_name, max_risk_cap).
        """
        if len(history) < 20:
            return "normal", 1.0

        cols = tickers or history.tickers
        sub = history.select(cols) if tickers else history

        ret = sub.log_returns()
        if len(ret) < 2:
            return "normal", 1.0

        market_vol = ret.std().mean() * np.sqrt(252)

        if market_vol > self.crisis_threshold:
            return "crisis", self.crisis_cap
        elif market_vol > self.caution_threshold:
            return "caution", self.caution_cap
        return "normal", 1.0
