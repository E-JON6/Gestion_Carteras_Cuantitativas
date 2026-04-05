"""Dual momentum: absolute (market on/off) + relative (filter losers)."""

from dataclasses import dataclass

from src.domain.asset import PriceHistory
from src.domain.trade import Signal


@dataclass
class DualMomentumSignaler:
    """Applies dual momentum filter to incoming signals.

    1. Absolute momentum: if the average momentum of all risky tickers
       is negative, returns empty (everything goes to defensive).
    2. Relative momentum: drops tickers whose momentum is below the
       median of the group.

    Args:
        lookback: Days for momentum calculation (default 252 = 12 months).
        skip: Recent days to skip (default 21 = 1 month).
    """

    lookback: int = 252
    skip: int = 21

    def filter(
        self,
        signals: list[Signal],
        history: PriceHistory,
    ) -> list[Signal]:
        if not signals:
            return signals

        df = history.df
        if len(df) < self.lookback + self.skip:
            return signals

        tickers = [s.ticker for s in signals]
        mom = self._compute_momentum(df, tickers)
        if not mom:
            return signals

        # Absolute momentum: market average
        avg_mom = sum(mom.values()) / len(mom)
        market_on = avg_mom > 0

        if not market_on:
            # Everything off → empty signals → defensive gets 100%
            return []

        # Relative momentum: keep only above-median tickers
        sorted_mom = sorted(mom.values())
        median_mom = sorted_mom[len(sorted_mom) // 2]

        filtered = []
        for s in signals:
            m = mom.get(s.ticker, 0.0)
            above_median = m >= median_mom

            filtered.append(Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=s.target_weight if above_median else 0.0,
                reason=s.reason,
                info={
                    **(s.info or {}),
                    "momentum": m,
                    "market_mom": avg_mom,
                    "market_on": market_on,
                    "above_median": above_median,
                },
            ))

        # Remove zero-weight signals
        return [s for s in filtered if s.target_weight > 1e-9]

    def _compute_momentum(self, df, tickers: list[str]) -> dict[str, float]:
        available = [t for t in tickers if t in df.columns]
        if not available:
            return {}

        end = df[available].iloc[-self.skip] if self.skip > 0 else df[available].iloc[-1]
        start = df[available].iloc[-self.lookback]

        return {
            t: (end[t] / start[t]) - 1.0
            for t in available
            if start[t] > 0
        }
