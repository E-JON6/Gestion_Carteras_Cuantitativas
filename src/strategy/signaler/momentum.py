"""Scales signal weights by momentum: reduce exposure when momentum is negative."""

from dataclasses import dataclass

from src.domain.asset import PriceHistory, PriceSnapshot
from src.domain.trade import Signal


@dataclass
class MomentumSignaler:
    """Scales incoming signals by a momentum factor.

    Computes 12-1 month momentum (return over last ``lookback`` days
    excluding the most recent ``skip`` days) per ticker. If momentum
    is negative, the weight is reduced by ``alpha_min``. If positive,
    weight is unchanged (or boosted by the momentum z-score if
    ``boost`` is True).

    Args:
        lookback: Days for momentum calculation (default 252 = 12 months).
        skip: Recent days to skip (default 21 = 1 month, avoids reversal).
        alpha_min: Minimum scaling factor for negative momentum (0-1).
        boost: If True, scale positive momentum proportionally.
    """

    lookback: int = 252
    skip: int = 21
    alpha_min: float = 0.3
    boost: bool = False

    def scale(
        self,
        signals: list[Signal],
        history: PriceHistory,
    ) -> list[Signal]:
        """Scale signal weights by momentum."""
        if not signals:
            return signals

        df = history.df
        if len(df) < self.lookback + self.skip:
            return signals

        mom = self._compute_momentum(df, [s.ticker for s in signals])

        scaled = []
        for s in signals:
            if s.ticker not in mom:
                scaled.append(s)
                continue

            m = mom[s.ticker]
            if m < 0:
                factor = self.alpha_min
            elif self.boost and m > 0:
                factor = 1.0 + m  # scale up by momentum magnitude
            else:
                factor = 1.0

            scaled.append(Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=s.target_weight * factor,
                reason=s.reason,
                info={**(s.info or {}), "momentum": m, "mom_factor": factor},
            ))

        return scaled

    def _compute_momentum(self, df, tickers: list[str]) -> dict[str, float]:
        """12-1 month momentum: return from t-lookback to t-skip."""
        available = [t for t in tickers if t in df.columns]
        if not available:
            return {}

        end = df[available].iloc[-self.skip] if self.skip > 0 else df[available].iloc[-1]
        start = df[available].iloc[-self.lookback]

        mom = {}
        for t in available:
            if start[t] > 0:
                mom[t] = (end[t] / start[t]) - 1.0
            else:
                mom[t] = 0.0

        return mom
