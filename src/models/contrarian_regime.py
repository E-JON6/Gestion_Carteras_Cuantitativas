"""Multi-factor contrarian regime detector.

Combines volatility, equal-weight portfolio drawdown and average pairwise
correlation. Unlike defensive detectors, this never reduces risky exposure;
instead it returns multipliers that *amplify* view aggressiveness and
*widen* no-trade bands during stress (so the strategy holds through panics
and buys harder into them).
"""

from dataclasses import dataclass

import numpy as np

from src.domain.asset import PriceHistory


@dataclass(frozen=True)
class RegimeAdjustments:
    """Result of a regime detection."""

    name: str
    view_scale_mult: float
    band_mult: float


@dataclass
class ContrarianRegimeDetector:
    """Contrarian regime detector based on vol, drawdown and correlation."""

    vol_caution_threshold: float = 0.28
    vol_crisis_threshold: float = 0.40
    dd_caution_threshold: float = 0.16
    dd_crisis_threshold: float = 0.28
    corr_caution_threshold: float = 0.38
    corr_crisis_threshold: float = 0.52
    dd_lookback: int = 252
    corr_lookback: int = 63

    caution_view_boost: float = 1.15
    crisis_view_boost: float = 1.30
    caution_band_mult: float = 1.5
    crisis_band_mult: float = 2.0

    trading_days: int = 252

    def detect(self, history: PriceHistory) -> RegimeAdjustments:
        rets = history.log_returns()
        if len(rets) < 20:
            return RegimeAdjustments("normal", 1.0, 1.0)

        vol = float(rets.std().mean() * np.sqrt(self.trading_days))

        lb = min(self.dd_lookback, len(rets))
        ew = rets.tail(lb).mean(axis=1)
        wealth = (1.0 + ew).cumprod()
        ew_mdd = float(-(wealth / wealth.cummax() - 1.0).min()) if len(wealth) else 0.0

        avg_corr = 0.0
        cl = min(self.corr_lookback, len(rets))
        if cl >= 20 and rets.shape[1] > 1:
            cmat = rets.tail(cl).corr().values
            tri = np.triu_indices(cmat.shape[0], k=1)
            avg_corr = float(np.nanmean(cmat[tri]))

        if (vol > self.vol_crisis_threshold
                or ew_mdd > self.dd_crisis_threshold
                or avg_corr > self.corr_crisis_threshold):
            return RegimeAdjustments("crisis", self.crisis_view_boost, self.crisis_band_mult)

        if (vol > self.vol_caution_threshold
                or ew_mdd > self.dd_caution_threshold
                or avg_corr > self.corr_caution_threshold):
            return RegimeAdjustments("caution", self.caution_view_boost, self.caution_band_mult)

        return RegimeAdjustments("normal", 1.0, 1.0)
