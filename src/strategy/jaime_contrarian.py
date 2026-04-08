"""Jaime contrarian strategy ported to the framework architecture.

Pipeline:
  1. CompositeSignal             → cross-sectional factor scores
  2. ContrarianRegimeDetector    → view_scale and band multipliers
  3. BlendedEwmaCovariance       → robust covariance
  4. BL posterior with composite → mu (prior = rf + delta · Sigma · w_inv_vol)
  5. Merton                      → optimal risky weights
  6. clip negatives + all-zero fallback
  7. TopNByWeightFilter          → keep top-N by raw weight
  8. MaxWeightFilter             → cap per ticker
  9. SectorCapFilter             → cap per sector (Jaime category mapping)
 10. normalise_weights           → no shorts + total cap
 11. min-weight prune + renorm
 12. liquidate excluded          → target=0 for currently-held tickers dropped
 13. fill_defensive              → defensive ticker absorbs the complement
 14. ProportionalBandFilter      → DN check (regime-adjusted base)
 15. CostAwareFilter             → ensure executable
 16. LeverageGuard                → safety net between rebalances
"""

from dataclasses import dataclass, field

import numpy as np

from src.domain.asset import PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.black_litterman import BlackLittermanModel
from src.models.composite_signal import CompositeSignal
from src.models.contrarian_regime import ContrarianRegimeDetector
from src.models.estimators.covariance import BlendedEwmaCovariance, CovEstimator
from src.models.estimators.r import FileRfrEstimator, RiskFreeRateEstimator
from src.models.leverage_guard import LeverageGuard
from src.models.merton_davisnorman import MertonDavisNormanModel
from src.strategy.filter.cost_aware import CostAwareFilter
from src.strategy.filter.max_weight import MaxWeightFilter
from src.strategy.filter.proportional_band import ProportionalBandFilter
from src.strategy.filter.sector_cap import SectorCapFilter
from src.strategy.filter.top_n_by_weight import TopNByWeightFilter
from src.strategy.transformers import fill_defensive, normalise_weights

from .base import Strategy


# ── Jaime defaults baked in (config.py: ETF_UNIVERSE) ──────────────────
# Used to map tickers → categories for the composite signal and the
# sector cap. Tickers not present here fall back to the universe sector.

_JAIME_TICKER_CATEGORIES = {
    "IWDA.L": "global_equity", "VGK": "europe_equity", "EWJ": "asia_developed",
    "IUSN.DE": "global_equity",
    "IEMA.L": "emerging_equity", "MCHI": "asia_emerging", "EWZ": "latam",
    "INDA": "asia_emerging", "EWT": "asia_emerging", "EWY": "asia_emerging",
    "XLY": "us_discretionary", "XLC": "us_communications",
    "XLF": "us_financials", "KBE": "us_financials",
    "XLE": "us_energy", "XOP": "us_energy",
    "XLI": "us_industrials", "IYT": "us_industrials",
    "XLB": "us_materials", "ITB": "us_housing",
    "XLV": "us_healthcare", "XBI": "us_healthcare",
    "XLP": "us_staples", "XLU": "us_utilities",
    "XLK": "tech", "SOXX": "tech", "IGV": "tech", "CIBR": "tech", "BOTZ": "tech",
    "VNQ": "real_estate", "LIT": "thematic",
    "GLD": "commodities", "SLV": "commodities", "DBA": "commodities", "COPX": "commodities",
    "TLT": "fixed_income", "IEF": "fixed_income", "TIP": "fixed_income",
    "HYG": "fixed_income", "LQD": "fixed_income", "EMB": "fixed_income",
    "BITO": "crypto",
}

# config.py: CATEGORY_SIGNAL_WEIGHTS
_JAIME_CATEGORY_WEIGHTS = {
    "global_equity":     {"momentum": 0.42, "reversal": 0.15, "trend": 0.33, "vol_penalty": 0.10},
    "europe_equity":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "asia_developed":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "emerging_equity":   {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "asia_emerging":     {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "latam":             {"momentum": 0.40, "reversal": 0.25, "trend": 0.25, "vol_penalty": 0.10},
    "us_discretionary":  {"momentum": 0.44, "reversal": 0.15, "trend": 0.31, "vol_penalty": 0.10},
    "us_communications": {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "us_financials":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_energy":         {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_industrials":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_materials":      {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_housing":        {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_healthcare":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_staples":        {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "us_utilities":      {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "tech":              {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "real_estate":       {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "thematic":          {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "commodities":       {"momentum": 0.35, "reversal": 0.25, "trend": 0.30, "vol_penalty": 0.10},
    "fixed_income":      {"momentum": 0.30, "reversal": 0.25, "trend": 0.35, "vol_penalty": 0.10},
    "crypto":            {"momentum": 0.54, "reversal": 0.05, "trend": 0.36, "vol_penalty": 0.05},
}


@dataclass(kw_only=True)
class JaimeContrarianStrategy(Strategy):
    """Composite-BL-Merton with proportional DN bands and contrarian regime."""

    defensive_ticker: str

    # Composite signal
    composite: CompositeSignal | None = None

    # Covariance
    cov_estimator: CovEstimator = field(default_factory=BlendedEwmaCovariance)

    # Black-Litterman
    bl_risk_aversion: float = 2.5  # config.BL_DELTA
    bl_tau: float = 0.05           # config.BL_TAU
    view_scale: float = 0.28       # config.VIEW_SCALE
    bl_prior_mode: str = "inv_vol"  # config.BL_PRIOR_WEIGHTS_MODE

    # Merton
    gamma: float = -0.8            # config.MERTON_GAMMA → gamma_eff = 1.8
    top_n: int = 20                # config.MERTON_N_TOP

    # Constraints
    max_individual_weight: float = 0.40  # config.MERTON_MAX_WEIGHT
    max_sector_weight: float = 0.35      # config.MERTON_MAX_SECTOR
    min_weight: float = 0.01             # config.MERTON_MIN_WEIGHT
    max_risky_fraction: float = 1.0      # always 100% invested

    # Regime
    regime_detector: ContrarianRegimeDetector = field(default_factory=ContrarianRegimeDetector)

    # DN bands (base, before regime multiplier)
    base_band: float = 0.05  # config.DN_BAND
    min_band: float = 0.02   # config.DN_MIN_BAND

    # Rebalance
    rebalance_every: int = 21  # ~monthly in trading days

    # Risk-free rate
    rfr_estimator: RiskFreeRateEstimator = field(default_factory=FileRfrEstimator)

    # Internal state
    _step_count: int = field(default=0, init=False, repr=False)
    _bl: BlackLittermanModel = field(init=False, repr=False)
    _top_n_filter: TopNByWeightFilter = field(init=False, repr=False)
    _max_weight_filter: MaxWeightFilter = field(init=False, repr=False)
    _sector_filter: SectorCapFilter = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)
    _last_regime: str = field(default="normal", init=False, repr=False)
    _last_scores: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "jaime_contrarian"
        if self.composite is None:
            self.composite = CompositeSignal(
                ticker_categories=dict(_JAIME_TICKER_CATEGORIES),
                category_weights={k: dict(v) for k, v in _JAIME_CATEGORY_WEIGHTS.items()},
            )

    @property
    def _risky_tickers(self) -> list[str]:
        return [t for t in self.universe.tickers if t != self.defensive_ticker]

    def _warmup(self, portfolio: Portfolio) -> None:
        self._bl = BlackLittermanModel(
            risk_aversion=self.bl_risk_aversion,
            tau=self.bl_tau,
        )
        self._top_n_filter = TopNByWeightFilter(n=self.top_n)
        self._max_weight_filter = MaxWeightFilter(max_weight=self.max_individual_weight)

        # Sector mapping: Jaime's hardcoded categories first, then universe.sector for the rest.
        sectors = dict(_JAIME_TICKER_CATEGORIES)
        for asset in self.universe:
            sectors.setdefault(asset.ticker, asset.sector)
        self._sector_filter = SectorCapFilter(
            sectors=sectors,
            max_sector_weight=self.max_sector_weight,
        )

        self._leverage_guard = LeverageGuard(max_leverage=portfolio.max_leverage)

    def reset(self) -> None:
        super().reset()
        self._step_count = 0
        self._last_regime = "normal"
        self._last_scores = {}

    def _generate_signals(
        self,
        prices: PriceSnapshot,
        portfolio: Portfolio,
    ) -> list[Signal]:
        self._step_count += 1
        is_rebalance_day = self._step_count % self.rebalance_every == 0
        leverage_exceeded = self._leverage_guard.is_triggered(portfolio, prices)
        if not is_rebalance_day and not leverage_exceeded:
            return []

        risky_history = self._history.select(self._risky_tickers)
        if len(risky_history) < 60:
            return []

        try:
            mu, cov, r, regime = self._estimate(risky_history)
        except Exception:
            return []
        self._last_regime = regime.name

        # Merton optimal weights from posterior mu
        try:
            merton = MertonDavisNormanModel(gamma=self.gamma, r=r)
            raw_weights = merton.optimal_weights(mu, cov)
        except Exception:
            return []

        # Build initial signals (clip negatives → all-zero fallback if needed)
        signals = self._build_initial_signals(prices, raw_weights, mu, r, regime)

        # Constraint chain (mirrors apply_constraints in Jaime/Models/merton.py)
        signals = self._top_n_filter.filter(signals)
        signals = self._max_weight_filter.filter(signals)
        signals = self._sector_filter.filter(signals)

        # Total cap (no shorts + max_risky_fraction)
        signals = self._normalise(signals, portfolio)

        # Drop tiny positions, then renormalise (Jaime does this last)
        signals = [s for s in signals if s.target_weight >= self.min_weight]
        signals = self._normalise(signals, portfolio)

        # Liquidate currently-held risky tickers that dropped out
        signals = self._add_liquidations(signals, portfolio, prices)

        # Defensive absorbs the complement (so the band check sees it too)
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # Davis-Norman with regime-adjusted base band
        band_filter = ProportionalBandFilter(
            base_band=self.base_band * regime.band_mult,
            min_band=self.min_band,
        )
        breached = band_filter.filter(signals, portfolio, prices)
        if not breached and not leverage_exceeded:
            return []
        if breached:
            signals = breached

        # Cost-aware safety net
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals

    # ── Helpers ─────────────────────────────────────────────────────────

    def _estimate(self, risky_history: PriceHistory):
        """Composite + cov + rfr + BL posterior. Returns (mu, cov, r, regime)."""
        scores = self.composite.compute(risky_history)
        self._last_scores = scores

        regime = self.regime_detector.detect(risky_history)
        view_scale_eff = self.view_scale * regime.view_scale_mult

        cov = self.cov_estimator.estimate(risky_history)
        r = self.rfr_estimator.estimate(self._history)

        tickers = list(cov.columns)
        n = len(tickers)
        market_w = self._market_weights(risky_history, tickers)

        # Prior with rf, posterior with composite-driven views
        pi_dict = self._bl.implied_returns(cov, market_w, rf=r)
        pi_arr = np.array([pi_dict[t] for t in tickers])
        scores_arr = np.array([scores.get(t, 0.0) for t in tickers])

        P = np.eye(n)
        Q = pi_arr + view_scale_eff * scores_arr

        # Idzorek-style continuous confidence
        abs_scores = np.abs(scores_arr)
        max_abs = abs_scores.max() if abs_scores.max() > 1e-8 else 1.0
        confidence = 0.3 + 0.6 * np.minimum(abs_scores / max_abs, 1.0)
        base_var = np.diag(P @ (self.bl_tau * cov.values) @ P.T)
        omega = np.diag(base_var / np.clip(confidence, 0.1, None))

        mu = self._bl.posterior_returns(cov, market_w, P, Q, omega, rf=r)
        return mu, cov, r, regime

    def _market_weights(self, history: PriceHistory, tickers: list[str]) -> dict[str, float]:
        """Equilibrium prior weights: equal-weight or inverse-volatility."""
        n = len(tickers)
        if self.bl_prior_mode == "inv_vol":
            vol = history.df[tickers].pct_change().std().values * np.sqrt(252)
            vol = np.maximum(vol, 1e-8)
            w = 1.0 / vol
            w = w / w.sum()
            return dict(zip(tickers, w))
        return {t: 1.0 / n for t in tickers}

    def _build_initial_signals(self, prices, raw_weights, mu, r, regime):
        """Clip negatives. If everything is zero, fall back to 1/n on first n."""
        clipped = {t: max(float(w), 0.0) for t, w in raw_weights.items()}

        if sum(clipped.values()) < 1e-10:
            tickers = list(raw_weights.keys())
            n_top = min(self.top_n, len(tickers))
            clipped = {t: 0.0 for t in tickers}
            for t in tickers[:n_top]:
                clipped[t] = 1.0 / n_top

        return [
            Signal(
                date=prices.date,
                ticker=t,
                target_weight=float(w),
                reason="jaime_contrarian",
                info={
                    "mu": mu.get(t, 0.0),
                    "r": r,
                    "score": self._last_scores.get(t, 0.0),
                    "regime": regime.name,
                    "raw_weight": float(raw_weights.get(t, 0.0)),
                },
            )
            for t, w in clipped.items()
        ]

    def _normalise(self, signals: list[Signal], portfolio: Portfolio) -> list[Signal]:
        if not signals:
            return signals
        weights = normalise_weights(
            {s.ticker: s.target_weight for s in signals},
            portfolio,
            self.max_risky_fraction,
        )
        return [
            Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=weights.get(s.ticker, 0.0),
                reason=s.reason,
                info=s.info,
            )
            for s in signals if s.ticker in weights
        ]

    def _add_liquidations(self, signals, portfolio, prices) -> list[Signal]:
        """Append target=0 signals for currently-held risky tickers not in signals."""
        selected = {s.ticker for s in signals}
        current = portfolio.positions_weights(prices)
        for t in self._risky_tickers:
            if t in selected or t == self.defensive_ticker:
                continue
            if abs(current.get(t, 0.0)) > 1e-9:
                signals.append(Signal(
                    date=prices.date,
                    ticker=t,
                    target_weight=0.0,
                    reason="liquidate_excluded",
                ))
        return signals
