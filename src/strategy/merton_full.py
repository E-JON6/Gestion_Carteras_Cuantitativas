from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.mu import MuEstimator, HistoricalMean
from src.models.estimators.covariance import CovEstimator, SampleCovariance
from src.models.leverage_guard import LeverageGuard
from src.models.volatility_regime import VolatilityRegimeDetector
from src.strategy.signaler import MertonSignaler
from src.strategy.filter.top_n import TopNFilter
from src.strategy.filter.frozen import FrozenAssetManager
from src.strategy.filter.no_trade_band import NoTradeBandFilter
from src.strategy.filter.cost_aware import CostAwareFilter
from src.strategy.transformers import normalise_weights, fill_defensive

from .base import Strategy


@dataclass(kw_only=True)
class MertonFullStrategy(Strategy):
    """Full Merton pipeline: top-N + frozen + vol regime + DN bands.

    Pipeline:
      1. MertonSignaler       → optimal risky weights
      2. TopNFilter           → keep top N by Omega (+ frozen)
      3. FrozenAssetManager   → freeze/thaw exited tickers
      4. Liquidate             → signal weight=0 for thawed tickers
      5. VolatilityRegime     → scale max_risky_fraction by regime
      6. Normalise            → apply scaled fraction
      7. NoTradeBandFilter    → skip tickers inside DN band
      8. fill_defensive       → XEON.DE absorbs remainder
      9. CostAwareFilter      → ensure executable
     10. LeverageGuard        → force rebalance if leverage drifts
    """

    defensive_ticker: str
    gamma: float = 0.5
    mu_estimator: MuEstimator = field(default_factory=HistoricalMean)
    cov_estimator: CovEstimator = field(default_factory=SampleCovariance)
    rebalance_every: int = 21
    max_risky_fraction: float = 0.8
    # Top-N
    top_n: int = 5
    # Frozen
    thaw_threshold: float = 0.02
    # Vol regime
    vol_caution_threshold: float = 0.20
    vol_crisis_threshold: float = 0.30
    vol_caution_cap: float = 0.70
    vol_crisis_cap: float = 0.40
    # DN bands
    use_dn_bands: bool = True
    band_scale: float = 1.0

    _step_count: int = field(default=0, init=False, repr=False)
    _signaler: MertonSignaler = field(init=False, repr=False)
    _top_n_filter: TopNFilter = field(init=False, repr=False)
    _frozen_mgr: FrozenAssetManager = field(init=False, repr=False)
    _vol_detector: VolatilityRegimeDetector = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "merton_full"

    def _warmup(self, portfolio: Portfolio) -> None:
        self._signaler = MertonSignaler(
            risky_tickers=self._risky_tickers,
            defensive_ticker=self.defensive_ticker,
            gamma=self.gamma,
            mu_estimator=self.mu_estimator,
            cov_estimator=self.cov_estimator,
            max_risky_fraction=1.0,  # normalise later after regime
        )
        self._top_n_filter = TopNFilter(n=self.top_n)
        self._frozen_mgr = FrozenAssetManager(thaw_threshold=self.thaw_threshold)
        self._vol_detector = VolatilityRegimeDetector(
            caution_threshold=self.vol_caution_threshold,
            crisis_threshold=self.vol_crisis_threshold,
            caution_cap=self.vol_caution_cap,
            crisis_cap=self.vol_crisis_cap,
        )
        self._leverage_guard = LeverageGuard(max_leverage=portfolio.max_leverage)

    def reset(self) -> None:
        super().reset()
        self._step_count = 0
        if hasattr(self, "_frozen_mgr"):
            self._frozen_mgr.reset()

    @property
    def _risky_tickers(self) -> list[str]:
        return [t for t in self.universe.tickers if t != self.defensive_ticker]

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

        # 1. Merton optimal weights
        signals = self._signaler.generate(self._history, prices, portfolio)
        if not signals:
            return []

        # 2. Top-N filter (pass frozen tickers through)
        risky_history = self._history.select(self._risky_tickers)
        signals = self._top_n_filter.filter(
            signals, risky_history, keep=self._frozen_mgr.frozen,
        )

        # 3. Freeze/thaw
        top_tickers = {s.ticker for s in signals}
        to_liquidate = self._frozen_mgr.update(top_tickers, portfolio, prices)

        # 4. Add liquidation signals for thawed tickers
        for ticker in to_liquidate:
            signals.append(Signal(
                date=prices.date, ticker=ticker, target_weight=0.0,
                reason="liquidate_frozen",
            ))

        # 5. Vol regime → effective risky fraction
        regime, regime_cap = self._vol_detector.detect(risky_history)
        effective_fraction = min(self.max_risky_fraction, regime_cap)

        # 6. Normalise with regime-adjusted fraction
        weights = {s.ticker: s.target_weight for s in signals}
        weights = normalise_weights(weights, portfolio, effective_fraction)
        signals = [
            Signal(
                date=s.date, ticker=s.ticker, target_weight=weights.get(s.ticker, s.target_weight),
                reason=s.reason, info={**(s.info or {}), "regime": regime},
            )
            for s in signals if s.ticker in weights
        ]

        # 7. DN bands filter
        if self.use_dn_bands and self._signaler.last_mu is not None:
            signals = NoTradeBandFilter(
                gamma=self.gamma,
                r=self._signaler.last_rfr,
                mu=self._signaler.last_mu,
                cov=self._signaler.last_cov,
                costs=self.universe.transaction_costs,
                band_scale=self.band_scale,
            ).filter(signals, portfolio, prices)

        # 8. Defensive
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # 9. Cost-aware filter
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals
