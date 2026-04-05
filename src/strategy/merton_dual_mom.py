from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.mu import MuEstimator, JamesSteinMean
from src.models.estimators.covariance import CovEstimator, LedoitWolfCovariance
from src.models.leverage_guard import LeverageGuard
from src.strategy.signaler import MertonSignaler, DualMomentumSignaler
from src.strategy.filter.top_n_by_weight import TopNByWeightFilter
from src.strategy.filter.no_trade_band import NoTradeBandFilter
from src.strategy.filter.cost_aware import CostAwareFilter
from src.strategy.transformers import normalise_weights, fill_defensive

from .base import Strategy


@dataclass(kw_only=True)
class MertonDualMomentumStrategy(Strategy):
    """Merton + dual momentum + Davis-Norman, fully invested.

    Pipeline:
      1. MertonSignaler         → weights for all risky tickers
      2. DualMomentumSignaler   → if market momentum < 0: all to defensive
                                  else: drop tickers below median momentum
      3. TopNByWeightFilter     → keep top N survivors
      4. Liquidate excluded     → signal weight=0 for held tickers not selected
      5. Normalise              → scale to max_risky_fraction
      6. NoTradeBandFilter      → skip tickers inside DN band
      7. fill_defensive         → defensive absorbs remainder
      8. CostAwareFilter        → ensure executable
      9. LeverageGuard          → force rebalance if leverage drifts

    When the market is off (negative absolute momentum), the strategy
    generates no risky signals — fill_defensive puts 100% in XEON.DE.
    When the market is on, only tickers with above-median momentum
    pass through to Merton's top-N selection.
    """

    defensive_ticker: str
    gamma: float = 0.5
    top_n: int = 5
    max_risky_fraction: float = 0.8
    rebalance_every: int = 21
    band_scale: float = 2.0
    momentum_lookback: int = 252
    momentum_skip: int = 21
    mu_estimator: MuEstimator = field(default_factory=JamesSteinMean)
    cov_estimator: CovEstimator = field(default_factory=LedoitWolfCovariance)

    _step_count: int = field(default=0, init=False, repr=False)
    _signaler: MertonSignaler = field(init=False, repr=False)
    _dual_mom: DualMomentumSignaler = field(init=False, repr=False)
    _top_n_filter: TopNByWeightFilter = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "merton_dual_mom"

    def _warmup(self, portfolio: Portfolio) -> None:
        self._signaler = MertonSignaler(
            risky_tickers=self._risky_tickers,
            defensive_ticker=self.defensive_ticker,
            gamma=self.gamma,
            mu_estimator=self.mu_estimator,
            cov_estimator=self.cov_estimator,
            max_risky_fraction=1.0,
        )
        self._dual_mom = DualMomentumSignaler(
            lookback=self.momentum_lookback,
            skip=self.momentum_skip,
        )
        self._top_n_filter = TopNByWeightFilter(n=self.top_n)
        self._leverage_guard = LeverageGuard(max_leverage=portfolio.max_leverage)

    def reset(self) -> None:
        super().reset()
        self._step_count = 0

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

        # 2. Dual momentum: market off → empty, market on → drop below-median
        signals = self._dual_mom.filter(signals, self._history)

        # 3. Top-N by Merton weight (from survivors)
        signals = self._top_n_filter.filter(signals)

        # 5. Normalise
        if signals:
            weights = normalise_weights(
                {s.ticker: s.target_weight for s in signals},
                portfolio,
                self.max_risky_fraction,
            )
            signals = [
                Signal(date=s.date, ticker=s.ticker, target_weight=weights[s.ticker],
                       reason=s.reason, info=s.info)
                for s in signals
            ]

        # 6. DN bands
        if signals and self._signaler.last_mu is not None:
            signals = NoTradeBandFilter(
                gamma=self.gamma,
                r=self._signaler.last_rfr,
                mu=self._signaler.last_mu,
                cov=self._signaler.last_cov,
                costs=self.universe.transaction_costs,
                band_scale=self.band_scale,
            ).filter(signals, portfolio, prices)

        # 7. Defensive
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # 8. Cost filter
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals
