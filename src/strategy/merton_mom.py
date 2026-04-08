from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.mu import MuEstimator, HistoricalMean
from src.models.estimators.covariance import CovEstimator, SampleCovariance
from src.models.leverage_guard import LeverageGuard
from src.strategy.signaler import MertonSignaler, MomentumSignaler
from src.strategy.filter.no_trade_band import NoTradeBandFilter
from src.strategy.filter.cost_aware import CostAwareFilter
from src.strategy.transformers import fill_defensive

from .base import Strategy


@dataclass(kw_only=True)
class MertonMomentumStrategy(Strategy):
    """Merton + momentum + Davis-Norman bands.

    Pipeline:
      1. MertonSignaler       → optimal risky weights
      2. MomentumSignaler     → reduce weight for negative momentum tickers
      3. NoTradeBandFilter    → skip tickers inside DN band
      4. fill_defensive       → defensive absorbs remainder
      5. CostAwareFilter      → ensure executable
      6. LeverageGuard        → force rebalance if leverage drifts
    """

    defensive_ticker: str
    gamma: float = 0.5
    mu_estimator: MuEstimator = field(default_factory=HistoricalMean)
    cov_estimator: CovEstimator = field(default_factory=SampleCovariance)
    rebalance_every: int = 21
    max_risky_fraction: float = 0.8
    # Momentum
    momentum_lookback: int = 252
    momentum_skip: int = 21
    momentum_alpha_min: float = 0.3
    # DN bands
    use_dn_bands: bool = True
    band_scale: float = 1.0

    _step_count: int = field(default=0, init=False, repr=False)
    _merton: MertonSignaler = field(init=False, repr=False)
    _momentum: MomentumSignaler = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "merton_mom"

    def _warmup(self, portfolio: Portfolio) -> None:
        self._merton = MertonSignaler(
            risky_tickers=self._risky_tickers,
            defensive_ticker=self.defensive_ticker,
            gamma=self.gamma,
            mu_estimator=self.mu_estimator,
            cov_estimator=self.cov_estimator,
            max_risky_fraction=self.max_risky_fraction,
        )
        self._momentum = MomentumSignaler(
            lookback=self.momentum_lookback,
            skip=self.momentum_skip,
            alpha_min=self.momentum_alpha_min,
        )
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
        signals = self._merton.generate(self._history, prices, portfolio)
        if not signals:
            return []

        # 2. Momentum scaling
        signals = self._momentum.scale(signals, self._history)

        # 3. DN bands filter
        if self.use_dn_bands and self._merton.last_mu is not None:
            signals = NoTradeBandFilter(
                gamma=self.gamma,
                r=self._merton.last_rfr,
                mu=self._merton.last_mu,
                cov=self._merton.last_cov,
                costs=self.universe.transaction_costs,
                band_scale=self.band_scale,
            ).filter(signals, portfolio, prices)

        # 4. Defensive
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # 5. Cost-aware filter
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals
