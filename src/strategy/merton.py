from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.mu import MuEstimator, HistoricalMean
from src.models.estimators.covariance import CovEstimator, SampleCovariance
from src.strategy.signaler import MertonSignaler
from src.strategy.filter.no_trade_band import NoTradeBandFilter
from src.strategy.filter.cost_aware import CostAwareFilter
from src.models.leverage_guard import LeverageGuard
from src.strategy.transformers import fill_defensive

from .base import Strategy


@dataclass(kw_only=True)
class MertonStrategy(Strategy):
    """Merton optimal allocation, fully invested.

    Pipeline: signaler → DN filter → defensive → cost filter.
    """

    defensive_ticker: str
    gamma: float = 0.5
    mu_estimator: MuEstimator = field(default_factory=HistoricalMean)
    cov_estimator: CovEstimator = field(default_factory=SampleCovariance)
    rebalance_every: int = 1
    use_dn_bands: bool = False
    band_scale: float = 1.0
    max_risky_fraction: float = 1.0

    _step_count: int = field(default=0, init=False, repr=False)
    _signaler: MertonSignaler = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "merton_dn" if self.use_dn_bands else "merton"

    def _warmup(self, portfolio: Portfolio) -> None:
        self._signaler = MertonSignaler(
            risky_tickers=self._risky_tickers,
            defensive_ticker=self.defensive_ticker,
            gamma=self.gamma,
            mu_estimator=self.mu_estimator,
            cov_estimator=self.cov_estimator,
            max_risky_fraction=self.max_risky_fraction,
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

        # 1. Signaler
        signals = self._signaler.generate(self._history, prices, portfolio)
        if not signals:
            return []

        # 2. DN bands filter
        if self.use_dn_bands and self._signaler.last_mu is not None:
            signals = NoTradeBandFilter(
                gamma=self.gamma,
                r=self._signaler.last_rfr,
                mu=self._signaler.last_mu,
                cov=self._signaler.last_cov,
                costs=self.universe.transaction_costs,
                band_scale=self.band_scale,
            ).filter(signals, portfolio, prices)

        # 3. Defensive
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # 4. Cost-aware filter
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals
