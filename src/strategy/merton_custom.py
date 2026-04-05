from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.estimators.mu import MuEstimator, JamesSteinMean
from src.models.estimators.covariance import CovEstimator, LedoitWolfCovariance
from src.models.leverage_guard import LeverageGuard
from src.strategy.signaler import MertonSignaler, MomentumSignaler
from src.strategy.filter.top_n_by_weight import TopNByWeightFilter
from src.strategy.filter.no_trade_band import NoTradeBandFilter
from src.strategy.filter.cost_aware import CostAwareFilter
from src.strategy.transformers import normalise_weights, fill_defensive

from .base import Strategy


@dataclass(kw_only=True)
class MertonCustomStrategy(Strategy):
    """Merton top-N + momentum + Davis-Norman, fully invested.

    Pipeline:
      1. MertonSignaler       → weights for all risky tickers
      2. TopNByWeightFilter   → keep top N by Merton weight
      3. MomentumSignaler     → penalise negative-momentum tickers
      4. Normalise            → scale to max_risky_fraction
      5. NoTradeBandFilter    → skip tickers inside DN band
      6. fill_defensive       → defensive absorbs remainder
      7. CostAwareFilter      → ensure executable
      8. LeverageGuard        → force rebalance if leverage drifts
    """

    defensive_ticker: str
    gamma: float = 0.5
    top_n: int = 5
    max_risky_fraction: float = 0.8
    rebalance_every: int = 21
    band_scale: float = 2.0
    momentum_alpha_min: float = 0.3
    mu_estimator: MuEstimator = field(default_factory=JamesSteinMean)
    cov_estimator: CovEstimator = field(default_factory=LedoitWolfCovariance)

    _step_count: int = field(default=0, init=False, repr=False)
    _signaler: MertonSignaler = field(init=False, repr=False)
    _top_n_filter: TopNByWeightFilter = field(init=False, repr=False)
    _leverage_guard: LeverageGuard = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = "merton_custom"

    def _warmup(self, portfolio: Portfolio) -> None:
        self._signaler = MertonSignaler(
            risky_tickers=self._risky_tickers,
            defensive_ticker=self.defensive_ticker,
            gamma=self.gamma,
            mu_estimator=self.mu_estimator,
            cov_estimator=self.cov_estimator,
            max_risky_fraction=1.0,  # normalise after top-N
        )
        self._top_n_filter = TopNByWeightFilter(n=self.top_n)
        self._momentum = MomentumSignaler(alpha_min=self.momentum_alpha_min)
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

        # 1. Merton optimal weights (all risky)
        signals = self._signaler.generate(self._history, prices, portfolio)
        if not signals:
            return []

        # 2. Keep top-N by Merton weight, liquidate the rest
        selected = self._top_n_filter.filter(signals)
        selected_tickers = {s.ticker for s in selected}

        current_weights = portfolio.positions_weights(prices)
        for ticker in self._risky_tickers:
            if ticker not in selected_tickers and current_weights.get(ticker, 0.0) > 1e-9:
                selected.append(Signal(
                    date=prices.date, ticker=ticker, target_weight=0.0,
                    reason="excluded",
                ))
        signals = selected

        # 3. Momentum: penalise negative-momentum tickers
        signals = self._momentum.scale(signals, self._history)

        # 4. Normalise to max_risky_fraction
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

        # 5. DN bands
        if self._signaler.last_mu is not None:
            signals = NoTradeBandFilter(
                gamma=self.gamma,
                r=self._signaler.last_rfr,
                mu=self._signaler.last_mu,
                cov=self._signaler.last_cov,
                costs=self.universe.transaction_costs,
                band_scale=self.band_scale,
            ).filter(signals, portfolio, prices)

        # 6. Defensive
        signals = fill_defensive(
            signals, self._risky_tickers, self.defensive_ticker, prices, portfolio,
        )

        # 7. Cost filter
        signals = CostAwareFilter(
            transaction_costs=self.universe.transaction_costs,
        ).filter(signals, portfolio, prices)

        return signals
