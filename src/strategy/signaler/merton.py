from dataclasses import dataclass, field

from src.domain.asset import PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from src.models.merton_davisnorman import MertonDavisNormanModel
from src.models.estimators.r import DefensiveRfrEstimator
from src.models.estimators.mu import MuEstimator, HistoricalMean
from src.models.estimators.covariance import CovEstimator, SampleCovariance
from src.strategy.transformers import normalise_weights


@dataclass
class MertonSignaler:
    """Generates target weight signals from Merton optimal allocation.

    Computes mu, cov, and risk-free rate from price history, then
    calculates Merton optimal weights for risky tickers. Normalises
    weights to respect portfolio constraints.

    After calling ``generate()``, the last estimated rfr, mu, and cov
    are available via ``last_rfr``, ``last_mu``, ``last_cov`` (useful
    for downstream filters like NoTradeBandFilter).
    """

    risky_tickers: list[str]
    defensive_ticker: str
    gamma: float = 0.5
    mu_estimator: MuEstimator = field(default_factory=HistoricalMean)
    cov_estimator: CovEstimator = field(default_factory=SampleCovariance)
    max_risky_fraction: float = 1.0

    def __post_init__(self):
        self._rfr_estimator = DefensiveRfrEstimator(self.defensive_ticker)
        self._last_rfr: float | None = None
        self._last_mu: dict[str, float] | None = None
        self._last_cov = None

    @property
    def last_rfr(self) -> float | None:
        return self._last_rfr

    @property
    def last_mu(self) -> dict[str, float] | None:
        return self._last_mu

    @property
    def last_cov(self):
        return self._last_cov

    def generate(
        self,
        history: PriceHistory,
        prices: PriceSnapshot,
        portfolio: Portfolio,
    ) -> list[Signal]:
        """Generate Merton optimal weight signals for risky tickers."""
        risky_history = history.select(self.risky_tickers)

        if len(risky_history) < 2:
            return []

        try:
            self._last_rfr = self._rfr_estimator.estimate(history)
            self._last_mu = self.mu_estimator.estimate(risky_history)
            self._last_cov = self.cov_estimator.estimate(risky_history)

            model = MertonDavisNormanModel(gamma=self.gamma, r=self._last_rfr)
            raw_weights = model.optimal_weights(self._last_mu, self._last_cov)
        except Exception:
            return []

        weights = normalise_weights(raw_weights, portfolio, self.max_risky_fraction)

        return [
            Signal(
                date=prices.date,
                ticker=t,
                target_weight=float(w),
                reason="merton",
                info={
                    "mu": self._last_mu[t],
                    "r": self._last_rfr,
                    "raw_weight": raw_weights[t],
                },
            )
            for t, w in weights.items()
        ]
