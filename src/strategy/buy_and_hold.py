from dataclasses import dataclass, field

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal
from .base import Strategy


@dataclass(kw_only=True)
class BuyAndHoldStrategy(Strategy):
    """Allocate to target weights and optionally rebalance.

    With ``rebalance_every=0`` (default): buy once and never touch.
    With ``rebalance_every=N``: rebalance to target weights every N days.

    If ``weights`` is None, distributes equal-weight across all
    tickers except those in ``exclude``.

    Args:
        weights: ticker -> target weight. None = equal-weight.
        exclude: tickers to exclude from equal-weight (e.g. defensive).
        rebalance_every: Days between rebalances. 0 = never (true buy & hold).
    """

    weights: dict[str, float] | None = None
    exclude: list[str] = field(default_factory=list)
    rebalance_every: int = 0

    _invested: bool = field(default=False, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            if self.rebalance_every > 0:
                self.name = "equal_weight_rebal"
            else:
                self.name = "buy_and_hold"

    def _warmup(self, portfolio: Portfolio) -> None:
        if self.weights is None:
            tickers = [t for t in self.universe.tickers if t not in self.exclude]
            n = len(tickers)
            self.weights = {t: portfolio.max_leverage / n for t in tickers}

    def reset(self) -> None:
        super().reset()
        self._invested = False
        self._step_count = 0

    def _generate_signals(
        self,
        prices: PriceSnapshot,
        portfolio: Portfolio,
    ) -> list[Signal]:
        self._step_count += 1

        if not self._invested:
            self._invested = True
            return self._emit(prices)

        if self.rebalance_every > 0 and self._step_count % self.rebalance_every == 0:
            return self._emit(prices)

        return []

    def _emit(self, prices: PriceSnapshot) -> list[Signal]:
        return [
            Signal(date=prices.date, ticker=t, target_weight=w, reason=self.name)
            for t, w in self.weights.items()
        ]
