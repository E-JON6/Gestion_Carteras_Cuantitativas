from dataclasses import dataclass, field

from ..asset import PriceSnapshot
from .snapshot import PortfolioSnapshot


class PortfolioError(Exception):
    ...

@dataclass
class Portfolio:
    """Pure state container for a portfolio.

    Knows only cash and positions (shares per ticker).
    Has NO knowledge of prices, commissions, or strategies.
    Does NOT validate business rules — that's the Broker's job.
    """

    initial_cash: float
    initial_positions: dict[str, float] | None = None

    max_leverage: float = 1.0
    allow_short: bool = False

    _cash: float = field(init=False)
    _positions: dict[str, float] = field(init=False)

    def __post_init__(self):
        self._cash = self.initial_cash
        self._positions = {}
        if self.initial_positions is not None:
            for ticker, shares in self.initial_positions.items():
                self._positions[ticker] = shares

    # Properties

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Copy of current positions."""
        return dict(self._positions)

    def shares(self, ticker: str) -> float:
        """Shares held for a specific ticker."""
        return self._positions.get(ticker, 0.0)

    # Queries (prices as argument, never stored)

    def positions_values(self, prices: PriceSnapshot) -> dict[str, float]:
        """Monetary value of each position."""
        return {t: s * prices.get_price(t) for t, s in self._positions.items()}

    def total_positions_value(self, prices: PriceSnapshot) -> float:
        """Total monetary value of all positions."""
        return sum(self.positions_values(prices).values())

    def total_value(self, prices: PriceSnapshot) -> float:
        """Total portfolio value = cash + sum of positions values."""
        return self._cash + self.total_positions_value(prices)

    def alpha(self, prices: PriceSnapshot) -> float:
        """Fraction of portfolio invested (not in cash)."""
        total = self.total_value(prices)
        if total == 0:
            return 0.0
        return (total - self._cash) / total

    def positions_weights(self, prices: PriceSnapshot) -> dict[str, float]:
        """Current weight of each position in the portfolio."""
        total = self.total_value(prices)
        if total == 0:
            return {t: 0.0 for t in self._positions}
        pos = self.positions_values(prices)
        return {t: v / total for t, v in pos.items()}

    # Mutations

    def execute_trade(self, ticker: str, shares_delta: float, cash_delta: float) -> None:
        """Apply trade, then validate consistency. Rollback on violation.

        Business-rule validation (shorts, cash, leverage) is the Broker's
        job.  The checks here are a post-condition safety net: if they
        fire, it means the Broker accepted an order it shouldn't have.
        """
        prev_shares = self._positions.get(ticker, 0.0)
        prev_cash = self._cash

        self._positions[ticker] = prev_shares + shares_delta
        self._cash = prev_cash + cash_delta

        try:
            self._validate_consistency(ticker)
        except PortfolioError:
            self._positions[ticker] = prev_shares
            self._cash = prev_cash
            raise

    def _validate_consistency(self, ticker: str) -> None:
        """Check portfolio invariants after a trade."""
        if not self.allow_short and self._positions[ticker] < -1e-9:
            raise PortfolioError(
                f"inconsistent state: position {ticker} = {self._positions[ticker]:.6f}"
            )
        if self.max_leverage <= 1.0 and self._cash < -1e-9:
            raise PortfolioError(
                f"inconsistent state: cash = {self._cash:.2f}"
            )

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash = self.initial_cash
        self._positions = {}
        if self.initial_positions is not None:
            for ticker, shares in self.initial_positions.items():
                self._positions[ticker] = shares

    # Snapshot

    def snapshot(self, prices: PriceSnapshot) -> PortfolioSnapshot:
        """Capture current state as an immutable snapshot."""
        pos_values = self.positions_values(prices)
        total_pos = sum(pos_values.values())
        total = self._cash + total_pos
        alpha = (total - self._cash) / total if total != 0 else 0.0
        weights = {t: v / total for t, v in pos_values.items()} if total != 0 else {t: 0.0 for t in self._positions}

        return PortfolioSnapshot(
            date=prices.date,
            total_value=total,
            cash=self._cash,
            total_positions_value=total_pos,
            alpha=alpha,
            positions=self.positions,
            positions_values=pos_values,
            positions_weights=weights,
        )
