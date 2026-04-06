from dataclasses import dataclass

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.portfolio.base import PortfolioError
from .trade import Trade
from .signal import Signal
from .order import Order, PendingOrder, AcceptedOrder


class BrokerError(Exception):
    """Exception for Broker errors."""


@dataclass(frozen=True)
class Broker:
    """Receives signals, creates orders, validates, and executes trades.

    Flow: Signal -> PendingOrder -> AcceptedOrder -> Trade
                                 -> RejectedOrder (no Trade)
                                 -> InvalidOrder (no Trade)
    """

    transaction_costs: dict[str, float]
    buy_adjustment: float = 0.005

    def execute(
        self,
        portfolio: Portfolio,
        signals: list[Signal],
        prices: PriceSnapshot,
    ) -> tuple[list[Order], list[Trade]]:
        """Process signals end-to-end."""
        orders = self._create_orders(portfolio, signals, prices)
        orders = self._validate_orders(portfolio, orders, prices)
        trades = self._execute_accepted(portfolio, orders)
        return orders, trades

    # ── Signal → Order ──────────────────────────────────────────────────

    def _create_orders(
        self,
        portfolio: Portfolio,
        signals: list[Signal],
        prices: PriceSnapshot,
    ) -> list[Order]:
        """Convert every signal into a PendingOrder or InvalidOrder."""
        current_weights = portfolio.positions_weights(prices)
        current_positions = portfolio.positions
        effective_value = self._effective_value(portfolio, signals, prices)

        return [
            self._signal_to_order(
                s, current_weights, current_positions, effective_value, prices,
            )
            for s in signals
        ]

    def _signal_to_order(
        self,
        signal: Signal,
        current_weights: dict[str, float],
        current_positions: dict[str, float],
        effective_value: float,
        prices: PriceSnapshot,
    ) -> Order:
        current_w = current_weights.get(signal.ticker, 0.0)

        if abs(signal.target_weight - current_w) < 1e-6:
            return signal.invalidate("weight unchanged")

        try:
            price = prices.get_price(signal.ticker)
        except Exception:
            return signal.invalidate(f"unknown ticker: {signal.ticker}")

        # Full liquidation: sell exact shares held (no rounding loss)
        if signal.target_weight == 0.0 and signal.ticker in current_positions:
            shares = -current_positions[signal.ticker]
            return signal.order(shares, price)

        shares = self._target_shares(signal.target_weight, current_w,
                                     effective_value, price)
        return signal.order(shares, price)

    def _target_shares(
        self,
        target_w: float,
        current_w: float,
        effective_value: float,
        price: float,
    ) -> float:
        """Shares needed to move from current_w to target_w.

        Buys are reduced by ``buy_adjustment`` to leave a cash buffer.
        """
        delta_value = (target_w - current_w) * effective_value
        if delta_value > 0:
            delta_value *= (1 - self.buy_adjustment)
        shares = delta_value / price
        return float(int(shares))  # truncate toward zero: whole shares only

    def _effective_value(
        self,
        portfolio: Portfolio,
        signals: list[Signal],
        prices: PriceSnapshot,
    ) -> float:
        """Portfolio value minus estimated total transaction costs."""
        total_value = portfolio.total_value(prices)
        current_weights = portfolio.positions_weights(prices)

        estimated_costs = sum(
            abs(s.target_weight - current_weights.get(s.ticker, 0.0))
            * total_value
            * self._tx_rate(s.ticker)
            for s in signals
        )
        return total_value - estimated_costs

    # ── Validate ────────────────────────────────────────────────────────

    def _validate_orders(
        self,
        portfolio: Portfolio,
        orders: list[Order],
        prices: PriceSnapshot,
    ) -> list[Order]:
        """Accept or reject every pending order (shorts, cash, leverage)."""
        pending = [o for o in orders if isinstance(o, PendingOrder)]
        non_pending = [o for o in orders if not isinstance(o, PendingOrder)]

        sells = self._validate_sells(pending, portfolio)
        accepted_sells = [o for o in sells if isinstance(o, AcceptedOrder)]
        buys = self._validate_buys(pending, portfolio, accepted_sells, prices)

        return non_pending + sells + buys

    # — Sells

    def _validate_sells(
        self,
        pending: list[PendingOrder],
        portfolio: Portfolio,
    ) -> list[Order]:
        """Accept sells; reject those that would create a disallowed short."""
        sell_orders = [o for o in pending if o.shares < 0]

        if portfolio.allow_short:
            return [o.accept() for o in sell_orders]

        projected = dict(portfolio.positions)
        validated: list[Order] = []

        for order in sell_orders:
            new_shares = projected.get(order.ticker, 0.0) + order.shares
            if new_shares < -1e-9:
                validated.append(order.reject("short selling not allowed"))
            else:
                projected[order.ticker] = new_shares
                validated.append(order.accept())

        return validated

    # — Buys

    def _validate_buys(
        self,
        pending: list[PendingOrder],
        portfolio: Portfolio,
        accepted_sells: list[AcceptedOrder],
        prices: PriceSnapshot,
    ) -> list[Order]:
        """Accept buys by descending target weight, checking cash and leverage."""
        remaining_cash = self._available_cash(portfolio, accepted_sells, prices)
        projected = self._projected_positions(portfolio, accepted_sells)
        nav = portfolio.total_value(prices)

        buys = sorted(
            [o for o in pending if o.shares > 0],
            key=lambda o: o.signal.target_weight,
            reverse=True,
        )

        validated: list[Order] = []
        for order in buys:
            total_cost = self._order_cost_with_tx(order)

            if total_cost > remaining_cash:
                validated.append(order.reject("insufficient cash"))
                continue

            if self._would_exceed_leverage(order, projected, nav, portfolio, prices):
                validated.append(order.reject("exceeds max leverage"))
                continue

            projected[order.ticker] = projected.get(order.ticker, 0.0) + order.shares
            validated.append(order.accept())
            remaining_cash -= total_cost

        return validated

    def _available_cash(
        self,
        portfolio: Portfolio,
        accepted_sells: list[AcceptedOrder],
        prices: PriceSnapshot,
    ) -> float:
        """Cash available for buying, accounting for leverage.

        With max_leverage=1: limited to current cash + sell proceeds.
        With max_leverage>1: the portfolio can borrow up to
        NAV * (max_leverage - 1) on top of its cash.
        """
        sell_proceeds = sum(
            o.abs_value_before_costs * (1 - self._tx_rate(o.ticker))
            for o in accepted_sells
        )
        cash = portfolio.cash + sell_proceeds

        if portfolio.max_leverage > 1.0:
            nav = portfolio.total_value(prices)
            borrowing_capacity = nav * (portfolio.max_leverage - 1.0)
            return cash + borrowing_capacity

        return cash

    def _projected_positions(
        self,
        portfolio: Portfolio,
        accepted_sells: list[AcceptedOrder],
    ) -> dict[str, float]:
        """Portfolio positions after applying accepted sells."""
        projected = dict(portfolio.positions)
        for sell in accepted_sells:
            projected[sell.ticker] = projected.get(sell.ticker, 0.0) + sell.shares
        return projected

    def _order_cost_with_tx(self, order: PendingOrder) -> float:
        """Order value including its transaction cost."""
        value = order.abs_value_before_costs
        return value * (1 + self._tx_rate(order.ticker))

    def _would_exceed_leverage(
        self,
        order: PendingOrder,
        projected: dict[str, float],
        nav: float,
        portfolio: Portfolio,
        prices: PriceSnapshot,
    ) -> bool:
        """Would accepting this order push gross leverage above max_leverage?"""
        test = dict(projected)
        test[order.ticker] = test.get(order.ticker, 0.0) + order.shares

        gross = sum(
            abs(s) * prices.get_price(t)
            for t, s in test.items() if s != 0
        )
        return nav > 0 and gross / nav > portfolio.max_leverage

    # ── Execute ─────────────────────────────────────────────────────────

    def _execute_accepted(
        self,
        portfolio: Portfolio,
        orders: list[Order],
    ) -> list[Trade]:
        """Execute accepted orders: sells first, then buys."""
        sells = [o for o in orders if isinstance(o, AcceptedOrder) and o.shares < 0]
        buys = [o for o in orders if isinstance(o, AcceptedOrder) and o.shares > 0]

        return [self._settle(portfolio, o) for o in sells + buys]

    def _settle(self, portfolio: Portfolio, order: AcceptedOrder) -> Trade:
        """Settle a single accepted order against the portfolio."""
        cost = order.abs_value_before_costs * self._tx_rate(order.ticker)
        cash_delta = -(order.shares * order.price) - cost

        try:
            portfolio.execute_trade(order.ticker, order.shares, cash_delta)
        except PortfolioError as e:
            raise BrokerError(
                f"Failed to execute order for {order.ticker}: {e}"
            ) from e

        return Trade(order=order, cost=cost)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _tx_rate(self, ticker: str) -> float:
        """Transaction cost rate for a ticker (0.0 if unknown)."""
        return self.transaction_costs.get(ticker, 0.0)
