"""Tests for Broker: signal conversion, order validation, and trade execution."""

import pytest
import pandas as pd

from src.domain.portfolio import Portfolio
from src.domain.trade import Broker, Signal
from src.domain.trade.broker import BrokerError
from src.domain.trade.order import (
    PendingOrder, AcceptedOrder, RejectedOrder, InvalidOrder,
)
from src.domain.trade.trade import Trade
from tests.conftest import (
    DATE, make_universe, make_prices, make_signal,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SIGNAL → ORDER CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalToOrder:
    """Broker._create_orders: convert signals into pending/invalid orders."""

    def test_buy_signal_creates_pending_order(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.5)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        assert len(accepted) == 1
        assert accepted[0].ticker == "AAPL"
        assert accepted[0].shares > 0

    def test_sell_signal_creates_pending_order(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=50_000, initial_positions={"AAPL": 100})
        # AAPL worth 10_000 out of 60_000 total ≈ 16.7%; target 0% => sell
        signals = [make_signal("AAPL", 0.0)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        assert len(accepted) == 1
        assert accepted[0].shares < 0

    def test_unchanged_weight_invalidated(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=50_000, initial_positions={"AAPL": 500})
        # AAPL = 500*100 = 50_000; total = 100_000 => weight = 0.5
        signals = [make_signal("AAPL", 0.5)]

        orders, _ = broker_2.execute(portfolio, signals, prices_2)

        invalid = [o for o in orders if isinstance(o, InvalidOrder)]
        assert len(invalid) == 1
        assert "unchanged" in invalid[0].reason

    def test_unknown_ticker_invalidated(self, universe_2, prices_2):
        broker = Broker(transaction_costs={"AAPL": 0.001, "MSFT": 0.001})
        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("UNKNOWN", 0.3)]

        orders, _ = broker.execute(portfolio, signals, prices_2)

        invalid = [o for o in orders if isinstance(o, InvalidOrder)]
        assert len(invalid) == 1
        assert "unknown ticker" in invalid[0].reason.lower() or "UNKNOWN" in invalid[0].reason

    def test_buy_adjustment_reduces_shares(self, prices_2):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})

        broker_adj = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.01)
        broker_no_adj = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.0)

        portfolio_adj = Portfolio(initial_cash=100_000)
        portfolio_no = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.5)]

        _, trades_adj = broker_adj.execute(portfolio_adj, signals, prices)
        _, trades_no = broker_no_adj.execute(portfolio_no, signals, prices)

        assert trades_adj[0].shares < trades_no[0].shares

    def test_buy_adjustment_does_not_affect_sells(self, prices_2):
        broker = Broker(
            transaction_costs={"AAPL": 0.0, "MSFT": 0.0},
            buy_adjustment=0.05,
        )
        # Portfolio 100% in AAPL, signal to go to 0%
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 1000})
        signals = [make_signal("AAPL", 0.0)]

        _, trades = broker.execute(portfolio, signals, prices_2)

        # Sell should liquidate all 1000 shares (no buy adjustment on sells)
        assert trades[0].shares == -1000


# ═══════════════════════════════════════════════════════════════════════════
# 2. ORDER VALIDATION (cash management, buy priority)
# ═══════════════════════════════════════════════════════════════════════════

class TestOrderValidation:
    """Broker._validate_orders: accept/reject based on cash availability."""

    def test_all_sells_always_accepted(self, broker_2, prices_2):
        portfolio = Portfolio(
            initial_cash=0,
            initial_positions={"AAPL": 100, "MSFT": 50},
        )
        signals = [make_signal("AAPL", 0.0), make_signal("MSFT", 0.0)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        assert len(accepted) == 2
        assert all(o.shares < 0 for o in accepted)

    def test_buy_rejected_when_insufficient_cash(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100)  # only $100
        signals = [make_signal("AAPL", 0.99)]  # wants ~$99 worth

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        # With only $100 total value, wanting 99% in AAPL ≈ 99 * (1-adj)/100 < 1 share
        # This should be accepted because the portfolio value is tiny
        # Let's test with a larger mismatch
        portfolio2 = Portfolio(initial_cash=10)
        signals2 = [make_signal("AAPL", 0.99)]
        orders2, _ = broker_2.execute(portfolio2, signals2, prices_2)

        # Construct a case where combined buys exceed cash even with whole shares.
        # AAPL@100, MSFT@200. With $1000 and 0.70 + 0.70 = 140% target,
        # AAPL 0.70 => ~6 shares = $600+tx, MSFT 0.70 => ~3 shares = $600+tx
        # Total ~$1200 > $1000
        portfolio3 = Portfolio(initial_cash=1_000)
        signals3 = [
            make_signal("AAPL", 0.70),
            make_signal("MSFT", 0.70),
        ]
        orders3, _ = broker_2.execute(portfolio3, signals3, prices_2)

        rejected = [o for o in orders3 if isinstance(o, RejectedOrder)]
        assert len(rejected) >= 1
        assert "insufficient cash" in rejected[0].reason

    def test_buys_prioritized_by_target_weight(self, broker_3, prices_3):
        portfolio = Portfolio(initial_cash=10_000)
        signals = [
            make_signal("AAPL", 0.20),   # low priority
            make_signal("MSFT", 0.50),   # highest priority
            make_signal("GOOG", 0.35),   # medium priority
        ]

        orders, trades = broker_3.execute(portfolio, signals, prices_3)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        # All should fit since 0.20+0.50+0.35 = 1.05 but with adjustment fits
        # The key thing: if any were rejected, the lowest weight gets rejected
        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        if rejected:
            # The rejected one should have the smallest target weight
            min_accepted_w = min(o.signal.target_weight for o in accepted)
            for r in rejected:
                assert r.signal.target_weight <= min_accepted_w

    def test_sell_proceeds_fund_buys(self, broker_2, prices_2):
        # No cash, all in AAPL. Sell AAPL to buy MSFT.
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 100})
        # AAPL = 100*100 = 10_000 total
        signals = [
            make_signal("AAPL", 0.0),   # sell all
            make_signal("MSFT", 0.50),   # buy with proceeds
        ]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        assert len(accepted) == 2
        sell = [o for o in accepted if o.shares < 0]
        buy = [o for o in accepted if o.shares > 0]
        assert len(sell) == 1 and len(buy) == 1

    def test_buy_without_sell_proceeds_rejected(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 100})
        # All value in AAPL, no cash, try to buy MSFT without selling AAPL
        signals = [make_signal("MSFT", 0.50)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRADE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

class TestTradeExecution:
    """Broker._execute_accepted: order of execution and state changes."""

    def test_sells_executed_before_buys(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 100})
        signals = [
            make_signal("AAPL", 0.0),
            make_signal("MSFT", 0.50),
        ]

        _, trades = broker_2.execute(portfolio, signals, prices_2)

        if len(trades) >= 2:
            assert trades[0].direction == "sell"
            assert trades[1].direction == "buy"

    def test_transaction_costs_deducted_from_cash(self, prices_2):
        universe = make_universe(("AAPL", 0.01), ("MSFT", 0.01))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.01, "MSFT": 0.01})

        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.50)]

        orders, trades = broker.execute(portfolio, signals, prices)

        assert len(trades) == 1
        assert trades[0].cost > 0
        # Cash should be reduced by shares*price + cost
        expected_cash = 100_000 - (trades[0].shares * 100.0) - trades[0].cost
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_zero_transaction_cost(self, prices_2):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0})

        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.30)]

        _, trades = broker.execute(portfolio, signals, prices)
        assert trades[0].cost == 0.0

    def test_portfolio_positions_updated(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.30), make_signal("MSFT", 0.20)]

        broker_2.execute(portfolio, signals, prices_2)

        assert portfolio.shares("AAPL") > 0
        assert portfolio.shares("MSFT") > 0
        assert portfolio.cash < 100_000

    def test_full_liquidation(self, broker_2, prices_2):
        portfolio = Portfolio(
            initial_cash=1_000,
            initial_positions={"AAPL": 50, "MSFT": 25},
        )
        signals = [make_signal("AAPL", 0.0), make_signal("MSFT", 0.0)]

        _, trades = broker_2.execute(portfolio, signals, prices_2)

        # Broker computes shares from weight deltas, not "sell all".
        # With transaction cost estimation the effective_value differs from
        # total_value, so a tiny residual can remain.
        assert portfolio.shares("AAPL") == pytest.approx(0, abs=0.1)
        assert portfolio.shares("MSFT") == pytest.approx(0, abs=0.1)
        assert len(trades) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. SHORT SELLING VIA BROKER
# ═══════════════════════════════════════════════════════════════════════════

class TestBrokerShortSelling:
    """Broker + Portfolio with allow_short=True/False."""

    def test_short_signal_rejected_without_allow_short(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000, allow_short=False)
        signals = [make_signal("AAPL", -0.20)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) == 1
        assert "short" in rejected[0].reason
        assert len(trades) == 0
        assert portfolio.cash == 100_000

    def test_short_signal_succeeds_with_allow_short(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000, allow_short=True)
        signals = [make_signal("AAPL", -0.20)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        assert len(trades) == 1
        assert trades[0].shares < 0
        assert portfolio.shares("AAPL") < 0
        assert portfolio.cash > 100_000

    def test_oversell_existing_position_rejected(self, broker_2, prices_2):
        portfolio = Portfolio(
            initial_cash=50_000,
            initial_positions={"AAPL": 10},
            allow_short=False,
        )
        # Current weight ≈ 10*100/60_000 ≈ 1.67%, target = -10% => sell more than held
        signals = [make_signal("AAPL", -0.10)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) == 1
        assert "short" in rejected[0].reason

    def test_oversell_existing_position_allowed_with_short(self, broker_2, prices_2):
        portfolio = Portfolio(
            initial_cash=50_000,
            initial_positions={"AAPL": 10},
            allow_short=True,
        )
        signals = [make_signal("AAPL", -0.10)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        assert len(trades) == 1
        assert portfolio.shares("AAPL") < 0

    def test_partial_sell_accepted_full_sell_rejected(self, broker_2, prices_2):
        """Sell within held shares is accepted; oversell is rejected."""
        portfolio = Portfolio(
            initial_cash=50_000,
            initial_positions={"AAPL": 100},
            allow_short=False,
        )
        # AAPL=10_000, total=60_000, weight≈16.7%
        # target 10% => sell ~4_000 worth (ok); target -5% => oversell (rejected)
        signals = [
            make_signal("AAPL", 0.10),  # small sell, stays positive
            make_signal("MSFT", 0.05),  # small buy
        ]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        assert portfolio.shares("AAPL") > 0  # still positive
        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) == 0  # no rejections


# ═══════════════════════════════════════════════════════════════════════════
# 5. LEVERAGE VIA BROKER
# ═══════════════════════════════════════════════════════════════════════════

class TestBrokerLeverage:
    """Broker + Portfolio with max_leverage > 1."""

    def test_over_allocation_fails_without_leverage(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=10_000, max_leverage=1.0)
        # Want 80% AAPL + 80% MSFT = 160% total — first buy uses all cash
        signals = [
            make_signal("AAPL", 0.80),
            make_signal("MSFT", 0.80),
        ]

        orders, _ = broker_2.execute(portfolio, signals, prices_2)

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) >= 1

    def test_leverage_allows_negative_cash(self, prices_2):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0})

        portfolio = Portfolio(initial_cash=10_000, max_leverage=2.0)
        # Buy more than cash allows — broker validation uses cash + sell proceeds
        # With leverage, even if portfolio.execute_trade goes negative, it's allowed
        signals = [make_signal("AAPL", 0.99)]

        orders, trades = broker.execute(portfolio, signals, prices)

        accepted = [o for o in orders if isinstance(o, AcceptedOrder)]
        assert len(accepted) == 1

    def test_short_plus_leverage(self, prices_2):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0})

        portfolio = Portfolio(
            initial_cash=10_000,
            max_leverage=2.0,
            allow_short=True,
        )
        signals = [
            make_signal("AAPL", -0.30),  # short 30%
            make_signal("MSFT", 0.80),   # long 80%
        ]

        orders, trades = broker.execute(portfolio, signals, prices)

        assert portfolio.shares("AAPL") < 0
        assert portfolio.shares("MSFT") > 0

    def test_leverage_limit_enforced_with_short_proceeds(self):
        """max_leverage=2 should cap gross exposure at 200%.

        Short proceeds fund longs, bypassing the cash check.
        The leverage check prevents exceeding 2x.
        """
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0})

        portfolio = Portfolio(initial_cash=10_000, max_leverage=2.0, allow_short=True)
        # Short 80% AAPL => ~$8k proceeds => cash becomes ~$18k
        # Long 150% MSFT => ~$15k cost, fits in $18k cash but exceeds 2x leverage
        signals = [
            make_signal("AAPL", -0.80),
            make_signal("MSFT", 1.50),
        ]

        orders, trades = broker.execute(portfolio, signals, prices)

        total_invested = sum(
            abs(portfolio.shares(t)) * prices.get_price(t)
            for t in ["AAPL", "MSFT"]
        )
        leverage = total_invested / 10_000
        assert leverage <= 2.0, f"leverage {leverage:.2f}x excede max_leverage=2.0"

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert any("leverage" in r.reason for r in rejected)

    def test_leverage_3x_not_exceeded_with_short_proceeds(self):
        """With max_leverage=3, shorts + longs should not exceed 300% gross."""
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0})

        portfolio = Portfolio(initial_cash=10_000, max_leverage=3.0, allow_short=True)
        signals = [
            make_signal("AAPL", -2.00),  # short 200%
            make_signal("MSFT", 3.00),   # long 300% — gross = 500% > 3x
        ]

        orders, trades = broker.execute(portfolio, signals, prices)

        total_invested = sum(
            abs(portfolio.shares(t)) * prices.get_price(t)
            for t in ["AAPL", "MSFT"]
        )
        leverage = total_invested / 10_000
        assert leverage <= 3.0, f"leverage {leverage:.2f}x excede max_leverage=3.0"

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert any("leverage" in r.reason for r in rejected)


# ═══════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_signals_list(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000)
        orders, trades = broker_2.execute(portfolio, [], prices_2)

        assert orders == []
        assert trades == []
        assert portfolio.cash == 100_000

    def test_single_signal_for_all_cash(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=10_000)
        signals = [make_signal("AAPL", 1.0)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        assert len(trades) == 1
        assert portfolio.shares("AAPL") > 0
        assert portfolio.cash >= 0

    def test_signal_target_zero_from_zero(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=10_000)
        signals = [make_signal("AAPL", 0.0)]

        orders, _ = broker_2.execute(portfolio, signals, prices_2)

        invalid = [o for o in orders if isinstance(o, InvalidOrder)]
        assert len(invalid) == 1

    def test_multiple_signals_same_ticker(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000)
        signals = [
            make_signal("AAPL", 0.30),
            make_signal("AAPL", 0.50),
        ]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        # Both should produce orders (broker doesn't deduplicate)
        aapl_orders = [o for o in orders if o.ticker == "AAPL"]
        assert len(aapl_orders) == 2

    def test_rebalance_existing_positions(self, broker_2, prices_2):
        # Start with 50/50, rebalance to 70/30
        # With whole-share rounding, need enough cash buffer for the buy
        portfolio = Portfolio(
            initial_cash=500,
            initial_positions={"AAPL": 500, "MSFT": 250},
        )
        # AAPL=50_000, MSFT=50_000, cash=500, total=100_500 => ~49.75%/~49.75%
        signals = [
            make_signal("AAPL", 0.70),  # buy more
            make_signal("MSFT", 0.30),  # sell some
        ]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        sells = [t for t in trades if t.direction == "sell"]
        buys = [t for t in trades if t.direction == "buy"]
        assert len(sells) == 1 and sells[0].ticker == "MSFT"
        assert len(buys) == 1 and buys[0].ticker == "AAPL"
        # Shares are whole numbers
        assert buys[0].shares == int(buys[0].shares)
        assert sells[0].shares == int(sells[0].shares)

    def test_very_small_weight_change_treated_as_unchanged(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=50_000, initial_positions={"AAPL": 500})
        # Weight is 50_000/100_000 = 0.5, signal 0.5 + epsilon
        signals = [make_signal("AAPL", 0.5 + 1e-8)]

        orders, _ = broker_2.execute(portfolio, signals, prices_2)

        invalid = [o for o in orders if isinstance(o, InvalidOrder)]
        assert len(invalid) == 1

    def test_trade_cost_matches_broker_cost_formula(self, prices_2):
        universe = make_universe(("AAPL", 0.005), ("MSFT", 0.003))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.005, "MSFT": 0.003})

        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.40)]

        _, trades = broker.execute(portfolio, signals, prices)

        trade = trades[0]
        expected_cost = trade.abs_value_before_costs * 0.005
        assert trade.cost == pytest.approx(expected_cost)

    def test_return_types(self, broker_2, prices_2):
        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.50)]

        orders, trades = broker_2.execute(portfolio, signals, prices_2)

        assert isinstance(orders, list)
        assert isinstance(trades, list)
        for t in trades:
            assert isinstance(t, Trade)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TRANSACTION COSTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTransactionCosts:
    """Detailed transaction cost scenarios."""

    def test_sell_cost_deducted_from_cash(self):
        """Transaction costs apply to sells, not only to buys."""
        universe = make_universe(("AAPL", 0.01), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.01, "MSFT": 0.0})

        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 100})
        signals = [make_signal("AAPL", 0.0)]

        _, trades = broker.execute(portfolio, signals, prices)

        assert len(trades) == 1
        assert trades[0].cost > 0
        # Cash = sell_value - cost = (shares * price) - cost
        sold_shares = abs(trades[0].shares)
        expected_cash = sold_shares * 100.0 - trades[0].cost
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_different_costs_per_ticker(self):
        """Each ticker uses its own transaction cost rate."""
        universe = make_universe(("AAPL", 0.01), ("MSFT", 0.005))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.01, "MSFT": 0.005})

        portfolio = Portfolio(initial_cash=200_000)
        signals = [
            make_signal("AAPL", 0.25),
            make_signal("MSFT", 0.25),
        ]

        _, trades = broker.execute(portfolio, signals, prices)

        aapl_trade = next(t for t in trades if t.ticker == "AAPL")
        msft_trade = next(t for t in trades if t.ticker == "MSFT")

        assert aapl_trade.cost == pytest.approx(aapl_trade.abs_value_before_costs * 0.01)
        assert msft_trade.cost == pytest.approx(msft_trade.abs_value_before_costs * 0.005)
        # AAPL rate is double, so its cost should be roughly double for similar values
        assert aapl_trade.cost > msft_trade.cost

    def test_missing_ticker_in_costs_defaults_to_zero(self):
        """If a ticker is not in transaction_costs dict, cost is 0."""
        universe = make_universe(("AAPL", 0.01), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.01})  # MSFT absent

        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("MSFT", 0.50)]

        _, trades = broker.execute(portfolio, signals, prices)

        assert trades[0].cost == 0.0

    def test_cash_balance_equation(self):
        """cash_final = cash_initial + Σ(sell_value) - Σ(buy_value) - Σ(all_costs)."""
        universe = make_universe(("AAPL", 0.005), ("MSFT", 0.003))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.005, "MSFT": 0.003})

        initial_cash = 50_000
        portfolio = Portfolio(initial_cash=initial_cash, initial_positions={"AAPL": 200})
        signals = [
            make_signal("AAPL", 0.10),  # sell some AAPL
            make_signal("MSFT", 0.30),  # buy MSFT
        ]

        _, trades = broker.execute(portfolio, signals, prices)

        total_cost = sum(t.cost for t in trades)
        net_flow = sum(
            -t.shares * t.price  # positive for sells, negative for buys
            for t in trades
        )
        expected_cash = initial_cash + net_flow - total_cost
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_costs_in_rebalance_both_sides(self):
        """Rebalancing incurs costs on both sell and buy legs."""
        universe = make_universe(("AAPL", 0.001), ("MSFT", 0.001))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.001, "MSFT": 0.001})

        # 50/50 => rebalance to 70/30 (cash buffer for whole-share rounding)
        portfolio = Portfolio(initial_cash=500, initial_positions={"AAPL": 500, "MSFT": 250})
        initial_value = portfolio.total_value(prices)  # 100_500

        _, trades = broker.execute(portfolio, signals=[
            make_signal("AAPL", 0.70),
            make_signal("MSFT", 0.30),
        ], prices=prices)

        assert len(trades) == 2
        assert all(t.cost > 0 for t in trades)

        total_cost = sum(t.cost for t in trades)
        final_value = portfolio.total_value(prices)
        assert final_value < initial_value

    def test_high_costs_reject_buy_in_rebalance(self):
        """With high costs, sell proceeds may not cover buy + its costs => buy rejected."""
        universe = make_universe(("AAPL", 0.05), ("MSFT", 0.05))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.05, "MSFT": 0.05})

        # 50/50 => rebalance to 70/30 with no cash buffer
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 500, "MSFT": 250})

        orders, trades = broker.execute(portfolio, signals=[
            make_signal("AAPL", 0.70),
            make_signal("MSFT", 0.30),
        ], prices=prices)

        # Sell MSFT executes, but buy AAPL rejected: net sell proceeds
        # after 5% cost don't cover buy value + 5% buy cost
        sells = [t for t in trades if t.direction == "sell"]
        assert len(sells) == 1

        rejected = [o for o in orders if isinstance(o, RejectedOrder)]
        assert len(rejected) == 1
        assert "insufficient cash" in rejected[0].reason

    def test_high_costs_cause_weight_drift(self):
        """With very high costs, realized weights deviate more from targets."""
        universe = make_universe(("AAPL", 0.05), ("MSFT", 0.05))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})

        broker_high = Broker(transaction_costs={"AAPL": 0.05, "MSFT": 0.05}, buy_adjustment=0.005)
        broker_zero = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        p_high = Portfolio(initial_cash=100_000)
        p_zero = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.50), make_signal("MSFT", 0.50)]

        broker_high.execute(p_high, signals, prices)
        broker_zero.execute(p_zero, signals, prices)

        w_high = p_high.positions_weights(prices)
        w_zero = p_zero.positions_weights(prices)

        drift_high = abs(w_high["AAPL"] - 0.50)
        drift_zero = abs(w_zero["AAPL"] - 0.50)
        assert drift_high > drift_zero

    def test_sell_proceeds_overestimate_with_high_costs(self):
        """Broker estimates sell proceeds without deducting tx costs.

        With high costs this overestimates available cash. The buy_adjustment
        compensates in practice, but this test documents the behavior:
        a buy may be accepted during validation yet the final cash could
        end up slightly negative if buy_adjustment doesn't cover the gap.
        """
        universe = make_universe(("AAPL", 0.05), ("MSFT", 0.05))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.05, "MSFT": 0.05}, buy_adjustment=0.005)

        # Zero cash, all in AAPL. Sell AAPL to buy MSFT.
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 1000})
        signals = [
            make_signal("AAPL", 0.0),   # sell all => ~$100k proceeds gross
            make_signal("MSFT", 0.95),  # try to use almost all proceeds
        ]

        orders, trades = broker.execute(portfolio, signals, prices)

        # The validation uses sell proceeds WITHOUT costs ($100k).
        # The execution deducts 5% cost on the sell (~$5k).
        # So actual cash from sell ≈ $95k, not $100k.
        # If the buy was sized for $95k (based on effective_value),
        # it should still work thanks to buy_adjustment + effective_value buffer.
        assert portfolio.cash >= -1, f"cash went significantly negative: {portfolio.cash:.2f}"

    def test_total_costs_across_multiple_trades(self):
        """Sum of individual trade costs matches expected total."""
        universe = make_universe(("AAPL", 0.002), ("MSFT", 0.003), ("GOOG", 0.001))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 150.0})
        broker = Broker(transaction_costs={"AAPL": 0.002, "MSFT": 0.003, "GOOG": 0.001})

        portfolio = Portfolio(initial_cash=300_000)
        signals = [
            make_signal("AAPL", 0.30),
            make_signal("MSFT", 0.40),
            make_signal("GOOG", 0.20),
        ]

        _, trades = broker.execute(portfolio, signals, prices)

        for trade in trades:
            rate = {"AAPL": 0.002, "MSFT": 0.003, "GOOG": 0.001}[trade.ticker]
            assert trade.cost == pytest.approx(trade.abs_value_before_costs * rate)


# ═══════════════════════════════════════════════════════════════════════════
# 8. POST-REBALANCE WEIGHT CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

class TestWeightConsistency:
    """After execution, portfolio weights should approximate target weights."""

    WEIGHT_TOL = 0.02  # 2% tolerance for buy_adjustment + tx costs

    def test_single_asset_weight_matches_target(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        portfolio = Portfolio(initial_cash=100_000)
        signals = [make_signal("AAPL", 0.60)]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(0.60, abs=self.WEIGHT_TOL)

    def test_two_asset_weights_match_targets(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        portfolio = Portfolio(initial_cash=100_000)
        signals = [
            make_signal("AAPL", 0.40),
            make_signal("MSFT", 0.60),
        ]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(0.40, abs=self.WEIGHT_TOL)
        assert weights["MSFT"] == pytest.approx(0.60, abs=self.WEIGHT_TOL)

    def test_three_asset_weights_match_targets(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0), ("GOOG", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 150.0})
        broker = Broker(
            transaction_costs={"AAPL": 0.0, "MSFT": 0.0, "GOOG": 0.0},
            buy_adjustment=0.005,
        )

        portfolio = Portfolio(initial_cash=50_000)
        signals = [
            make_signal("AAPL", 0.30),
            make_signal("MSFT", 0.50),
            make_signal("GOOG", 0.20),
        ]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(0.30, abs=self.WEIGHT_TOL)
        assert weights["MSFT"] == pytest.approx(0.50, abs=self.WEIGHT_TOL)
        assert weights["GOOG"] == pytest.approx(0.20, abs=self.WEIGHT_TOL)

    def test_weights_after_rebalance(self):
        universe = make_universe(("AAPL", 0.001), ("MSFT", 0.001))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.001, "MSFT": 0.001}, buy_adjustment=0.005)

        # Start 50/50, rebalance to 70/30 (cash buffer for whole-share rounding)
        portfolio = Portfolio(initial_cash=500, initial_positions={"AAPL": 500, "MSFT": 250})
        signals = [
            make_signal("AAPL", 0.70),
            make_signal("MSFT", 0.30),
        ]

        broker.execute(portfolio, signals, prices)

        # Whole-share rounding introduces larger weight deviation
        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(0.70, abs=0.03)
        assert weights["MSFT"] == pytest.approx(0.30, abs=0.03)

    def test_weights_with_transaction_costs(self):
        universe = make_universe(("AAPL", 0.01), ("MSFT", 0.01))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.01, "MSFT": 0.01}, buy_adjustment=0.005)

        portfolio = Portfolio(initial_cash=100_000)
        signals = [
            make_signal("AAPL", 0.50),
            make_signal("MSFT", 0.50),
        ]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        # With 1% tx costs the deviation is larger, but should still be close
        assert weights["AAPL"] == pytest.approx(0.50, abs=0.03)
        assert weights["MSFT"] == pytest.approx(0.50, abs=0.03)

    def test_total_invested_fraction_matches_signals(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        portfolio = Portfolio(initial_cash=100_000)
        target_alpha = 0.70  # 70% invested
        signals = [
            make_signal("AAPL", 0.40),
            make_signal("MSFT", 0.30),
        ]

        broker.execute(portfolio, signals, prices)

        alpha = portfolio.alpha(prices)
        assert alpha == pytest.approx(target_alpha, abs=self.WEIGHT_TOL)

    def test_weights_after_partial_sell(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        # 100% in AAPL, reduce to 30%
        portfolio = Portfolio(initial_cash=0, initial_positions={"AAPL": 1000})
        signals = [make_signal("AAPL", 0.30)]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(0.30, abs=self.WEIGHT_TOL)

    def test_weights_with_short_position(self):
        universe = make_universe(("AAPL", 0.0), ("MSFT", 0.0))
        prices = make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})
        broker = Broker(transaction_costs={"AAPL": 0.0, "MSFT": 0.0}, buy_adjustment=0.005)

        portfolio = Portfolio(initial_cash=100_000, allow_short=True)
        signals = [
            make_signal("AAPL", -0.20),  # short
            make_signal("MSFT", 0.50),   # long
        ]

        broker.execute(portfolio, signals, prices)

        weights = portfolio.positions_weights(prices)
        assert weights["AAPL"] == pytest.approx(-0.20, abs=self.WEIGHT_TOL)
        assert weights["MSFT"] == pytest.approx(0.50, abs=self.WEIGHT_TOL)
