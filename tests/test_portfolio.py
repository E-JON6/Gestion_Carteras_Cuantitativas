"""Tests for Portfolio: state management, short-selling and leverage constraints."""

import pytest

from src.domain.portfolio import Portfolio, PortfolioError
from tests.conftest import make_universe, make_prices


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def universe():
    return make_universe(("AAPL", 0.001), ("MSFT", 0.001))


@pytest.fixture
def prices(universe):
    return make_prices(universe, {"AAPL": 100.0, "MSFT": 200.0})


# ── Initial state ───────────────────────────────────────────────────────────

class TestInitialState:
    def test_cash_equals_initial(self):
        p = Portfolio(initial_cash=50_000)
        assert p.cash == 50_000

    def test_empty_positions(self):
        p = Portfolio(initial_cash=10_000)
        assert p.positions == {}

    def test_initial_positions_copied(self):
        p = Portfolio(initial_cash=0, initial_positions={"AAPL": 10})
        assert p.shares("AAPL") == 10

    def test_total_value_cash_only(self, universe, prices):
        p = Portfolio(initial_cash=10_000)
        assert p.total_value(prices) == 10_000

    def test_total_value_with_positions(self, prices):
        p = Portfolio(initial_cash=5_000, initial_positions={"AAPL": 10})
        # 5000 + 10*100 = 6000
        assert p.total_value(prices) == 6_000

    def test_alpha_zero_when_cash_only(self, prices):
        p = Portfolio(initial_cash=10_000)
        assert p.alpha(prices) == 0.0

    def test_alpha_with_positions(self, prices):
        p = Portfolio(initial_cash=0, initial_positions={"AAPL": 10})
        assert p.alpha(prices) == 1.0

    def test_weights(self, prices):
        # AAPL=10*100=1000, MSFT=5*200=1000, cash=0 => 50/50
        p = Portfolio(initial_cash=0, initial_positions={"AAPL": 10, "MSFT": 5})
        w = p.positions_weights(prices)
        assert w["AAPL"] == pytest.approx(0.5)
        assert w["MSFT"] == pytest.approx(0.5)


# ── execute_trade: basic ────────────────────────────────────────────────────

class TestExecuteTradeBasic:
    def test_buy_updates_shares_and_cash(self):
        p = Portfolio(initial_cash=10_000)
        p.execute_trade("AAPL", shares_delta=5, cash_delta=-500)
        assert p.shares("AAPL") == 5
        assert p.cash == 9_500

    def test_sell_updates_shares_and_cash(self):
        p = Portfolio(initial_cash=1_000, initial_positions={"AAPL": 10})
        p.execute_trade("AAPL", shares_delta=-3, cash_delta=300)
        assert p.shares("AAPL") == 7
        assert p.cash == 1_300

    def test_sell_all_shares(self):
        p = Portfolio(initial_cash=0, initial_positions={"AAPL": 10})
        p.execute_trade("AAPL", shares_delta=-10, cash_delta=1_000)
        assert p.shares("AAPL") == 0


# ── execute_trade: short selling ────────────────────────────────────────────

class TestShortSelling:
    def test_short_blocked_by_default(self):
        p = Portfolio(initial_cash=10_000)
        with pytest.raises(PortfolioError):
            p.execute_trade("AAPL", shares_delta=-5, cash_delta=500)

    def test_short_blocked_oversell(self):
        p = Portfolio(initial_cash=10_000, initial_positions={"AAPL": 3})
        with pytest.raises(PortfolioError):
            p.execute_trade("AAPL", shares_delta=-5, cash_delta=500)

    def test_short_allowed_when_enabled(self):
        p = Portfolio(initial_cash=10_000, allow_short=True)
        p.execute_trade("AAPL", shares_delta=-5, cash_delta=500)
        assert p.shares("AAPL") == -5
        assert p.cash == 10_500

    def test_short_oversell_allowed(self):
        p = Portfolio(initial_cash=10_000, initial_positions={"AAPL": 3}, allow_short=True)
        p.execute_trade("AAPL", shares_delta=-10, cash_delta=1_000)
        assert p.shares("AAPL") == -7


# ── execute_trade: leverage / negative cash ─────────────────────────────────

class TestLeverage:
    def test_negative_cash_blocked_by_default(self):
        p = Portfolio(initial_cash=100)
        with pytest.raises(PortfolioError):
            p.execute_trade("AAPL", shares_delta=10, cash_delta=-200)

    def test_negative_cash_blocked_at_leverage_1(self):
        p = Portfolio(initial_cash=100, max_leverage=1.0)
        with pytest.raises(PortfolioError):
            p.execute_trade("AAPL", shares_delta=10, cash_delta=-200)

    def test_negative_cash_allowed_with_leverage(self):
        p = Portfolio(initial_cash=100, max_leverage=2.0)
        p.execute_trade("AAPL", shares_delta=10, cash_delta=-200)
        assert p.cash == -100
        assert p.shares("AAPL") == 10

    def test_leverage_with_short_combined(self):
        p = Portfolio(initial_cash=100, max_leverage=2.0, allow_short=True)
        p.execute_trade("AAPL", shares_delta=-5, cash_delta=500)
        assert p.shares("AAPL") == -5
        assert p.cash == 600


# ── reset ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_restores_cash(self):
        p = Portfolio(initial_cash=10_000)
        p.execute_trade("AAPL", shares_delta=5, cash_delta=-500)
        p.reset()
        assert p.cash == 10_000

    def test_reset_clears_positions(self):
        p = Portfolio(initial_cash=10_000)
        p.execute_trade("AAPL", shares_delta=5, cash_delta=-500)
        p.reset()
        assert p.positions == {}

    def test_reset_restores_initial_positions(self):
        p = Portfolio(initial_cash=10_000, initial_positions={"AAPL": 10})
        p.execute_trade("AAPL", shares_delta=-10, cash_delta=1_000)
        p.reset()
        assert p.shares("AAPL") == 10
        assert p.cash == 10_000
