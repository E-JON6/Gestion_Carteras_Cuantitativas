"""Tests for MinWeightFilter, OperativaExporter, PortfolioStateManager, and VLTracker."""

import json
import tempfile

import pytest
import pandas as pd

from src.domain.trade import Signal, Trade
from src.domain.trade.order import AcceptedOrder
from src.strategy.filter.min_weight import MinWeightFilter
from src.io.excel_exporter import OperativaExporter
from src.io.portfolio_state import PortfolioStateManager
from src.io.vl_tracker import VLTracker
from tests.conftest import make_universe, make_prices, make_signal, DATE


# ── MinWeightFilter ────────────────────────────────────────────────────────

class TestMinWeightFilter:

    def test_filters_below_threshold(self):
        f = MinWeightFilter(min_weight=0.01)
        signals = [
            make_signal("AAPL", 0.05),
            make_signal("MSFT", 0.005),  # below 1%
            make_signal("GOOG", 0.02),
        ]
        result = f.filter(signals)
        assert len(result) == 2
        assert all(s.target_weight >= 0.01 for s in result)

    def test_keeps_liquidation_signals(self):
        f = MinWeightFilter(min_weight=0.01)
        signals = [
            make_signal("AAPL", 0.0),  # liquidation
            make_signal("MSFT", 0.005),  # below threshold
        ]
        result = f.filter(signals)
        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_custom_threshold(self):
        f = MinWeightFilter(min_weight=0.05)
        signals = [
            make_signal("AAPL", 0.03),
            make_signal("MSFT", 0.10),
        ]
        result = f.filter(signals)
        assert len(result) == 1
        assert result[0].ticker == "MSFT"

    def test_empty_signals(self):
        f = MinWeightFilter()
        assert f.filter([]) == []


# ── OperativaExporter ──────────────────────────────────────────────────────

class TestOperativaExporter:

    @pytest.fixture
    def universe(self):
        return make_universe(("XLK", 0.001), ("GLD", 0.002))

    @pytest.fixture
    def exporter(self, universe, tmp_path):
        return OperativaExporter(universe=universe, output_dir=str(tmp_path))

    def _make_trade(self, ticker, shares, price):
        signal = Signal(date=DATE, ticker=ticker, target_weight=0.5, reason="test")
        order = AcceptedOrder(signal=signal, shares=shares, price=price)
        cost = abs(shares) * price * 0.001
        return Trade(order=order, cost=cost)

    def test_export_creates_excel(self, exporter, tmp_path):
        trades = [
            self._make_trade("XLK", 100, 230.0),
            self._make_trade("GLD", -50, 180.0),
        ]
        path = exporter.export(trades, DATE)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".xlsx"

        df = pd.read_excel(path)
        assert list(df.columns) == ["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"]
        assert len(df) == 2

    def test_buy_price_includes_ct(self, exporter, tmp_path):
        trades = [self._make_trade("XLK", 100, 200.0)]
        path = exporter.export(trades, DATE)
        df = pd.read_excel(path)
        row = df.iloc[0]
        assert row["Precio Ejecutado"] == pytest.approx(200.0 * 1.001)

    def test_sell_price_subtracts_ct(self, exporter, tmp_path):
        trades = [self._make_trade("GLD", -50, 180.0)]
        path = exporter.export(trades, DATE)
        df = pd.read_excel(path)
        row = df.iloc[0]
        assert row["Precio Ejecutado"] == pytest.approx(180.0 * (1 - 0.002))

    def test_no_trades_returns_none(self, exporter):
        assert exporter.export([], DATE) is None


# ── PortfolioStateManager ──────────────────────────────────────────────────

class TestPortfolioStateManager:

    @pytest.fixture
    def mgr(self, tmp_path):
        return PortfolioStateManager(state_dir=str(tmp_path))

    @pytest.fixture
    def portfolio(self, universe_2, prices_2):
        p = __import__("src.domain.portfolio", fromlist=["Portfolio"]).Portfolio(
            initial_cash=10_000_000
        )
        return p

    def test_save_and_load(self, mgr, universe_2, prices_2):
        from src.domain.portfolio import Portfolio
        p = Portfolio(initial_cash=10_000_000)
        mgr.save(p, prices_2, DATE)

        loaded = mgr.load()
        assert loaded is not None
        assert loaded["cash"] == 10_000_000
        assert loaded["date"] == DATE.isoformat()

    def test_load_into_portfolio(self, mgr, universe_2, prices_2):
        from src.domain.portfolio import Portfolio
        p = Portfolio(initial_cash=10_000_000)
        p.execute_trade("AAPL", 100, -10_000)
        mgr.save(p, prices_2, DATE)

        p2 = Portfolio(initial_cash=0)
        restored_date = mgr.load_into_portfolio(p2)
        assert restored_date == DATE
        assert p2.cash == p.cash
        assert p2.shares("AAPL") == 100

    def test_value_history(self, mgr, prices_2):
        from src.domain.portfolio import Portfolio
        p = Portfolio(initial_cash=10_000_000)
        d1 = pd.Timestamp("2024-06-01")
        d2 = pd.Timestamp("2024-06-02")
        mgr.save(p, prices_2, d1)
        mgr.save(p, prices_2, d2)

        series = mgr.value_history()
        assert len(series) == 2

    def test_load_empty(self, mgr):
        assert mgr.load() is None


# ── VLTracker ──────────────────────────────────────────────────────────────

class TestVLTracker:

    @pytest.fixture
    def tracker(self, tmp_path):
        return VLTracker(state_dir=str(tmp_path))

    def test_record_and_history(self, tracker):
        tracker.record(pd.Timestamp("2024-01-01"), 10_000_000)
        tracker.record(pd.Timestamp("2024-01-02"), 10_050_000)
        tracker.record(pd.Timestamp("2024-01-03"), 10_020_000)

        df = tracker.history()
        assert len(df) == 3
        assert "return" in df.columns
        assert "cumulative_return" in df.columns
        assert "drawdown" in df.columns

    def test_metrics(self, tracker):
        for i in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            value = 10_000_000 + i * 10_000
            tracker.record(date, value)

        m = tracker.metrics()
        assert "total_return" in m
        assert "sharpe_ratio" in m
        assert m["total_return"] > 0

    def test_export_seguimiento(self, tracker, tmp_path):
        tracker.record(pd.Timestamp("2024-01-01"), 10_000_000)
        tracker.record(pd.Timestamp("2024-01-02"), 10_050_000)

        out = tracker.export_seguimiento(output_path=str(tmp_path / "seg.xlsx"))
        assert out.exists()

    def test_duplicate_date_overwrites(self, tracker):
        tracker.record(pd.Timestamp("2024-01-01"), 10_000_000)
        tracker.record(pd.Timestamp("2024-01-01"), 10_100_000)

        df = tracker.history()
        assert len(df) == 1
        assert df.iloc[0]["total_value"] == 10_100_000
