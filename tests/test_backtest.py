"""Tests for Backtest, WalkForwardBacktest, and RollingBacktest engines."""

import pytest
import pandas as pd

from src.domain.portfolio import Portfolio
from src.backtesting import Backtest, WalkForwardBacktest, RollingBacktest
from src.backtesting.result import BacktestResult, StrategyResult, MultipleResult
from tests.conftest import make_universe, make_price_history, FixedWeightStrategy


# ── Shared helpers ──────────────────────────────────────────────────────────

@pytest.fixture
def universe():
    return make_universe(("AAPL", 0.0), ("MSFT", 0.0))


@pytest.fixture
def history_500(universe):
    return make_price_history(universe, 500)


@pytest.fixture
def history_100(universe):
    return make_price_history(universe, 100)


def make_strategy(universe, name="test", weights=None):
    return FixedWeightStrategy(
        universe=universe,
        name=name,
        weights=weights or {"AAPL": 0.6, "MSFT": 0.4},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktest:

    def test_returns_multiple_result(self, universe, history_500):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=50,
        )
        result = engine.run()
        assert isinstance(result, MultipleResult)

    def test_strategy_result_has_one_window(self, universe, history_500):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=50,
        )
        sr = engine.run().result("test")
        assert isinstance(sr, StrategyResult)
        assert sr.n_windows == 1

    def test_simulation_days_match_history_minus_warmup(self, universe, history_500):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=100,
        )
        result = engine.run().result("test").combined
        assert len(result.snapshots) == 400

    def test_warmup_zero(self, universe, history_100):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_100,
            warmup_size=0,
        )
        result = engine.run().result("test").combined
        assert len(result.snapshots) == 100

    def test_warmup_with_date(self, universe, history_500):
        dates = history_500.dates
        split_date = dates[200]
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=str(split_date.date()),
        )
        result = engine.run().result("test").combined
        assert result.dates[0] >= split_date

    def test_warmup_with_timestamp(self, universe, history_500):
        split_date = history_500.dates[200]
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=split_date,
        )
        result = engine.run().result("test").combined
        assert result.dates[0] == split_date

    def test_portfolio_changes_during_simulation(self, universe, history_100):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_100,
            warmup_size=0,
        )
        result = engine.run().result("test").combined
        assert result.total_trades > 0
        assert result.final_value != result.initial_value

    def test_multiple_strategies_independent(self, universe, history_500):
        strat_a = make_strategy(universe, name="all_aapl", weights={"AAPL": 1.0})
        strat_b = make_strategy(universe, name="all_msft", weights={"MSFT": 1.0})
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat_a, strat_b],
            history=history_500,
            warmup_size=50,
        )
        result = engine.run()
        a = result.result("all_aapl").combined
        b = result.result("all_msft").combined
        # Same number of simulation days
        assert len(a.snapshots) == len(b.snapshots)
        # Different final values (different allocations)
        assert a.final_value != b.final_value

    def test_duplicate_strategy_names_rejected(self, universe, history_100):
        strat_a = make_strategy(universe, name="same")
        strat_b = make_strategy(universe, name="same")
        with pytest.raises(ValueError, match="multiple strategies"):
            Backtest(
                portfolio=Portfolio(initial_cash=100_000),
                universe=universe,
                strategies=[strat_a, strat_b],
                history=history_100,
                warmup_size=0,
            )

    def test_strategy_reset_between_runs(self, universe, history_100):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_100,
            warmup_size=10,
        )
        r1 = engine.run().result("test").combined.final_value
        r2 = engine.run().result("test").combined.final_value
        assert r1 == pytest.approx(r2)


# ═══════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

class TestWalkForwardBacktest:

    def test_strategy_history_trimmed(self, universe, history_500):
        strat = make_strategy(universe)
        window_size = 50
        engine = WalkForwardBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=100,
            window_size=window_size,
        )
        engine.run()
        # After simulation, strategy history should be at most window_size
        assert len(strat.history) <= window_size

    def test_history_never_exceeds_window_during_sim(self, universe, history_500):
        """After the first trim, history stays bounded at window_size + 1."""
        window_size = 60
        sizes_seen = []

        class TrackingStrategy(FixedWeightStrategy):
            def _generate_signals(self, prices, portfolio):
                sizes_seen.append(len(self._history))
                return super()._generate_signals(prices, portfolio)

        strat = TrackingStrategy(
            universe=universe, name="track",
            weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        WalkForwardBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=window_size,  # start at window_size so trim is effective immediately
            window_size=window_size,
        ).run()

        # on_step appends before _on_after_step trims, so max is window_size + 1
        assert max(sizes_seen) <= window_size + 1

    def test_same_sim_days_as_backtest(self, universe, history_500):
        strat_bt = make_strategy(universe, name="bt")
        strat_wf = make_strategy(universe, name="wf")
        warmup = 100

        r_bt = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat_bt],
            history=history_500,
            warmup_size=warmup,
        ).run().result("bt").combined

        r_wf = WalkForwardBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat_wf],
            history=history_500,
            warmup_size=warmup,
            window_size=100,
        ).run().result("wf").combined

        assert len(r_bt.snapshots) == len(r_wf.snapshots)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROLLING BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

class TestRollingBacktest:

    def test_multiple_windows(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        assert sr.n_windows > 1

    def test_last_window_ends_on_last_date(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        last_window = sr.windows[-1]
        assert last_window.dates[-1] == history_500.dates[-1]

    def test_windows_are_chronological(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        for i in range(1, sr.n_windows):
            assert sr.windows[i].dates[0] > sr.windows[i - 1].dates[0]

    def test_non_overlapping_test_periods(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
            step_size=50,
        )
        sr = engine.run().result("test")
        for i in range(1, sr.n_windows):
            prev_end = sr.windows[i - 1].dates[-1]
            curr_start = sr.windows[i].dates[0]
            assert curr_start > prev_end

    def test_portfolio_carries_over(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        # initial_snapshot of window N+1 captures portfolio state
        # BEFORE trading on the first day — same positions/cash as
        # end of window N (value differs because prices changed).
        for i in range(sr.n_windows - 1):
            end_snap = sr.windows[i].snapshots[-1]
            next_initial = sr.windows[i + 1].initial_snapshot
            assert end_snap.cash == pytest.approx(next_initial.cash)
            assert end_snap.positions == next_initial.positions

    def test_step_size_defaults_to_test_size(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        assert engine.step_size == 50

    def test_history_too_short_raises(self, universe, history_100):
        strat = make_strategy(universe)
        with pytest.raises(ValueError, match="too short"):
            RollingBacktest(
                portfolio=Portfolio(initial_cash=100_000),
                universe=universe,
                strategies=[strat],
                history=history_100,
                train_size=80,
                test_size=30,
            ).run()

    def test_strategy_reinitialized_per_window(self, universe, history_500):
        """Each window produces its own BacktestResult; totals match combined."""
        strat = make_strategy(universe, weights={"AAPL": 0.6, "MSFT": 0.4})
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        total_sim_days = sum(len(w.snapshots) for w in sr.windows)
        assert total_sim_days == len(sr.combined.snapshots)

    def test_overlapping_test_periods(self, universe, history_500):
        """step_size < test_size gives overlapping test windows."""
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
            step_size=25,
        )
        sr = engine.run().result("test")
        # With step=25 and test=50, windows overlap by 25 days
        assert sr.n_windows > 1
        for i in range(1, sr.n_windows):
            prev_end = sr.windows[i - 1].dates[-1]
            curr_start = sr.windows[i].dates[0]
            assert curr_start <= prev_end  # overlapping

    def test_strategy_step_count_resets_per_window(self, universe, history_500):
        """strategy.reset() is called each window, so step_count restarts."""
        strat = make_strategy(universe, weights={"AAPL": 0.6, "MSFT": 0.4})
        strat.rebalance_every = 5
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        # Each window should have signals on the same relative days
        # (every 5th step). If reset didn't work, the count would
        # accumulate and signal frequency would drift.
        for w in sr.windows:
            days_with_signals = sum(1 for s in w.signals if len(s) > 0)
            expected = len(w.snapshots) // 5
            assert days_with_signals == pytest.approx(expected, abs=1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. BACKTEST RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestResult:

    @pytest.fixture
    def result(self, universe, history_500):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=50,
        )
        return engine.run().result("test").combined

    def test_metrics_computed(self, result):
        assert "sharpe_ratio" in result.metrics
        assert "annualized_return" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "annualized_volatility" in result.metrics
        assert "calmar_ratio" in result.metrics

    def test_initial_value(self, result):
        assert result.initial_value == 100_000

    def test_value_over_time_length(self, result):
        assert len(result.value_over_time) == len(result.snapshots)

    def test_returns_one_shorter(self, result):
        assert len(result.returns_over_time) == len(result.snapshots) - 1

    def test_drawdown_nonpositive(self, result):
        assert (result.drawdown <= 0).all()

    def test_final_return_consistent(self, result):
        expected = result.final_value / result.initial_value - 1
        assert result.final_return == pytest.approx(expected)

    def test_total_trades_positive(self, result):
        assert result.total_trades > 0

    def test_total_orders_ge_trades(self, result):
        assert result.total_orders >= result.total_trades

    def test_order_counts_sum(self, result):
        assert (result.accepted_orders + result.rejected_orders + result.invalid_orders
                == result.total_orders)

    def test_dates_index(self, result):
        assert isinstance(result.dates, pd.DatetimeIndex)
        assert len(result.dates) == len(result.snapshots)

    def test_weights_df_columns(self, result):
        wdf = result.weights_df
        assert "AAPL" in wdf.columns or "MSFT" in wdf.columns

    def test_summary_is_series(self, result):
        s = result.summary
        assert isinstance(s, pd.Series)
        assert "sharpe_ratio" in s.index
        assert "total_trades" in s.index

    def test_trades_df(self, result):
        df = result.trades_df
        assert "ticker" in df.columns
        assert "shares" in df.columns
        assert len(df) == result.total_trades

    def test_target_weights_df(self, result):
        df = result.target_weights_df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(result.snapshots)

    def test_values_df_columns(self, result):
        df = result.values_df
        assert "cash" in df.columns
        assert "total_value" in df.columns

    def test_total_costs_with_nonzero_costs(self, universe, history_500):
        universe_costs = make_universe(("AAPL", 0.01), ("MSFT", 0.01))
        history = make_price_history(universe_costs, 500)
        strat = FixedWeightStrategy(
            universe=universe_costs, name="costly",
            weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        result = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe_costs,
            strategies=[strat],
            history=history,
            warmup_size=50,
        ).run().result("costly").combined
        assert result.total_costs > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. STRATEGY RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyResult:

    def test_single_window_combined_is_same(self, universe, history_500):
        strat = make_strategy(universe)
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            warmup_size=50,
        )
        sr = engine.run().result("test")
        assert sr.combined is sr.windows[0]

    def test_multi_window_combined_concatenates(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        total_days = sum(len(w.snapshots) for w in sr.windows)
        assert len(sr.combined.snapshots) == total_days

    def test_combined_metrics_exist(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        assert "sharpe_ratio" in sr.metrics

    def test_window_access(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        for i in range(sr.n_windows):
            w = sr.window(i)
            assert isinstance(w, BacktestResult)
            assert len(w.snapshots) > 0

    def test_combined_initial_value_from_first_window(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        assert sr.combined.initial_value == sr.windows[0].initial_value

    def test_combined_final_value_from_last_window(self, universe, history_500):
        strat = make_strategy(universe)
        engine = RollingBacktest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat],
            history=history_500,
            train_size=100,
            test_size=50,
        )
        sr = engine.run().result("test")
        assert sr.combined.final_value == pytest.approx(sr.windows[-1].final_value)


# ═══════════════════════════════════════════════════════════════════════════
# 6. MULTIPLE RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestMultipleResult:

    @pytest.fixture
    def multi(self, universe, history_500):
        strat_a = make_strategy(universe, name="A", weights={"AAPL": 0.8, "MSFT": 0.2})
        strat_b = make_strategy(universe, name="B", weights={"AAPL": 0.2, "MSFT": 0.8})
        engine = Backtest(
            portfolio=Portfolio(initial_cash=100_000),
            universe=universe,
            strategies=[strat_a, strat_b],
            history=history_500,
            warmup_size=50,
        )
        return engine.run()

    def test_strategies_list(self, multi):
        assert set(multi.strategies) == {"A", "B"}

    def test_result_access(self, multi):
        assert isinstance(multi.result("A"), StrategyResult)
        assert isinstance(multi.result("B"), StrategyResult)

    def test_missing_strategy_raises(self, multi):
        with pytest.raises(ValueError):
            multi.result("nonexistent")

    def test_summary_df(self, multi):
        df = multi.summary_df
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"A", "B"}
        assert "sharpe_ratio" in df.columns

    def test_value_df(self, multi):
        df = multi.value_df
        assert "A" in df.columns and "B" in df.columns
        assert len(df) == 450  # 500 - 50 warmup

    def test_returns_df(self, multi):
        df = multi.returns_df
        assert "A" in df.columns and "B" in df.columns

    def test_cumulative_returns_df(self, multi):
        df = multi.cumulative_returns_df
        assert "A" in df.columns and "B" in df.columns

    def test_drawdown_df(self, multi):
        df = multi.drawdown_df
        assert "A" in df.columns and "B" in df.columns
        assert (df <= 0).all().all()

    def test_strategies_start_from_same_state(self, multi):
        a = multi.result("A").combined
        b = multi.result("B").combined
        assert a.initial_value == b.initial_value == 100_000
