"""Walk-forward optimization: optimize on train, validate on test, roll forward."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
import logging

import pandas as pd

from src.domain.asset import Universe, PriceHistory
from src.domain.portfolio import Portfolio
from src.strategy.base import Strategy
from src.backtesting.backtest import Backtest
from src.backtesting.result import BacktestResult, StrategyResult
from .optimizer import GridSearchOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowResult:
    """Result of one train/test window."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    best_train_metric: float
    test_result: BacktestResult
    optimization_result: OptimizationResult


@dataclass(frozen=True)
class WalkForwardResult:
    """Combined result of all walk-forward windows."""

    windows: list[WindowResult]

    _strategy_result: StrategyResult = field(init=False, repr=False)

    def __post_init__(self):
        test_results = [w.test_result for w in self.windows]
        object.__setattr__(self, "_strategy_result", StrategyResult(windows=test_results))

    @property
    def combined(self) -> BacktestResult:
        return self._strategy_result.combined

    @property
    def combined_metrics(self) -> dict:
        return self._strategy_result.metrics

    @property
    def params_over_time(self) -> pd.DataFrame:
        records = []
        for w in self.windows:
            record = {
                "window": w.window_index,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_metric": w.best_train_metric,
            }
            record.update(w.best_params)
            records.append(record)
        return pd.DataFrame(records)

    @property
    def oos_is_ratio(self) -> float | None:
        """Ratio of OOS metric to average IS metric. >= 0.6 is robust."""
        if not self.windows:
            return None
        avg_is = sum(w.best_train_metric for w in self.windows) / len(self.windows)
        oos_metric = self.combined_metrics.get("sharpe_ratio", 0.0)
        if avg_is == 0:
            return None
        return oos_metric / avg_is

    def summary(self) -> pd.DataFrame:
        records = []
        for w in self.windows:
            test_metrics = w.test_result.summary
            record = {
                "window": w.window_index,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_metric": w.best_train_metric,
                "test_return": test_metrics.get("total_return", None),
                "test_sharpe": test_metrics.get("sharpe_ratio", None),
            }
            record.update({f"param_{k}": v for k, v in w.best_params.items()})
            records.append(record)
        return pd.DataFrame(records)


@dataclass
class WalkForwardOptimizer:
    """Walk-Forward Optimization: optimize on train, validate on test, roll forward.

    Windows are built backwards from the end of the data so the last
    test window always ends on the last available date.

    Args:
        strategy_cls: Strategy class to instantiate.
        param_grid: Iterable of param dicts (ParameterGrid, RandomParameterGrid, list).
        metric: Metric to optimize.
        maximize: True to maximize, False to minimize.
        train_size: Days of train data per window.
        test_size: Days of test data per window.
        step_size: How far the window slides (default = test_size).
        base_strategy_kwargs: Fixed kwargs for every strategy instance.
    """

    strategy_cls: type[Strategy]
    param_grid: Iterator[dict[str, Any]]
    metric: str = "sharpe_ratio"
    maximize: bool = True
    train_size: int = 504
    test_size: int = 63
    step_size: int | None = None
    base_strategy_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.test_size

    def run(
        self,
        universe: Universe,
        full_history: PriceHistory,
        portfolio_factory: Callable[[], Portfolio],
    ) -> WalkForwardResult:
        """Run walk-forward optimization over the full history."""
        index_windows = self._compute_windows(full_history)

        logger.info(f"Walk-Forward: {len(index_windows)} windows, {len(self.param_grid)} combos each")

        results = []
        for i, (train_idx, test_idx) in enumerate(index_windows):
            logger.info(f"Window {i + 1}/{len(index_windows)}")

            train_history = self._slice(full_history, train_idx)
            test_history = self._slice(full_history, test_idx)

            # Optimize on train
            optimizer = GridSearchOptimizer(
                strategy_cls=self.strategy_cls,
                param_grid=self.param_grid,
                metric=self.metric,
                maximize=self.maximize,
                warmup_size=0,
                base_strategy_kwargs=self.base_strategy_kwargs,
            )
            opt_result = optimizer.optimize(universe, train_history, portfolio_factory)

            # Validate on test (train as warmup)
            test_result = self._backtest_on_test(
                universe, train_history, test_history, opt_result.best_params, portfolio_factory,
            )

            dates = full_history.dates
            results.append(WindowResult(
                window_index=i,
                train_start=dates[train_idx[0]],
                train_end=dates[train_idx[1] - 1],
                test_start=dates[test_idx[0]],
                test_end=dates[test_idx[1] - 1],
                best_params=opt_result.best_params,
                best_train_metric=opt_result.best_metric,
                test_result=test_result,
                optimization_result=opt_result,
            ))

        return WalkForwardResult(windows=results)

    def _compute_windows(
        self, history: PriceHistory,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Compute (train, test) index pairs, built backwards from the end."""
        total = len(history)
        min_required = self.train_size + self.test_size

        if total < min_required:
            raise ValueError(
                f"History too short: {total} days, "
                f"need at least {min_required} (train={self.train_size} + test={self.test_size})"
            )

        raw = []
        end = total
        while end - min_required >= 0:
            test_start = end - self.test_size
            train_start = test_start - self.train_size
            raw.append(((train_start, test_start), (test_start, end)))
            end -= self.step_size

        return list(reversed(raw))

    def _slice(self, history: PriceHistory, s: tuple[int, int]) -> PriceHistory:
        return PriceHistory(history.universe, history.df.iloc[s[0]:s[1]])

    def _backtest_on_test(
        self,
        universe: Universe,
        train_history: PriceHistory,
        test_history: PriceHistory,
        best_params: dict[str, Any],
        portfolio_factory: Callable[[], Portfolio],
    ) -> BacktestResult:
        """Run OOS backtest on test period, using train data as warmup."""
        combined_df = pd.concat([train_history.df, test_history.df])
        combined_history = PriceHistory(universe, combined_df)

        kwargs = dict(self.base_strategy_kwargs)
        kwargs.update(best_params)
        kwargs.setdefault("name", f"{self.strategy_cls.__name__}_oos")
        strategy = self.strategy_cls(universe=universe, **kwargs)

        engine = Backtest(
            portfolio=portfolio_factory(),
            universe=universe,
            strategies=[strategy],
            history=combined_history,
            warmup_size=len(train_history),
        )
        return engine.run().result(strategy.name).combined
