"""Grid search and random search optimizers."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
import logging

from src.domain.asset import Universe, PriceHistory
from src.domain.portfolio import Portfolio
from src.strategy.base import Strategy
from src.backtesting.backtest import Backtest
from src.backtesting.result import BacktestResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a parameter search."""

    best_params: dict[str, Any]
    best_metric: float
    all_results: list[tuple[dict[str, Any], float]]
    best_backtest: BacktestResult | None


@dataclass
class GridSearchOptimizer:
    """Runs a parameter search over a grid or random sample.

    Works with any iterable of param dicts — ``ParameterGrid``,
    ``RandomParameterGrid``, or a plain list.

    Args:
        strategy_cls: Strategy class to instantiate.
        param_grid: Iterable of param dicts to try.
        metric: Metric name from BacktestResult.metrics (default: sharpe_ratio).
        maximize: True to maximize, False to minimize.
        warmup_size: Warmup for each backtest.
        base_strategy_kwargs: Fixed kwargs passed to every strategy instance.
    """

    strategy_cls: type[Strategy]
    param_grid: Iterator[dict[str, Any]]
    metric: str = "sharpe_ratio"
    maximize: bool = True
    warmup_size: int | str = 252
    base_strategy_kwargs: dict = field(default_factory=dict)

    def optimize(
        self,
        universe: Universe,
        history: PriceHistory,
        portfolio_factory: Callable[[], Portfolio],
    ) -> OptimizationResult:
        """Run search and return the best parameter combination."""
        best_params = None
        best_metric = self._worst_metric()
        best_backtest = None
        all_results = []

        total = len(self.param_grid) if hasattr(self.param_grid, "__len__") else "?"

        for i, params in enumerate(self.param_grid):
            logger.info(f"[{i + 1}/{total}] Testing {params}")

            result = self._try_backtest(params, universe, history, portfolio_factory)
            if result is None:
                all_results.append((params, self._worst_metric()))
                continue

            metric_value = result.metrics.get(self.metric, self._worst_metric())
            all_results.append((params, metric_value))

            if self._is_better(metric_value, best_metric):
                best_metric = metric_value
                best_params = params
                best_backtest = result
                logger.info(f"  New best: {self.metric}={best_metric:.4f}")

        if best_params is None and all_results:
            best_params = all_results[0][0]
            logger.warning(f"All combos failed. Falling back to {best_params}")

        return OptimizationResult(
            best_params=best_params or {},
            best_metric=best_metric,
            all_results=all_results,
            best_backtest=best_backtest,
        )

    def _try_backtest(
        self,
        params: dict[str, Any],
        universe: Universe,
        history: PriceHistory,
        portfolio_factory: Callable[[], Portfolio],
    ) -> BacktestResult | None:
        try:
            strategy = self._build_strategy(universe, params)
            engine = Backtest(
                portfolio=portfolio_factory(),
                universe=universe,
                strategies=[strategy],
                history=history,
                warmup_size=self.warmup_size,
            )
            return engine.run().result(strategy.name).combined
        except Exception as e:
            logger.warning(f"Failed for {params}: {e}")
            return None

    def _build_strategy(self, universe: Universe, params: dict[str, Any]) -> Strategy:
        kwargs = dict(self.base_strategy_kwargs)
        kwargs.update(params)
        kwargs.setdefault("name", f"{self.strategy_cls.__name__}_opt")
        return self.strategy_cls(universe=universe, **kwargs)

    def _worst_metric(self) -> float:
        return float("-inf") if self.maximize else float("inf")

    def _is_better(self, new: float, current: float) -> bool:
        return new > current if self.maximize else new < current
