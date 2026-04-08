# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Python framework for designing, backtesting, and comparing daily trading strategies on ETFs. Academic project (master's work). No packaging setup — pure Python with direct imports from `src/`.

## Commands

```bash
# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_backtest.py

# Run a specific test
python -m pytest tests/test_backtest.py::test_name -v

# Install dependencies
pip install -r requirements.txt
```

No build step, no linter configured, no CI pipeline.

## Architecture

### Simulation Loop (per day)

```
Strategy._generate_signals(prices, portfolio) → list[Signal]
    ↓
Broker.execute(portfolio, signals, prices) → (orders, trades)
    ↓
Portfolio updates positions and cash
```

This loop runs inside `BacktestEngine._simulate()` (`src/backtesting/engine.py`).

### Key Domain Types

- **Signal** (`src/domain/trade/signal.py`): Frozen dataclass — desired target weight for a ticker. Not an order.
- **Portfolio** (`src/domain/portfolio/base.py`): Mutable state — cash + positions (shares). Supports leverage and short constraints.
- **PriceHistory** / **PriceSnapshot** (`src/domain/asset/`): Historical prices DataFrame wrapper and single-day price view.
- **Universe** (`src/domain/asset/universe.py`): Set of Assets with transaction costs and sectors. Loaded from YAML in `inputs/universes/`.
- **Broker** (`src/domain/trade/broker.py`): Converts signals to orders, validates constraints, executes (sells first, then buys).

### Strategy Pattern

All strategies inherit from `Strategy` (`src/strategy/base.py`) using `@dataclass(kw_only=True)`. Two abstract methods:
- `_warmup(portfolio)` — called once with warmup data
- `_generate_signals(prices, portfolio)` → `list[Signal]` — called each simulation day

Strategies compose reusable components:
- **Signalers** (`src/strategy/signaler/`): Generate raw signals (Merton, Momentum, VolTargeting, DualMomentum)
- **Filters** (`src/strategy/filter/`): Transform/reduce signals (TopN, NoTradeBand, CostAware, Frozen)
- **Transformers** (`src/strategy/transformers/`): Utilities like `normalise_weights`, `fill_defensive`

Typical pipeline: `Signaler → Filters → normalise_weights → fill_defensive → CostAwareFilter`

### Models (`src/models/`)

Financial math: MertonDavisNorman (optimal weights + no-trade bands), BlackLitterman, OmegaRanker, VolatilityRegimeDetector, LeverageGuard.

### Estimators (`src/models/estimators/`)

All take `PriceHistory`, return dicts or DataFrames. Categories: mu (return), sigma (volatility), covariance, risk-free rate.

### Backtesting Engines (`src/backtesting/`)

All inherit from `BacktestEngine` and differ only in `_split_history()`:
- `Backtest` — single warmup + simulation window
- `WalkForwardBacktest` — sliding fixed-size window
- `RollingBacktest` — re-initializes strategy across train/test windows

Optimization: `GridSearchOptimizer`, `WalkForwardOptimizer` with `ParameterGrid` / `RandomParameterGrid`.

### Results (`src/backtesting/result/`)

- `BacktestResult` — single window: snapshots, signals, orders, trades
- `StrategyResult` — one strategy across windows (has `.combined`)
- `MultipleResult` — multiple strategies with comparison plots and `summary_df`

### IO (`src/io/`)

Data loading with provider pattern: `YFinanceProvider` (live), `FileProvider` (cached CSV). Universe YAML files in `inputs/universes/`.

## Conventions

- All domain objects are `@dataclass` (many frozen).
- Strategies use `@dataclass(kw_only=True)`.
- Python 3.12 (uses `X | Y` union syntax).
- Notebooks in `notebooks/` for interactive analysis.
- `GUIDE.md` contains detailed usage documentation and strategy creation guide.
