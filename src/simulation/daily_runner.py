from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.domain.asset import Universe, PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Broker, Trade
from src.io.provider import DataProvider
from src.io.excel_exporter import OperativaExporter
from src.io.portfolio_state import PortfolioStateManager
from src.io.vl_tracker import VLTracker
from src.strategy.base import Strategy


@dataclass
class DailyRunner:
    """Orchestrates daily live trading simulation.

    Flow per day:
        1. Load portfolio state (or create new if day 1)
        2. Download today's close prices
        3. Mark-to-market: record VL BEFORE trading
        4. Execute strategy -> signals -> broker -> trades
        5. Generate operativa Excel (if trades)
        6. Save updated portfolio state
        7. Record post-trade VL
    """

    strategy: Strategy
    universe: Universe
    provider: DataProvider
    initial_cash: float = 10_000_000.0
    max_leverage: float = 2.0
    warmup_days: int = 504
    state_dir: str = "outputs/state"
    operativa_dir: str = "outputs/operativa"
    group_name: str = "Grupo4"
    risk_free_rate: float = 0.0

    _portfolio: Portfolio = field(init=False, repr=False)
    _broker: Broker = field(init=False, repr=False)
    _state_mgr: PortfolioStateManager = field(init=False, repr=False)
    _exporter: OperativaExporter = field(init=False, repr=False)
    _vl_tracker: VLTracker = field(init=False, repr=False)

    def __post_init__(self):
        self._portfolio = Portfolio(
            initial_cash=self.initial_cash,
            max_leverage=self.max_leverage,
        )
        self._broker = Broker(transaction_costs=self.universe.transaction_costs)
        self._state_mgr = PortfolioStateManager(state_dir=self.state_dir)
        self._exporter = OperativaExporter(
            universe=self.universe,
            output_dir=self.operativa_dir,
            group_name=self.group_name,
        )
        self._vl_tracker = VLTracker(
            state_dir=self.state_dir,
            risk_free_rate=self.risk_free_rate,
        )

    def run_today(self, date: pd.Timestamp | None = None) -> dict:
        """Execute the full daily pipeline. Returns a summary dict."""
        if date is None:
            date = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

        # 1. Load state or start fresh
        saved_date = self._state_mgr.load_into_portfolio(self._portfolio)
        is_first_day = saved_date is None

        # 2. Download price history (warmup + today)
        history_start = date - pd.tseries.offsets.BDay(self.warmup_days + 10)
        prices_df = self.provider.get_prices(
            self.universe.tickers,
            start=history_start,
            end=date,
        )
        history = PriceHistory(self.universe, prices_df, fill_na=True)

        # Ensure today is in the history
        if date not in history.dates:
            # Try to get the last available date
            available = history.dates
            if len(available) == 0:
                return {"error": f"No price data available up to {date}"}
            date = available[-1]

        today_prices = history.at(date)

        # 3. Record pre-trade VL (mark-to-market)
        pre_trade_vl = self._portfolio.total_value(today_prices)

        # 4. Initialize strategy if first day
        if is_first_day or not self.strategy.initialized:
            warmup_end_idx = max(0, len(history) - 2)
            warmup_df = history.df.iloc[:warmup_end_idx]
            warmup_history = PriceHistory(self.universe, warmup_df, fill_na=True)
            self.strategy.initialize(warmup_history, self._portfolio)

        # 5. Generate signals and execute
        signals = self.strategy.on_step(today_prices, self._portfolio)
        orders, trades = self._broker.execute(self._portfolio, signals, today_prices)

        # 6. Export operativa Excel
        excel_path = self._exporter.export(trades, date)

        # 7. Save portfolio state
        self._state_mgr.save(self._portfolio, today_prices, date)

        # 8. Record post-trade VL
        post_trade_vl = self._portfolio.total_value(today_prices)
        self._vl_tracker.record(date, post_trade_vl)

        return {
            "date": date,
            "pre_trade_vl": pre_trade_vl,
            "post_trade_vl": post_trade_vl,
            "n_signals": len(signals),
            "n_trades": len(trades),
            "n_orders": len(orders),
            "cash": self._portfolio.cash,
            "positions": self._portfolio.positions,
            "excel_path": str(excel_path) if excel_path else None,
            "is_first_day": is_first_day,
        }

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def vl_tracker(self) -> VLTracker:
        return self._vl_tracker

    @property
    def state_manager(self) -> PortfolioStateManager:
        return self._state_mgr
