from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

_MADRID_TZ = ZoneInfo("Europe/Madrid")

from src.domain.asset import Universe, PriceHistory, PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Broker, Signal, Trade
from src.domain.trade.order import AcceptedOrder
from src.io.provider import DataProvider
from src.io.excel_exporter import OperativaExporter
from src.io.portfolio_state import PortfolioStateManager
from src.io.vl_tracker import VLTracker
from src.strategy.base import Strategy


# IUSE.L transaction cost (from Historial, not in jaime universe)
_IUSE_CT = 0.00021


@dataclass
class DailyRunner:
    """Orchestrates daily live trading simulation.

    Flow per day:
        1. Load portfolio state (or migrate from estado.json if day 1)
        2. Download today's close prices
        3. Mark-to-market: record VL BEFORE trading
        4. Execute strategy -> signals -> broker -> trades
        5. Generate operativa Excel (if trades)
        6. Save updated portfolio state + estado.json + historial
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
    _costes_acumulados: float = field(init=False, repr=False, default=0.0)
    _n_operaciones: int = field(init=False, repr=False, default=0)

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
            date = pd.Timestamp(datetime.now(_MADRID_TZ).strftime("%Y-%m-%d"))

        # 1. Load state or start fresh
        saved_date = self._state_mgr.load_into_portfolio(self._portfolio)
        is_first_day = saved_date is None

        migration_trades: list[Trade] = []
        extra_costs: dict[str, float] | None = None

        if is_first_day:
            migration_trades, extra_costs = self._migrate_from_estado(date)
        else:
            # Load cumulative counters from estado.json
            estado = self._state_mgr.load_estado()
            if estado:
                self._costes_acumulados = estado.get("costes_acumulados", 0.0)
                self._n_operaciones = estado.get("n_operaciones", 0)

        # 2. Download price history (warmup + today)
        history_start = date - pd.tseries.offsets.BDay(self.warmup_days + 10)
        prices_df = self.provider.get_prices(
            self.universe.tickers,
            start=history_start,
            end=date + pd.Timedelta(days=1),
        )
        history = PriceHistory(self.universe, prices_df, fill_na=True)

        # Ensure today is in the history
        if date not in history.dates:
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
        orders, strategy_trades = self._broker.execute(
            self._portfolio, signals, today_prices,
        )

        # 6. Merge migration + strategy trades
        all_trades = migration_trades + strategy_trades

        # 7. Update cumulative counters
        day_costs = sum(t.cost for t in all_trades)
        self._costes_acumulados += day_costs
        self._n_operaciones += len(all_trades)

        # 8. Export operativa Excel
        excel_path = self._exporter.export(
            all_trades, date, extra_costs=extra_costs,
        )

        # 9. Save portfolio state (internal detailed history)
        self._state_mgr.save(
            self._portfolio, today_prices, date,
            trades=all_trades,
            strategy_name=self.strategy.name,
        )

        # 10. Compute post-trade VL
        post_trade_vl = self._portfolio.total_value(today_prices)

        # 10b. On migration: seed legacy VL + operations BEFORE today's VL
        #      so today's record overwrites the legacy value for the same date
        if is_first_day and migration_trades:
            vl_entries, legacy_ops = self._state_mgr.load_legacy_from_historial(
                group_name=self.group_name,
            )
            self._state_mgr.seed_legacy_data(
                vl_entries, legacy_ops, self._vl_tracker,
            )

        # 10c. Record today's post-trade VL (overwrites legacy if same date)
        self._vl_tracker.record(date, post_trade_vl)

        # 11. Save estado.json
        self._state_mgr.save_estado(
            portfolio=self._portfolio,
            prices=today_prices,
            date=date,
            costes_acumulados=self._costes_acumulados,
            n_operaciones=self._n_operaciones,
            strategy_name=self.strategy.name,
            capital_inicial=self.initial_cash,
        )

        # 12. Save historial row
        retorno_diario, retorno_acum = self._compute_returns(post_trade_vl)
        n_posiciones = len([
            s for s in self._portfolio.positions.values() if abs(s) > 1e-9
        ])
        importe_operado = sum(t.abs_value_before_costs for t in all_trades)

        if all_trades:
            decision = "TRANSICION" if migration_trades else "REBALANCEO"
        else:
            decision = "MANTENER"

        historial_path = self._state_mgr.save_historial(
            date=date,
            valor_cartera=post_trade_vl,
            retorno_diario=retorno_diario,
            retorno_acum=retorno_acum,
            n_posiciones=n_posiciones,
            importe_operado=importe_operado,
            coste_dia=day_costs,
            n_ops_dia=len(all_trades),
            n_ops_acum=self._n_operaciones,
            costes_acum=self._costes_acumulados,
            cash=self._portfolio.cash,
            decision=decision,
            group_name=self.group_name,
        )

        return {
            "date": date,
            "pre_trade_vl": pre_trade_vl,
            "post_trade_vl": post_trade_vl,
            "n_signals": len(signals),
            "n_trades": len(all_trades),
            "n_orders": len(orders),
            "cash": self._portfolio.cash,
            "positions": self._portfolio.positions,
            "today_prices": today_prices,
            "history_df": history.df.iloc[-504:].copy(),
            "trades": all_trades,
            "excel_path": str(excel_path) if excel_path else None,
            "historial_path": str(historial_path),
            "is_first_day": is_first_day,
            "migration_trades": migration_trades,
            "costes_acumulados": self._costes_acumulados,
            "n_operaciones": self._n_operaciones,
        }

    # ── Migration from legacy estado.json ──────────────────────────

    def _migrate_from_estado(
        self, date: pd.Timestamp,
    ) -> tuple[list[Trade], dict[str, float] | None]:
        """Bootstrap portfolio from estado.json (previous strategy).

        Liquidates IUSE.L (not in universe), keeps XEON.DE.
        Returns (migration_trades, extra_costs_for_operativa).
        """
        estado = self._state_mgr.load_estado()
        if not estado or "participaciones" not in estado:
            # No legacy state — start fresh
            self._costes_acumulados = 0.0
            self._n_operaciones = 0
            return [], None

        iuse_shares = estado.get("participaciones", 0.0)
        xeon_shares = estado.get("participaciones_rf", 0.0)
        prev_costes = estado.get("costes_acumulados", 0.0)
        prev_n_ops = estado.get("n_operaciones", 0)

        migration_trades = []
        migration_cost = 0.0
        total_cash = 0.0

        # Sell IUSE.L (not in universe)
        if iuse_shares > 0:
            iuse_price = self._fetch_single_price("IUSE.L", date)
            cost = iuse_shares * iuse_price * _IUSE_CT
            signal = Signal(
                date=date, ticker="IUSE.L",
                target_weight=0.0, reason="liquidation",
            )
            order = AcceptedOrder(signal=signal, shares=-iuse_shares, price=iuse_price)
            migration_trades.append(Trade(order=order, cost=cost))
            migration_cost += cost
            total_cash += iuse_shares * iuse_price - cost

        # Sell XEON.DE too (CT=0) so portfolio starts empty and strategy
        # will rebalance on the first step (portfolio_uninvested=True)
        if xeon_shares > 0:
            xeon_price = self._fetch_single_price("XEON.DE", date)
            signal = Signal(
                date=date, ticker="XEON.DE",
                target_weight=0.0, reason="liquidation",
            )
            order = AcceptedOrder(signal=signal, shares=-xeon_shares, price=xeon_price)
            migration_trades.append(Trade(order=order, cost=0.0))
            total_cash += xeon_shares * xeon_price

        # Set portfolio to pure cash — strategy will allocate from scratch
        self._portfolio._cash = total_cash
        self._portfolio._positions = {}

        self._costes_acumulados = prev_costes + migration_cost
        self._n_operaciones = prev_n_ops + len(migration_trades)

        extra_costs = {"IUSE.L": _IUSE_CT} if migration_trades else None
        return migration_trades, extra_costs

    def _fetch_single_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Fetch the closing price for a single ticker on or before date."""
        start = date - pd.Timedelta(days=10)
        end = date + pd.Timedelta(days=1)
        prices_df = self.provider.get_prices([ticker], start=start, end=end)
        # Use the last available price up to date
        prices_df = prices_df.loc[prices_df.index <= date]
        if prices_df.empty:
            raise ValueError(f"No price data for {ticker} up to {date}")
        return float(prices_df[ticker].iloc[-1])

    # ── Helpers ─────────────────────────────────────────────────────

    def _compute_returns(self, current_vl: float) -> tuple[float, float]:
        """Compute daily and cumulative returns from VL history."""
        vl_df = self._vl_tracker.history()
        if len(vl_df) < 2:
            retorno_acum = (current_vl - self.initial_cash) / self.initial_cash
            return 0.0, retorno_acum

        prev_vl = vl_df["total_value"].iloc[-2]
        retorno_diario = (current_vl - prev_vl) / prev_vl
        retorno_acum = (current_vl / self.initial_cash) - 1
        return retorno_diario, retorno_acum

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def vl_tracker(self) -> VLTracker:
        return self._vl_tracker

    @property
    def state_manager(self) -> PortfolioStateManager:
        return self._state_mgr
