from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Trade


@dataclass
class PortfolioStateManager:
    """Persists and loads portfolio state (cash + positions) between sessions.

    State is stored as JSON with the current snapshot, a history of
    daily VL values, daily positions, and all executed operations.
    """

    state_dir: str = "outputs/state"

    @property
    def _state_path(self) -> Path:
        return Path(self.state_dir) / "portfolio_state.json"

    def save(
        self,
        portfolio: Portfolio,
        prices: PriceSnapshot,
        date: pd.Timestamp,
        trades: list[Trade] | None = None,
    ) -> None:
        """Save current portfolio state to disk."""
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)

        state = self._load_raw() if self._state_path.exists() else {
            "history": [],
            "positions_history": [],
            "operations_history": [],
        }

        total_value = portfolio.total_value(prices)

        snapshot = {
            "date": date.isoformat(),
            "cash": portfolio.cash,
            "positions": portfolio.positions,
            "total_value": total_value,
        }
        state["current"] = snapshot

        # VL history (dedup by date)
        history = state.get("history", [])
        history = [h for h in history if h["date"] != date.isoformat()]
        history.append({"date": date.isoformat(), "total_value": total_value, "cash": portfolio.cash})
        history.sort(key=lambda h: h["date"])
        state["history"] = history

        # Positions history (daily snapshot of all positions with values)
        pos_history = state.get("positions_history", [])
        pos_history = [p for p in pos_history if p["date"] != date.isoformat()]
        for ticker, shares in portfolio.positions.items():
            if shares != 0:
                price = prices.get_price(ticker)
                value = shares * price
                weight = value / total_value if total_value else 0.0
                pos_history.append({
                    "date": date.isoformat(),
                    "ticker": ticker,
                    "shares": shares,
                    "price": price,
                    "value": value,
                    "weight": weight,
                })
        pos_history.sort(key=lambda p: (p["date"], p["ticker"]))
        state["positions_history"] = pos_history

        # Operations history (all trades)
        ops_history = state.get("operations_history", [])
        if trades:
            date_str = date.isoformat()
            # Remove any operations for today (idempotent re-runs)
            ops_history = [o for o in ops_history if o["date"] != date_str]
            for t in trades:
                ct = abs(t.shares) * t.price * 0  # cost already in t.cost
                ops_history.append({
                    "date": date_str,
                    "ticker": t.ticker,
                    "shares": t.shares,
                    "price": t.price,
                    "value": t.shares * t.price,
                    "cost": t.cost,
                    "direction": "buy" if t.shares > 0 else "sell",
                })
            ops_history.sort(key=lambda o: (o["date"], o["ticker"]))
        state["operations_history"] = ops_history

        self._state_path.write_text(json.dumps(state, indent=2, default=str))

    def load(self) -> dict | None:
        """Load last saved state. Returns None if no state exists."""
        if not self._state_path.exists():
            return None
        state = self._load_raw()
        return state.get("current")

    def load_into_portfolio(self, portfolio: Portfolio) -> pd.Timestamp | None:
        """Restore a portfolio from saved state. Returns the date of saved state."""
        current = self.load()
        if current is None:
            return None
        portfolio._cash = current["cash"]
        portfolio._positions = {k: v for k, v in current["positions"].items()}
        return pd.Timestamp(current["date"])

    def value_history(self) -> pd.Series:
        """Return the VL time series from saved history."""
        if not self._state_path.exists():
            return pd.Series(dtype=float)
        state = self._load_raw()
        history = state.get("history", [])
        if not history:
            return pd.Series(dtype=float)
        dates = [pd.Timestamp(h["date"]) for h in history]
        values = [h["total_value"] for h in history]
        return pd.Series(values, index=dates, name="total_value")

    def positions_history(self) -> pd.DataFrame:
        """Return all daily position snapshots as a DataFrame."""
        if not self._state_path.exists():
            return pd.DataFrame()
        state = self._load_raw()
        records = state.get("positions_history", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def operations_history(self) -> pd.DataFrame:
        """Return all historical trades as a DataFrame."""
        if not self._state_path.exists():
            return pd.DataFrame()
        state = self._load_raw()
        records = state.get("operations_history", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_raw(self) -> dict:
        return json.loads(self._state_path.read_text())
