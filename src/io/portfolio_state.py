from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio


@dataclass
class PortfolioStateManager:
    """Persists and loads portfolio state (cash + positions) between sessions.

    State is stored as JSON with the current snapshot and a history of
    daily VL values for performance tracking.
    """

    state_dir: str = "outputs/state"

    @property
    def _state_path(self) -> Path:
        return Path(self.state_dir) / "portfolio_state.json"

    def save(self, portfolio: Portfolio, prices: PriceSnapshot, date: pd.Timestamp) -> None:
        """Save current portfolio state to disk."""
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)

        state = self._load_raw() if self._state_path.exists() else {"history": []}

        snapshot = {
            "date": date.isoformat(),
            "cash": portfolio.cash,
            "positions": portfolio.positions,
            "total_value": portfolio.total_value(prices),
        }

        state["current"] = snapshot

        # Append to history (avoid duplicates for same date)
        history = state.get("history", [])
        history = [h for h in history if h["date"] != date.isoformat()]
        history.append({
            "date": date.isoformat(),
            "total_value": snapshot["total_value"],
            "cash": snapshot["cash"],
        })
        history.sort(key=lambda h: h["date"])
        state["history"] = history

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

    def _load_raw(self) -> dict:
        return json.loads(self._state_path.read_text())
