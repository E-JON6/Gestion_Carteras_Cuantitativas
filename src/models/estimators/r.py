import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.domain.asset import PriceHistory


_logger = logging.getLogger(__name__)

# Bundled €STR (≈ ECB DFR) step series, resolved relative to the project root.
DEFAULT_ESTR_PATH = (
    Path(__file__).resolve().parents[3] / "inputs" / "rates" / "estr.csv"
)


class RiskFreeRateEstimator(ABC):
    """Estimator for the risk-free rate.

    All estimators accept a PriceHistory and return an
    annualised risk-free rate as a float.
    """

    @abstractmethod
    def estimate(self, history: PriceHistory) -> float: ...


@dataclass
class DefensiveRfrEstimator(RiskFreeRateEstimator):
    """Estimates annualised risk-free rate from a defensive ticker's price history.

    Uses geometric annualisation: (last / first) ^ (252 / n_days) - 1.
    """

    defensive_ticker: str
    window: int = 252

    def estimate(self, history: PriceHistory) -> float:
        if self.defensive_ticker not in history.tickers:
            raise ValueError(f"defensive ticker {self.defensive_ticker} not in history")

        series = history.df[self.defensive_ticker].dropna()
        if len(series) < 2:
            raise ValueError(
                f"not enough data to estimate rfr from {self.defensive_ticker}"
            )

        series = series.iloc[-self.window:]
        total_return = series.iloc[-1] / series.iloc[0]
        n_days = len(series)
        return float(total_return ** (252 / n_days) - 1)


@dataclass
class FixedRiskFreeRate(RiskFreeRateEstimator):
    """Returns a fixed risk-free rate (useful for testing or known rates)."""

    rate: float = 0.0

    def estimate(self, history: PriceHistory) -> float:
        return self.rate


@dataclass
class FileRfrEstimator(RiskFreeRateEstimator):
    """Reads a step series of risk-free rates from a CSV file.

    The CSV must have two columns: ``date`` (parseable timestamp) and
    ``rate`` (decimal, e.g. ``0.025`` for 2.5%). For each call, the
    estimator returns the latest rate on or before ``history.dates[-1]``
    (forward-fill semantics — the rate persists until the next change).

    On any error (file missing, parse failure, no data on/before the date)
    it logs a warning *once* and returns ``fallback``.

    Defaults to the bundled ``inputs/rates/estr.csv`` (€STR ≈ ECB DFR).
    """

    path: Path | str = field(default_factory=lambda: DEFAULT_ESTR_PATH)
    fallback: float = 0.0

    _cache: pd.Series | None = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
    _warned: bool = field(default=False, init=False, repr=False)

    def estimate(self, history: PriceHistory) -> float:
        if len(history) == 0:
            return self._warn_and_fallback("history is empty")

        target = history.dates[-1]
        self._ensure_loaded()

        if self._cache is None or self._cache.empty:
            return self._warn_and_fallback("series unavailable")

        valid = self._cache.loc[:target]
        if valid.empty:
            return self._warn_and_fallback(f"no data on or before {target.date()}")

        return float(valid.iloc[-1])

    # ── Internals ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            df = pd.read_csv(self.path)
            if "date" not in df.columns or "rate" not in df.columns:
                self._warn_and_fallback(
                    f"{self.path} must have columns 'date' and 'rate'"
                )
                return

            df = df.dropna(subset=["date", "rate"])
            if df.empty:
                self._warn_and_fallback(f"{self.path} has no usable rows")
                return

            idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
            if idx.tz is not None:
                idx = idx.tz_localize(None)

            self._cache = (
                pd.Series(df["rate"].astype(float).values, index=idx)
                .sort_index()
            )
        except Exception as e:
            self._warn_and_fallback(f"failed to load {self.path}: {e}")

    def _warn_and_fallback(self, reason: str) -> float:
        if not self._warned:
            _logger.warning(
                "FileRfrEstimator(%s): %s. Falling back to %s.",
                self.path, reason, self.fallback,
            )
            self._warned = True
        return self.fallback
