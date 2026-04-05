from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.performance import PerformanceReport


@dataclass
class VLTracker:
    """Tracks the daily Net Asset Value (Valor Liquidativo) over time.

    Persists to CSV and can export a full seguimiento Excel with
    multiple sheets: VL, positions, operations, metrics.
    """

    state_dir: str = "outputs/state"
    risk_free_rate: float = 0.0

    @property
    def _csv_path(self) -> Path:
        return Path(self.state_dir) / "vl_history.csv"

    def record(self, date: pd.Timestamp, total_value: float) -> None:
        """Append a VL entry for a given date."""
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)

        entry = pd.DataFrame([{"date": date, "total_value": total_value}])

        if self._csv_path.exists():
            existing = pd.read_csv(self._csv_path, parse_dates=["date"])
            # Remove duplicate date if re-running same day
            existing = existing[existing["date"] != date]
            combined = pd.concat([existing, entry], ignore_index=True)
        else:
            combined = entry

        combined.sort_values("date", inplace=True)
        combined.to_csv(self._csv_path, index=False)

    def history(self) -> pd.DataFrame:
        """Load the full VL history as a DataFrame with computed columns."""
        if not self._csv_path.exists():
            return pd.DataFrame(columns=["date", "total_value", "return", "cumulative_return", "drawdown"])

        df = pd.read_csv(self._csv_path, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["return"] = df["total_value"].pct_change()
        df["cumulative_return"] = df["total_value"] / df["total_value"].iloc[0] - 1
        cummax = df["total_value"].cummax()
        df["drawdown"] = (df["total_value"] - cummax) / cummax

        return df

    def metrics(self) -> dict:
        """Compute performance metrics from the VL history."""
        df = self.history()
        if len(df) < 2:
            return {}
        values = pd.Series(df["total_value"].values, index=df["date"])
        report = PerformanceReport(values=values, risk_free_rate=self.risk_free_rate)
        return report.compute()

    def export_seguimiento(
        self,
        positions_history: list[dict] | None = None,
        operations_history: list[dict] | None = None,
        output_path: str | None = None,
    ) -> Path:
        """Export a full tracking Excel workbook.

        Sheets:
            - VL: date, total_value, return, cumulative_return, drawdown
            - Posiciones: date, ticker, shares, weight, value (if provided)
            - Operaciones: all executed trades (if provided)
            - Metricas: summary performance metrics
        """
        out = Path(output_path or "outputs/reports/seguimiento.xlsx")
        out.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            # VL sheet
            vl_df = self.history()
            if not vl_df.empty:
                vl_df.to_excel(writer, sheet_name="VL", index=False)

            # Posiciones sheet
            if positions_history:
                pos_df = pd.DataFrame(positions_history)
                pos_df.to_excel(writer, sheet_name="Posiciones", index=False)

            # Operaciones sheet
            if operations_history:
                ops_df = pd.DataFrame(operations_history)
                ops_df.to_excel(writer, sheet_name="Operaciones", index=False)

            # Metricas sheet
            m = self.metrics()
            if m:
                metrics_df = pd.DataFrame([m]).T
                metrics_df.columns = ["Value"]
                metrics_df.index.name = "Metric"
                metrics_df.to_excel(writer, sheet_name="Metricas")

        return out
