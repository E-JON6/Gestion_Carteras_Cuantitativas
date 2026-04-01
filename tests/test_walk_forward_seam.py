import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from walk_forward_bl import run_walk_forward


class WalkForwardSeamTests(unittest.TestCase):
    def test_oos_uses_best_is_params_without_window_mixing(self):
        dates = pd.bdate_range("2014-01-01", "2018-12-31")
        fake_data = {
            "backtest_start": pd.Timestamp("2014-01-01"),
            "returns": pd.DataFrame({"wealth": np.zeros(len(dates))}, index=dates),
            "prices": pd.DataFrame({"RISK": np.ones(len(dates)), "XEON.DE": np.ones(len(dates))}, index=dates),
            "todos_tickers": ["RISK", "XEON.DE"],
        }
        call_log = []

        def fake_run_backtest(start=None, end=None, params=None, save_results=True, output_dir=None):
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            params = dict(params or {})
            phase = "OOS" if start.year >= 2017 else "IS"
            call_log.append({"phase": phase, "start": start, "end": end, "params": params, "save_results": save_results})

            if phase == "IS":
                if start.year == 2014:
                    sharpe = 2.0 if params == {"omega_window_fast": 42, "gamma": -2, "recalib_freq": 5} else 1.0
                elif start.year == 2015:
                    sharpe = 2.5 if params == {"omega_window_fast": 21, "gamma": -1, "recalib_freq": 5} else 1.0
                else:
                    sharpe = 0.5
            else:
                if start.year == 2017:
                    self.assertEqual(params, {"omega_window_fast": 42, "gamma": -2, "recalib_freq": 5})
                elif start.year == 2018:
                    self.assertEqual(params, {"omega_window_fast": 21, "gamma": -1, "recalib_freq": 5})
                sharpe = 1.1

            wealth = pd.Series([10_000_000.0, 10_100_000.0], index=[start, end], name="wealth")
            history = pd.DataFrame({"wealth": wealth}, index=wealth.index)
            return {
                "history": history,
                "trades": pd.DataFrame(),
                "wealth": wealth,
                "sharpe": sharpe,
                "cagr": 0.10,
                "max_dd": -0.05,
                "avg_rf": 0.02,
                "start": start,
                "end": end,
                "params": params,
            }

        with patch("walk_forward_bl._load_shared_data", return_value=(fake_data, pd.Series(0.02, index=dates))), patch(
            "walk_forward_bl.run_backtest", side_effect=fake_run_backtest
        ):
            result = run_walk_forward(
                is_years=3,
                oos_years=1,
                paso_years=1,
                param_grid={
                    "omega_window_fast": [21, 42],
                    "gamma": [-1, -2],
                    "recalib_freq": [5],
                },
                save_results=False,
            )

        oos_calls = [entry for entry in call_log if entry["phase"] == "OOS"]
        is_calls = [entry for entry in call_log if entry["phase"] == "IS"]
        self.assertEqual(len(oos_calls), 2)
        self.assertEqual(len(is_calls), 8)
        self.assertTrue(all(not entry["save_results"] for entry in call_log))
        self.assertEqual([entry["start"].year for entry in oos_calls], [2017, 2018])
        self.assertEqual(result["summary"]["param_omega_window_fast"].tolist(), [42, 21])
        self.assertEqual(result["summary"]["param_gamma"].tolist(), [-2, -1])
        self.assertAlmostEqual(float(result["ratio_oos_is"]), 1.1 / ((2.0 + 2.5) / 2.0), places=8)


if __name__ == "__main__":
    unittest.main()
