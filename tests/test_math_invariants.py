import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from engine_multiasset import MultiAssetBacktestEngine
from main_black_litterman import run_backtest
from Models.black_litterman import run_black_litterman
from Models.merton import compute_optimal_weights
from portfolio.registrador import run_registrador_v0
from strategies.bl_omega_strategy import BLOmegaStrategy


class SwitchToRiskAfterJumpStrategy:
    """Mantiene XEON al inicio y pasa a riesgo después del salto ya realizado."""

    def __init__(self, tickers):
        self.tickers = tickers
        self.name = "switch-after-jump"

    def get_initial_weights(self, returns_df, risk_free_rate, lambda_per_ticker):
        del returns_df, risk_free_rate, lambda_per_ticker
        return np.array([0.0, 1.0], dtype=float)

    def decide(self, date, current_weights, returns_df, risk_free_rate, lambda_per_ticker, day_index):
        del date, current_weights, returns_df, risk_free_rate, lambda_per_ticker
        if day_index == 1:
            return {
                "rebalance": True,
                "target_weights": np.array([1.0, 0.0], dtype=float),
                "reason": "switch_after_observing_jump",
            }
        return None


class AlwaysRiskStrategy:
    def __init__(self, tickers):
        self.tickers = tickers
        self.name = "always-risk"

    def get_initial_weights(self, returns_df, risk_free_rate, lambda_per_ticker):
        del returns_df, risk_free_rate, lambda_per_ticker
        return np.array([1.0, 0.0], dtype=float)

    def decide(self, date, current_weights, returns_df, risk_free_rate, lambda_per_ticker, day_index):
        del date, current_weights, returns_df, risk_free_rate, lambda_per_ticker, day_index
        return None


class MathInvariantTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=140)
        rng = np.random.default_rng(7)
        self.returns_df = pd.DataFrame(
            {
                "R1": 0.0010 + rng.normal(0.0, 0.0002, len(self.dates)),
                "R2": 0.0008 + rng.normal(0.0, 0.0002, len(self.dates)),
                "R3": 0.0006 + rng.normal(0.0, 0.0002, len(self.dates)),
                "R4": -0.0002 + rng.normal(0.0, 0.0002, len(self.dates)),
                "XEON.DE": 0.00005 + rng.normal(0.0, 0.00001, len(self.dates)),
            },
            index=self.dates,
        )
        self.prices_df = 100.0 * (1.0 + self.returns_df).cumprod()

    def test_black_litterman_and_merton_contracts_and_sum_of_weights(self):
        bl_result = run_black_litterman(self.returns_df, risk_free_rate=0.02, xeon_ticker="XEON.DE")
        required_bl_keys = {"mu_BL", "Sigma", "tickers", "top_tickers", "omega_scores", "Q", "confidence"}
        self.assertTrue(required_bl_keys.issubset(bl_result.keys()))
        self.assertIn("XEON.DE", bl_result["tickers"])
        self.assertNotIn("XEON.DE", bl_result["top_tickers"])
        self.assertEqual(len(bl_result["top_tickers"]), 3)
        self.assertEqual(bl_result["Q"].shape, (3,))
        self.assertEqual(bl_result["confidence"].shape, (3,))
        self.assertEqual(bl_result["Sigma"].shape, (len(bl_result["tickers"]), len(bl_result["tickers"])))

        merton_result = compute_optimal_weights(
            bl_result,
            risk_free_rate=0.02,
            xeon_ticker="XEON.DE",
            sigma_mercado=0.18,
            categoria_por_ticker={
                "R1": "equity",
                "R2": "equity_2",
                "R3": "bond",
                "R4": "commodity",
            },
        )
        required_merton_keys = {"weights_array", "weights_dict", "selected_tickers", "weight_xeon", "w_raw", "regime", "liquidar"}
        self.assertTrue(required_merton_keys.issubset(merton_result.keys()))
        self.assertEqual(merton_result["weights_array"].shape, (len(bl_result["tickers"]),))
        self.assertAlmostEqual(float(np.sum(merton_result["weights_array"])), 1.0, places=8)
        self.assertGreaterEqual(float(merton_result["weight_xeon"]), 0.0)

    def test_merton_long_only_goes_all_in_xeon_when_all_excess_returns_are_non_positive(self):
        tickers = ["R1", "R2", "R3", "XEON.DE"]
        bl_result = {
            "mu_BL": np.array([0.01, 0.015, 0.02, 0.02], dtype=float),
            "Sigma": np.diag([0.04, 0.05, 0.06, 0.0001]),
            "tickers": tickers,
        }
        result = compute_optimal_weights(bl_result, risk_free_rate=0.02, xeon_ticker="XEON.DE", sigma_mercado=0.10)
        self.assertEqual(result["weights_dict"], {})
        self.assertEqual(result["selected_tickers"], [])
        self.assertAlmostEqual(float(result["weight_xeon"]), 1.0, places=10)
        self.assertAlmostEqual(float(result["weights_array"][tickers.index("XEON.DE")]), 1.0, places=10)
        self.assertTrue(np.allclose(result["weights_array"][:3], 0.0))

    def test_strategy_initial_weights_sum_to_one(self):
        strategy = BLOmegaStrategy(
            tickers=list(self.returns_df.columns),
            xeon_ticker="XEON.DE",
            recalib_freq=5,
            gamma=-2,
            categoria_por_ticker={"R1": "equity", "R2": "equity_2", "R3": "bond", "R4": "commodity"},
            omega_window_fast=21,
        )
        weights = strategy.get_initial_weights(
            self.returns_df,
            risk_free_rate=0.02,
            lambda_per_ticker={ticker: 0.0 for ticker in self.returns_df.columns},
        )
        self.assertEqual(weights.shape, (len(self.returns_df.columns),))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue(np.all(weights >= 0.0))

    def test_engine_has_no_lookahead(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {
                "RISK": [100.0, 200.0, 200.0],
                "XEON.DE": [100.0, 100.0, 100.0],
            },
            index=dates,
        )
        returns = prices.pct_change().fillna(0.0)
        rf = pd.Series(0.0, index=dates)
        engine = MultiAssetBacktestEngine(
            prices_df=prices,
            returns_df=returns,
            risk_free_series=rf,
            xeon_ticker="XEON.DE",
            lambda_per_ticker={"RISK": 0.0, "XEON.DE": 0.0},
            start_date=dates[0],
            end_date=dates[-1],
            initial_wealth=100.0,
            warmup_days=0,
        )
        result = engine.run(SwitchToRiskAfterJumpStrategy(["RISK", "XEON.DE"]))
        history = result["history"]
        self.assertAlmostEqual(float(history.iloc[1]["wealth"]), 100.0, places=8)
        self.assertAlmostEqual(float(history.iloc[-1]["wealth"]), 100.0, places=8)
        self.assertAlmostEqual(float(history.iloc[1]["weight_RISK"]), 1.0, places=8)

    def test_manual_backtest_wealth_matches_hand_calculation(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {
                "RISK": [100.0, 110.0, 99.0],
                "XEON.DE": [100.0, 100.0, 100.0],
            },
            index=dates,
        )
        returns = prices.pct_change().fillna(0.0)
        rf = pd.Series(0.0, index=dates)
        engine = MultiAssetBacktestEngine(
            prices_df=prices,
            returns_df=returns,
            risk_free_series=rf,
            xeon_ticker="XEON.DE",
            lambda_per_ticker={"RISK": 0.0, "XEON.DE": 0.0},
            start_date=dates[0],
            end_date=dates[-1],
            initial_wealth=100.0,
            warmup_days=0,
        )
        result = engine.run(AlwaysRiskStrategy(["RISK", "XEON.DE"]))
        wealth = result["history"]["wealth"]
        self.assertAlmostEqual(float(wealth.iloc[0]), 100.0, places=8)
        self.assertAlmostEqual(float(wealth.iloc[1]), 110.0, places=8)
        self.assertAlmostEqual(float(wealth.iloc[2]), 99.0, places=8)

    def test_engine_output_contract_keys_exist(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {
                "RISK": [100.0, 101.0, 102.0],
                "XEON.DE": [100.0, 100.0, 100.0],
            },
            index=dates,
        )
        returns = prices.pct_change().fillna(0.0)
        rf = pd.Series(0.0, index=dates)
        engine = MultiAssetBacktestEngine(
            prices_df=prices,
            returns_df=returns,
            risk_free_series=rf,
            xeon_ticker="XEON.DE",
            lambda_per_ticker={"RISK": 0.0, "XEON.DE": 0.0},
            start_date=dates[0],
            end_date=dates[-1],
            initial_wealth=100.0,
            warmup_days=0,
        )
        result = engine.run(AlwaysRiskStrategy(["RISK", "XEON.DE"]))
        expected_keys = {"history", "trades", "portfolio", "strategy_name", "tickers", "xeon_ticker", "start_date", "end_date", "initial_wealth"}
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertIn("wealth", result["history"].columns)
        self.assertIn("weight_RISK", result["history"].columns)
        self.assertIn("weight_XEON.DE", result["history"].columns)

    def test_registrador_is_self_financed(self):
        market_data = {
            "tickers": ["RISK", "XEON.DE"],
            "prices": pd.DataFrame(
                {
                    "RISK": [100.0, 100.0],
                    "XEON.DE": [100.0, 100.0],
                },
                index=pd.bdate_range("2024-01-01", periods=2),
            ),
            "transaction_costs": {"RISK": 0.10, "XEON.DE": 0.01},
        }
        dn_result = {
            "final_weights_full": {"RISK": 0.6, "XEON.DE": 0.4},
            "weight_xeon": 0.4,
        }
        fake_execution = {
            "date": market_data["prices"].index[-1],
            "prices": {"RISK": 100.0, "XEON.DE": 100.0},
            "used_next_day": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "orders.xlsx"
            with patch("portfolio.registrador.get_execution_data_v0", return_value=fake_execution):
                result = run_registrador_v0(
                    dn_result,
                    market_data,
                    current_positions={"RISK": 0.0, "XEON.DE": 0.0},
                    total_value=100.0,
                    output_path=str(output_path),
                )
        self.assertGreaterEqual(float(result["cash_slack"]), -1e-6)
        self.assertLessEqual(float(result["buy_executed_notional"]), float(result["budget"]) + 1e-6)
        net_cash_needed = float(result["buy_executed_notional"]) - float(result["sell_executed_notional"])
        self.assertLessEqual(net_cash_needed, float(result["budget"]) + 1e-6)
        self.assertAlmostEqual(
            net_cash_needed + float(result["cash_slack"]),
            float(result["budget"]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
