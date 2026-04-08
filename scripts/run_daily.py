#!/usr/bin/env python3
"""Daily trading simulation runner.

Usage:
    python scripts/run_daily.py --universe jaime --strategy merton_custom
    python scripts/run_daily.py --universe jaime --strategy merton_custom --date 2026-03-12
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import pandas as pd

_MADRID_TZ = ZoneInfo("Europe/Madrid")

from src.io import IO, YFinanceProvider, ReportGenerator, EmailSender
from src.simulation import DailyRunner


def build_strategy(name: str, universe, defensive_ticker: str = "XEON.DE"):
    """Build a strategy by name."""
    if name == "merton_custom":
        from src.strategy.merton_custom import MertonCustomStrategy
        from src.models.estimators.mu import JamesSteinMean
        from src.models.estimators.covariance import LedoitWolfCovariance
        return MertonCustomStrategy(
            universe=universe,
            defensive_ticker=defensive_ticker,
            gamma=0.5,
            top_n=5,
            max_risky_fraction=0.8,
            mu_estimator=JamesSteinMean(),
            cov_estimator=LedoitWolfCovariance(),
            rebalance_every=21,
            band_scale=2.0,
        )
    elif name == "merton_full":
        from src.strategy.merton_full import MertonFullStrategy
        from src.models.estimators.mu import JamesSteinMean
        from src.models.estimators.covariance import LedoitWolfCovariance
        return MertonFullStrategy(
            universe=universe,
            defensive_ticker=defensive_ticker,
            gamma=0.5,
            mu_estimator=JamesSteinMean(),
            cov_estimator=LedoitWolfCovariance(),
        )
    elif name == "merton_mom":
        from src.strategy.merton_mom import MertonMomentumStrategy
        from src.models.estimators.mu import JamesSteinMean
        from src.models.estimators.covariance import LedoitWolfCovariance
        return MertonMomentumStrategy(
            universe=universe,
            defensive_ticker=defensive_ticker,
            gamma=0.3,
            mu_estimator=JamesSteinMean(),
            cov_estimator=LedoitWolfCovariance(),
            momentum_lookback=252,
            momentum_skip=21,
            momentum_alpha_min=0.3,
            use_dn_bands=True,
            band_scale=2.0,
            max_risky_fraction=1.0,
            rebalance_every=21,
        )
    elif name == "merton_dual_mom":
        from src.strategy.merton_dual_mom import MertonDualMomentumStrategy
        from src.models.estimators.mu import JamesSteinMean
        from src.models.estimators.covariance import LedoitWolfCovariance
        return MertonDualMomentumStrategy(
            universe=universe,
            defensive_ticker=defensive_ticker,
            gamma=0.5,
            mu_estimator=JamesSteinMean(),
            cov_estimator=LedoitWolfCovariance(),
        )
    elif name == "merton":
        from src.strategy.merton import MertonStrategy
        return MertonStrategy(
            universe=universe,
            defensive_ticker=defensive_ticker,
        )
    elif name == "buy_and_hold":
        from src.strategy.buy_and_hold import BuyAndHoldStrategy
        return BuyAndHoldStrategy(universe=universe)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run daily trading simulation")
    parser.add_argument("--universe", required=True, help="Universe name (e.g. jaime)")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g. merton_custom)")
    parser.add_argument("--date", default=None, help="Date to simulate (YYYY-MM-DD). Default: today")
    parser.add_argument("--defensive", default="XEON.DE", help="Defensive ticker")
    parser.add_argument("--cash", type=float, default=10_000_000, help="Initial cash")
    parser.add_argument("--output-dir", default="outputs", help="Output directory root")
    parser.add_argument("--no-email", action="store_true", help="Skip sending emails")
    args = parser.parse_args()

    # Load .env for SMTP credentials
    load_dotenv()

    # Load universe
    io = IO()
    universe = io.load_universe(args.universe)
    print(f"Universe: {args.universe} ({len(universe)} assets)")

    # Build strategy
    strategy = build_strategy(args.strategy, universe, args.defensive)
    print(f"Strategy: {strategy.name}")

    # Parse date — default: today in Madrid time
    if args.date:
        date = pd.Timestamp(args.date)
    else:
        date = pd.Timestamp(datetime.now(_MADRID_TZ).strftime("%Y-%m-%d"))
        print(f"No --date provided. Using Madrid date: {date.strftime('%Y-%m-%d')}")

    # Build runner
    runner = DailyRunner(
        strategy=strategy,
        universe=universe,
        provider=YFinanceProvider(),
        initial_cash=args.cash,
        state_dir=f"{args.output_dir}/state",
        operativa_dir=f"{args.output_dir}/operativa",
    )

    # Execute
    print(f"\nRunning for date: {date or 'today'}...")
    print("-" * 50)

    result = runner.run_today(date)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    # Print summary
    print(f"Date:           {result['date'].strftime('%Y-%m-%d')}")
    print(f"First day:      {result['is_first_day']}")
    print(f"Pre-trade VL:   {result['pre_trade_vl']:,.2f} EUR")
    print(f"Post-trade VL:  {result['post_trade_vl']:,.2f} EUR")
    print(f"Cash:           {result['cash']:,.2f} EUR")
    print(f"Signals:        {result['n_signals']}")
    print(f"Trades:         {result['n_trades']}")

    if result['excel_path']:
        print(f"\nOperativa Excel: {result['excel_path']}")
    else:
        print("\nNo trades today — no Excel generated.")

    print(f"Historial:      {result.get('historial_path', 'N/A')}")
    print(f"Costes acum.:   {result.get('costes_acumulados', 0):,.2f} EUR")
    print(f"N ops acum.:    {result.get('n_operaciones', 0)}")

    # Print positions
    positions = result['positions']
    if positions:
        print(f"\nPositions ({len(positions)}):")
        for ticker, shares in sorted(positions.items()):
            if shares != 0:
                print(f"  {ticker:12s}  {shares:>12,.2f} shares")

    # Print VL metrics if enough history
    metrics = runner.vl_tracker.metrics()
    if metrics:
        print(f"\nPerformance metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:25s}  {v:>10.4f}")

    # Generate HTML dashboard + seguimiento Excel
    print("\nGenerating report...")
    reporter = ReportGenerator(
        report_dir=f"{args.output_dir}/reports",
        initial_cash=args.cash,
    )
    report_path = reporter.generate(
        result=result,
        vl_tracker=runner.vl_tracker,
        state_mgr=runner.state_manager,
        universe=universe,
        strategy=strategy,
    )
    seguimiento_path = Path(f"{args.output_dir}/reports/seguimiento.xlsx")
    print(f"Report:          {report_path}")
    print(f"Seguimiento:     {seguimiento_path}")
    print(f"Latest:          {args.output_dir}/reports/latest.html")

    # Send emails
    if not args.no_email:
        print("\nSending emails...")
        try:
            sender = EmailSender.from_env()
            email_results = sender.send_all(
                result=result,
                vl_tracker=runner.vl_tracker,
                state_mgr=runner.state_manager,
                universe=universe,
                report_path=Path(report_path),
                seguimiento_path=seguimiento_path,
            )
            for name, ok in email_results.items():
                status = "sent" if ok else "FAILED"
                print(f"  Email ({name}): {status}")
        except Exception as e:
            print(f"  Email setup failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
