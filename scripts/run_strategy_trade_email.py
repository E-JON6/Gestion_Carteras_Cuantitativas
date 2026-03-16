"""Run a strategy to the latest market close and optionally email today's trade."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf


def _bootstrap_repo_root() -> Path:
    """Add the repo root to sys.path so local imports work from this script."""
    root = Path(__file__).resolve().parents[1]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root


REPO_ROOT = _bootstrap_repo_root()

from backtest.engine import BacktestEngine
from config import (
    DATA_START,
    DEFAULT_ETF_TICKER,
    DEFAULT_LAMBDA_L,
    DEFAULT_LAMBDA_M,
    ETF_CONFIGS,
    INITIAL_WEALTH,
    RISK_FREE_FRED,
    SIMULATION_START,
)
from strategies.benchmark_merton import BenchmarkMerton
from strategies.buyhold import BuyAndHold
from strategies.e1_dn_adaptive import E1_DN_Adaptive
from strategies.e2_dn_momentum import E2_DN_Momentum
from strategies.e3_dn_voltarget import E3_DN_VolTarget
from strategies.e4_dn_defensive import E4_DN_Defensive
from strategies.e5_dn_drawdown import E5_DN_Drawdown
from services.trade_email_service import send_daily_trade_email


@dataclass(frozen=True)
class StrategySelection:
    code: str
    label: str
    factory: Callable[[], object]


STRATEGIES: dict[str, StrategySelection] = {
    "BUYHOLD": StrategySelection("BUYHOLD", "Buy & Hold", lambda: BuyAndHold()),
    "BENCHMARK": StrategySelection("BENCHMARK", "Benchmark: Merton Puro", BenchmarkMerton),
    "E1": StrategySelection("E1", "E1: DN Adaptativo", E1_DN_Adaptive),
    "E2": StrategySelection("E2", "E2: DN + Momentum", E2_DN_Momentum),
    "E3": StrategySelection("E3", "E3: DN + Vol Targeting", E3_DN_VolTarget),
    "E4": StrategySelection("E4", "E4: DN + Regimen Defensivo", E4_DN_Defensive),
    "E5": StrategySelection("E5", "E5: DN + Drawdown Shield", E5_DN_Drawdown),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strategy to the latest close and email today's trade if confirmed."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_ETF_TICKER,
        help="ETF ticker to simulate. Default: %(default)s",
    )
    parser.add_argument(
        "--strategy",
        default="E4",
        choices=sorted(STRATEGIES.keys()),
        help="Strategy code to run. Default: %(default)s",
    )
    parser.add_argument(
        "--start-date",
        default=SIMULATION_START,
        help="Simulation start date in YYYY-MM-DD. Default: %(default)s",
    )
    parser.add_argument(
        "--initial-wealth",
        type=float,
        default=INITIAL_WEALTH,
        help="Initial wealth used by the simulation. Default: %(default)s",
    )
    parser.add_argument(
        "--template-path",
        default=str(REPO_ROOT / "docs" / "Operativa_GrupoX.xlsx"),
        help="Excel template path. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs"),
        help="Directory for generated workbooks. Default: %(default)s",
    )
    parser.add_argument(
        "--trade-mode",
        default="latest-date",
        choices=["latest-date", "latest-trade"],
        help=(
            "Choose today's trade only ('latest-date') or force a test path by "
            "using the most recent generated trade in the simulation ('latest-trade'). "
            "Default: %(default)s"
        ),
    )
    return parser.parse_args()


def _normalize_download_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel("Ticker")

    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def download_risky_data(ticker: str, start_date: str) -> pd.DataFrame:
    start = min(pd.Timestamp(DATA_START), pd.Timestamp(start_date)).strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, progress=False)
    data = _normalize_download_frame(data)
    if data.empty:
        raise RuntimeError(f"Could not download market data for ticker '{ticker}'")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(
            f"Downloaded data for '{ticker}' is missing columns: {', '.join(missing_columns)}"
        )

    return data[required_columns].dropna()


def download_risk_free_series(start_date: str) -> pd.Series:
    start = min(pd.Timestamp(DATA_START), pd.Timestamp(start_date)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={RISK_FREE_FRED}"

    try:
        df = pd.read_csv(url)
        date_column = None
        for candidate in ("DATE", "observation_date"):
            if candidate in df.columns:
                date_column = candidate
                break

        if date_column is None or RISK_FREE_FRED not in df.columns:
            raise ValueError("Unexpected FRED CSV format")

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df[RISK_FREE_FRED] = pd.to_numeric(df[RISK_FREE_FRED], errors="coerce")
        df = df.dropna(subset=[date_column, RISK_FREE_FRED])
        df = df[df[date_column] >= pd.Timestamp(start)]
        series = df.set_index(date_column)[RISK_FREE_FRED] / 100.0
        series.index = pd.to_datetime(series.index).tz_localize(None)

        if series.empty:
            raise ValueError("FRED returned no valid risk-free rows")

        return series.rename("risk_free_rate")
    except Exception as exc:
        raise RuntimeError(f"Failed to download {RISK_FREE_FRED} from FRED: {exc}") from exc


def build_strategy(code: str):
    selection = STRATEGIES[code]
    return selection.factory()


def run_strategy_to_latest(
    ticker: str,
    strategy_code: str,
    start_date: str,
    initial_wealth: float,
):
    risky_data = download_risky_data(ticker, start_date)
    latest_close_date = risky_data.index[-1]
    risk_free = download_risk_free_series(start_date)
    strategy = build_strategy(strategy_code)

    etf_config = ETF_CONFIGS.get(ticker, {})
    lambda_l = etf_config.get("lambda_L")
    lambda_m = etf_config.get("lambda_M")
    lambda_l = DEFAULT_LAMBDA_L if lambda_l is None else lambda_l
    lambda_m = DEFAULT_LAMBDA_M if lambda_m is None else lambda_m

    engine = BacktestEngine(
        risky_data=risky_data,
        risk_free_series=risk_free,
        start_date=start_date,
        end_date=latest_close_date.strftime("%Y-%m-%d"),
        lambda_L=lambda_l,
        lambda_M=lambda_m,
        initial_wealth=initial_wealth,
        sigma_method="ewma",
    )
    result = engine.run(strategy)
    return result, risky_data, latest_close_date, lambda_l, lambda_m


def _build_trade_instructions_for_date(
    trades_df: pd.DataFrame,
    risky_data: pd.DataFrame,
    trade_date,
    ct_buy: float,
    ct_sell: float,
):
    selected_trades = trades_df[trades_df["date"] == trade_date]
    if selected_trades.empty:
        return None

    price = float(risky_data.loc[trade_date, "Close"])
    instructions: list[dict[str, float | str]] = []

    for _, trade in selected_trades.iterrows():
        trade_type = trade["type"]
        delta = float(trade["delta"])
        ct = ct_buy if trade_type == "buy" else ct_sell
        executed_price = price * (1 + ct) if trade_type == "buy" else price * (1 - ct)
        quantity = delta / executed_price

        instructions.append(
            {
                "quantity": quantity,
                "price": price,
                "ct": ct,
            }
        )

    return instructions


def select_trade_instructions(
    result,
    risky_data: pd.DataFrame,
    latest_date,
    ct_buy: float,
    ct_sell: float,
    trade_mode: str,
):
    trades_df = result["trades"]
    if trades_df.empty:
        return None, None

    latest_date_instructions = _build_trade_instructions_for_date(
        trades_df,
        risky_data,
        latest_date,
        ct_buy,
        ct_sell,
    )
    if latest_date_instructions is not None:
        return latest_date, latest_date_instructions

    if trade_mode != "latest-trade":
        return None, None

    selected_trade_date = trades_df["date"].max()
    if pd.isna(selected_trade_date):
        return None, None

    selected_instructions = _build_trade_instructions_for_date(
        trades_df,
        risky_data,
        selected_trade_date,
        ct_buy,
        ct_sell,
    )
    return selected_trade_date, selected_instructions


def build_trade_rows(ticker: str, instructions: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for instruction in instructions:
        rows.append(
            {
                "id": ticker,
                "quantity": instruction["quantity"],
                "price": instruction["price"],
                "ct": instruction["ct"],
            }
        )
    return rows


def format_terminal_summary(
    ticker: str,
    strategy_code: str,
    result,
    risky_data: pd.DataFrame,
    latest_date,
    selected_trade_date,
    trade_mode: str,
    trade_rows: list[dict[str, float | str]] | None,
) -> str:
    history = result["history"]
    portfolio = result["portfolio"]
    latest_state = history.loc[latest_date]
    close_price = float(risky_data.loc[latest_date, "Close"])
    lines = [
        "",
        "=" * 70,
        f"Strategy run complete: {strategy_code} on {ticker}",
        f"Latest market date: {latest_date.date()}",
        f"Latest close: {close_price:.4f}",
        f"Portfolio wealth: {float(latest_state['wealth']):,.2f}",
        f"Risky allocation alpha: {float(latest_state['alpha']):.4f}",
        f"Total costs: {portfolio.total_costs:,.2f}",
    ]

    if trade_mode == "latest-trade" and selected_trade_date is not None and selected_trade_date != latest_date:
        lines.append(f"Email test mode uses most recent trade date: {selected_trade_date.date()}")

    if not trade_rows:
        lines.append("No trade was generated on the latest market date.")
    else:
        label = "Selected trade instruction(s):"
        if selected_trade_date is not None:
            label = f"Selected trade instruction(s) for {selected_trade_date.date()}:"
        lines.append(label)
        for index, trade in enumerate(trade_rows, start=1):
            lines.append(
                f"  {index}. id={trade['id']} quantity={float(trade['quantity']):.6f} "
                f"price={float(trade['price']):.4f} ct={float(trade['ct']):.6f}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)


def prompt_for_confirmation() -> bool:
    response = input("Send the email with the generated workbook? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def main() -> int:
    args = parse_args()

    try:
        result, risky_data, latest_date, lambda_l, lambda_m = run_strategy_to_latest(
            ticker=args.ticker,
            strategy_code=args.strategy,
            start_date=args.start_date,
            initial_wealth=args.initial_wealth,
        )
        selected_trade_date, raw_instructions = select_trade_instructions(
            result,
            risky_data,
            latest_date,
            lambda_l,
            lambda_m,
            args.trade_mode,
        )
        trade_rows = build_trade_rows(args.ticker, raw_instructions) if raw_instructions else None
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        format_terminal_summary(
            args.ticker,
            args.strategy,
            result,
            risky_data,
            latest_date,
            selected_trade_date,
            args.trade_mode,
            trade_rows,
        )
    )

    if not trade_rows:
        return 0

    if not prompt_for_confirmation():
        print("Email cancelled.")
        return 0

    try:
        response = send_daily_trade_email(
            trade_rows,
            template_path=args.template_path,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print("ERROR sending email:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print("Email sent successfully.")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
