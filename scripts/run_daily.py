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


# ── Formatting helpers ─────────────────────────────────────────────────

_G = "\033[32m"   # green
_R = "\033[31m"   # red
_Y = "\033[33m"   # yellow
_C = "\033[36m"   # cyan
_B = "\033[1m"    # bold
_D = "\033[2m"    # dim
_0 = "\033[0m"    # reset


def _eur(v: float) -> str:
    return f"\u20ac{v:>14,.2f}"


def _pct(v: float) -> str:
    color = _G if v >= 0 else _R
    return f"{color}{v:+.4%}{_0}"


def _section(title: str) -> None:
    print(f"\n{_B}{_C}{'=' * 60}{_0}")
    print(f"{_B}{_C}  {title}{_0}")
    print(f"{_B}{_C}{'=' * 60}{_0}")


def _kv(label: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{_D}{label:<24}{_0} {value}")


# ── Strategy factory ───────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────

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

    # ── Setup ──────────────────────────────────────────────────────

    io = IO()
    universe = io.load_universe(args.universe)
    strategy = build_strategy(args.strategy, universe, args.defensive)

    if args.date:
        date = pd.Timestamp(args.date)
    else:
        date = pd.Timestamp(datetime.now(_MADRID_TZ).strftime("%Y-%m-%d"))

    _section("CONFIGURACION")
    _kv("Fecha", date.strftime("%Y-%m-%d (%A)"))
    _kv("Universo", f"{args.universe} ({len(universe)} activos)")
    _kv("Estrategia", strategy.name)
    _kv("Capital inicial", _eur(args.cash))

    # ── Execute ────────────────────────────────────────────────────

    runner = DailyRunner(
        strategy=strategy,
        universe=universe,
        provider=YFinanceProvider(),
        initial_cash=args.cash,
        state_dir=f"{args.output_dir}/state",
        operativa_dir=f"{args.output_dir}/operativa",
    )

    _section("EJECUCION")
    print(f"  Descargando precios y ejecutando estrategia...")

    result = runner.run_today(date)

    if "error" in result:
        print(f"\n  {_R}{_B}ERROR: {result['error']}{_0}")
        sys.exit(1)

    # ── Migration info ─────────────────────────────────────────────

    if result.get("migration_trades"):
        _section("TRANSICION DE ESTRATEGIA")
        print(f"  {_Y}Liquidando posiciones de la estrategia anterior (E4){_0}\n")
        for t in result["migration_trades"]:
            direction = "VENTA"
            print(f"    {_R}{direction}{_0}  {t.ticker:<10}  "
                  f"{abs(t.shares):>12,.2f} shares  @  {_eur(t.price).strip()}  "
                  f"{_D}(coste: {_eur(t.cost).strip()}){_0}")
        cash_after = sum(
            abs(t.shares) * t.price - t.cost for t in result["migration_trades"]
        )
        print(f"\n  {_D}Cash tras liquidacion:{_0} {_eur(cash_after)}")

    # ── Valor Liquidativo ──────────────────────────────────────────

    _section("VALOR LIQUIDATIVO")

    pre = result["pre_trade_vl"]
    post = result["post_trade_vl"]
    pnl = post - pre
    pnl_color = _G if pnl >= 0 else _R
    arrow = "\u25b2" if pnl >= 0 else "\u25bc"

    _kv("VL pre-trade", _eur(pre))
    _kv("VL post-trade", f"{_B}{_eur(post)}{_0}")
    _kv("P&L del dia", f"{pnl_color}{arrow} {_eur(pnl).strip()}{_0}")
    _kv("Cash disponible", _eur(result["cash"]))
    _kv("Retorno acumulado", _pct(post / args.cash - 1))

    # ── Trades ─────────────────────────────────────────────────────

    strategy_trades = [
        t for t in result.get("trades", [])
        if t not in result.get("migration_trades", [])
    ]

    _section(f"OPERATIVA  ({len(strategy_trades)} trades)")

    if strategy_trades:
        print(f"\n  {'Ticker':<10}  {'Dir':>6}  {'Shares':>14}  {'Precio':>12}  {'Valor':>14}  {'Coste':>10}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*10}")
        for t in strategy_trades:
            d = "COMPRA" if t.shares > 0 else "VENTA"
            dc = _G if t.shares > 0 else _R
            value = abs(t.shares * t.price)
            print(f"  {t.ticker:<10}  {dc}{d:>6}{_0}  {abs(t.shares):>14,.2f}  "
                  f"{_eur(t.price).strip():>12}  {_eur(value).strip():>14}  "
                  f"{_D}{_eur(t.cost).strip():>10}{_0}")
        total_cost = sum(t.cost for t in strategy_trades)
        total_value = sum(abs(t.shares * t.price) for t in strategy_trades)
        print(f"  {'-'*10}  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*10}")
        print(f"  {'TOTAL':<10}  {'':>6}  {'':>14}  {'':>12}  "
              f"{_eur(total_value).strip():>14}  {_Y}{_eur(total_cost).strip():>10}{_0}")
    else:
        print(f"\n  {_D}Sin trades de estrategia hoy (MANTENER){_0}")

    # ── Posiciones ─────────────────────────────────────────────────

    positions = {t: s for t, s in result["positions"].items() if abs(s) > 1e-9}
    today_prices = result["today_prices"]

    _section(f"CARTERA  ({len(positions)} posiciones)")

    if positions:
        print(f"\n  {'Ticker':<10}  {'Shares':>14}  {'Precio':>12}  {'Valor':>14}  {'Peso':>8}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*8}")
        total_pos_value = 0.0
        for ticker in sorted(positions, key=lambda t: -abs(positions[t] * today_prices.get_price(t))):
            shares = positions[ticker]
            price = today_prices.get_price(ticker)
            value = shares * price
            weight = value / post if post else 0
            total_pos_value += value
            w_color = _R if abs(weight) < 0.01 else _0
            print(f"  {ticker:<10}  {shares:>14,.2f}  "
                  f"{_eur(price).strip():>12}  {_eur(value).strip():>14}  "
                  f"{w_color}{weight:>7.2%}{_0}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*8}")
        leverage = total_pos_value / post if post else 0
        _kv("Posiciones", _eur(total_pos_value), indent=2)
        _kv("Cash", _eur(result["cash"]), indent=2)
        _kv("Apalancamiento", f"{leverage:.1%}", indent=2)

    # ── Metricas ───────────────────────────────────────────────────

    metrics = runner.vl_tracker.metrics()
    if metrics:
        _section("METRICAS DE RENDIMIENTO")
        _kv("Retorno total", _pct(metrics.get("total_return", 0)))
        _kv("Retorno anualizado", _pct(metrics.get("annualized_return", 0)))
        _kv("Volatilidad anualizada", f"{metrics.get('annualized_volatility', 0):.4%}")
        _kv("Sharpe ratio", f"{metrics.get('sharpe_ratio', 0):+.4f}")
        _kv("Max drawdown", f"{_R}{metrics.get('max_drawdown', 0):.4%}{_0}")
        _kv("Calmar ratio", f"{metrics.get('calmar_ratio', 0):+.4f}")

    # ── Acumulados ─────────────────────────────────────────────────

    _section("ACUMULADOS")
    _kv("Operaciones totales", f"{result.get('n_operaciones', 0)}")
    _kv("Costes totales", f"{_Y}{_eur(result.get('costes_acumulados', 0))}{_0}")

    # ── Report generation ──────────────────────────────────────────

    _section("REPORTES")
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

    _kv("Dashboard HTML", str(report_path))
    _kv("Latest", f"{args.output_dir}/reports/latest.html")
    _kv("Seguimiento Excel", str(seguimiento_path))
    if result["excel_path"]:
        _kv("Operativa Excel", result["excel_path"])
    _kv("Historial Excel", result.get("historial_path", "N/A"))

    # ── Emails ─────────────────────────────────────────────────────

    if args.no_email:
        _section("EMAILS (deshabilitados)")
        print(f"  {_D}Se omite el envio de emails (--no-email){_0}")
    else:
        _section("EMAILS")
        try:
            sender = EmailSender.from_env()

            if sender.is_test_mode:
                print(f"  {_Y}MODO PRUEBA{_0} — afi y summary van al mismo destinatario")
                print(f"  {_D}No se envia operativa a Afi real{_0}")
                print(f"  {_D}Destinatario: {sender.summary_recipient}{_0}")
            else:
                print(f"  {_D}Afi:     {sender.afi_recipient}{_0}")
                print(f"  {_D}Summary: {sender.summary_recipient}{_0}")

            print()
            email_results = sender.send_all(
                result=result,
                vl_tracker=runner.vl_tracker,
                state_mgr=runner.state_manager,
                universe=universe,
                report_path=Path(report_path),
                seguimiento_path=seguimiento_path,
            )

            for label, ok in email_results.items():
                if label == "operativa" and sender.is_test_mode:
                    print(f"  {_D}Operativa (Afi):{_0}  {_Y}omitido (modo prueba){_0}")
                elif ok:
                    print(f"  {_D}{label.capitalize()}:{_0}       {_G}enviado OK{_0}")
                else:
                    print(f"  {_D}{label.capitalize()}:{_0}       {_R}FALLO{_0}")

        except Exception as e:
            print(f"  {_R}Error configurando emails: {e}{_0}")

    # ── Done ───────────────────────────────────────────────────────

    print(f"\n{_G}{_B}{'=' * 60}{_0}")
    decision = "TRANSICION" if result.get("migration_trades") else (
        "REBALANCEO" if strategy_trades else "MANTENER"
    )
    print(f"{_G}{_B}  COMPLETADO  |  {result['date'].strftime('%Y-%m-%d')}  |  "
          f"{decision}  |  VL: {_eur(post).strip()}{_0}")
    print(f"{_G}{_B}{'=' * 60}{_0}\n")


if __name__ == "__main__":
    main()
