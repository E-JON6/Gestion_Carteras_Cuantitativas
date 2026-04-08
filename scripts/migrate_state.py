#!/usr/bin/env python3
"""One-time migration script: transfer state from the root project to daily_trading_strat_imp.

Reads the current portfolio state from the root project's outputs/state/,
liquidates all existing positions at current market prices, and creates
a fresh starting state for the improved strategy.

Usage:
    python scripts/migrate_state.py --source ../outputs/state --universe jaime [--date 2026-04-08]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.io import IO, YFinanceProvider
from src.io.portfolio_state import PortfolioStateManager
from src.io.vl_tracker import VLTracker

_MADRID_TZ = ZoneInfo("Europe/Madrid")


def main():
    parser = argparse.ArgumentParser(description="Migrate state from root project")
    parser.add_argument("--source", required=True, help="Path to source state dir (e.g. ../outputs/state)")
    parser.add_argument("--universe", required=True, help="Universe name")
    parser.add_argument("--dest", default="outputs/state", help="Destination state dir")
    parser.add_argument("--date", default=None, help="Migration date (YYYY-MM-DD)")
    parser.add_argument("--group", default="Grupo4", help="Group name")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.date:
        date = pd.Timestamp(args.date)
    else:
        date = pd.Timestamp(datetime.now(_MADRID_TZ).strftime("%Y-%m-%d"))

    # 1. Read source estado.json
    estado_path = source / "estado.json"
    if not estado_path.exists():
        print(f"ERROR: No estado.json found at {estado_path}")
        sys.exit(1)

    estado = json.loads(estado_path.read_text())
    print(f"Source strategy: {estado.get('estrategia', 'unknown')}")
    print(f"Source VL: €{estado.get('valor_cartera', 0):,.2f}")
    print(f"Source positions: {estado.get('posiciones', {})}")

    # 2. Copy VL history for continuity
    vl_source = source / "vl_history.csv"
    if vl_source.exists():
        shutil.copy2(vl_source, dest / "vl_history.csv")
        print(f"Copied VL history ({vl_source})")
    else:
        print("WARNING: No vl_history.csv found — starting fresh VL history")

    # 3. Copy historial Excel for continuity
    historial_source = source / f"Historial_{args.group}.xlsx"
    if historial_source.exists():
        shutil.copy2(historial_source, dest / f"Historial_{args.group}.xlsx")
        print(f"Copied historial Excel")

    # 4. Create fresh estado.json with cash = previous VL
    # (all positions liquidated, starting fresh)
    valor_cartera = estado.get("valor_cartera", 10_000_000.0)
    capital_inicial = estado.get("capital_inicial", 10_000_000.0)
    prev_costes = estado.get("costes_acumulados", 0.0)
    prev_n_ops = estado.get("n_operaciones", 0)

    new_estado = {
        "iniciado": True,
        "estrategia": "pending_first_run",
        "fecha_inicio": estado.get("fecha_inicio", date.strftime("%Y-%m-%d")),
        "capital_inicial": capital_inicial,
        "valor_cartera": valor_cartera,
        "posiciones": {},
        "cash": valor_cartera,
        "peak_valor": max(estado.get("peak_valor", 0.0), valor_cartera),
        "n_operaciones": prev_n_ops,
        "costes_acumulados": prev_costes,
        "fecha_ultima_ejecucion": date.strftime("%Y-%m-%d"),
    }

    (dest / "estado.json").write_text(json.dumps(new_estado, indent=4))
    print(f"\nNew estado.json created:")
    print(f"  Cash (= previous VL): €{valor_cartera:,.2f}")
    print(f"  Positions: none (liquidated)")
    print(f"  Costes acum. inherited: €{prev_costes:,.2f}")
    print(f"  N ops inherited: {prev_n_ops}")

    # 5. Create portfolio_state.json with cash only
    portfolio_state = {
        "current": {
            "date": date.isoformat(),
            "cash": valor_cartera,
            "positions": {},
            "total_value": valor_cartera,
        },
        "history": [],
        "positions_history": [],
        "operations_history": [],
    }

    # Inherit history from source if available
    source_state_path = source / "portfolio_state.json"
    if source_state_path.exists():
        source_state = json.loads(source_state_path.read_text())
        portfolio_state["history"] = source_state.get("history", [])
        portfolio_state["positions_history"] = source_state.get("positions_history", [])
        portfolio_state["operations_history"] = source_state.get("operations_history", [])
        print(f"  Inherited {len(portfolio_state['history'])} VL history entries")
        print(f"  Inherited {len(portfolio_state['operations_history'])} operations")

    (dest / "portfolio_state.json").write_text(
        json.dumps(portfolio_state, indent=2, default=str)
    )

    print(f"\nMigration complete. Run the daily script to start the new strategy:")
    print(f"  python scripts/run_daily.py --universe {args.universe} --strategy merton_custom --no-email")


if __name__ == "__main__":
    main()
