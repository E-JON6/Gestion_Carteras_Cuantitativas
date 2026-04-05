"""
Analisis de la cartera en vivo desde el inicio de la estrategia hasta hoy.

Reconstruye posiciones y P&L acumulando los archivos Excel
``results/operaciones_rebalanceo_YYYY-MM-DD.xlsx`` (mismas ordenes que el
registrador genero para el bróker). Calcula metricas, costes estimados y
comparacion con benchmarks (SPY, MSCI World via URTH).

Salidas (por defecto en ``results/informe_cartera_vivo/``):
  - ``informe_cartera_vivo_{hoy}.xlsx`` (resumen, metricas, serie, historial de archivos)
  - ``nav_cartera_mtm_completo.csv``, ``patrimonio_vs_benchmark.csv``
  - ``detalle_operaciones_historico.csv``
  - Grafico PNG si matplotlib esta disponible

Opcional en config: ``PORTFOLIO_LIVE_INITIAL_POSITIONS`` (Excel con posiciones
antes del primer archivo de operaciones), ``PORTFOLIO_LIVE_START_DATE``.

Uso:
    python main_portfolio_backtest.py
"""

from __future__ import annotations

import argparse

from portfolio.live_portfolio_report import run_live_portfolio_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe cartera viva desde operaciones_rebalanceo_*.xlsx")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Carpeta con los Excel de operaciones (default: config PORTFOLIO_LIVE_RESULTS_DIR)",
    )
    parser.add_argument("--start", default=None, help="Fecha inicio YYYY-MM-DD (opcional)")
    parser.add_argument("--end", default=None, help="Fecha fin YYYY-MM-DD (default: hoy)")
    parser.add_argument(
        "--initial-positions",
        default=None,
        help="Excel con posiciones antes del primer archivo de operaciones (opcional)",
    )
    args = parser.parse_args()

    r = run_live_portfolio_report(
        results_dir=args.results_dir,
        start_date=args.start,
        end_date=args.end,
        initial_positions_path=args.initial_positions,
    )

    print("\n=== Informe cartera viva ===\n")
    print(f"Directorio informe: {r['output_dir']}")
    print(f"Excel principal:    {r['excel_path']}")
    print(f"Archivos operaciones leidos: {r['operaciones_files']}")
    print(f"Coste comisiones estimado (acum.): {r['coste_total_estimado_eur']:,.2f} EUR")
    print(f"Notional acumulado (|orden|):     {r['notional_acumulado_eur']:,.2f} EUR")
    m = r["metrics_formatted"]
    print("\nMetricas cartera (serie patrimonio EUR):")
    for k, v in m.items():
        print(f"  {k}: {v}")
    if r.get("plot_path"):
        print(f"\nGrafico: {r['plot_path']}")
    print("\nListo. Conserva la carpeta de informes y el CSV historial_ejecuciones para auditoria.\n")


if __name__ == "__main__":
    main()
