"""
Script principal: backtesting de la estrategia E2 DN+Momentum
optimizada mediante Walk-Forward Analysis.

Parámetros óptimos obtenidos por WF (score combinado Sharpe+Calmar+Consistencia):
    momentum_lookback = 252 días  (12 meses — Jegadeesh & Titman 1993)
    momentum_skip     = 21 días   (excluir último mes — reversión a corto plazo)
    alpha_min         = 0.10      (posición mínima defensiva)
    Score WF          = 0.90 | Consistencia = 100% (positivo en los 4 folds)

Estrategias comparadas:
    - Buy & Hold    : referencia pasiva (100% en el ETF)
    - Benchmark     : Merton puro (rebalanceo semanal sin bandas DN)
    - E2 Momentum   : Davis-Norman + señal momentum con parámetros óptimos WF

Fases:
    - IN-SAMPLE  2010-2024 : validación retrospectiva con parámetros fijados
    - OOS        2025      : test final con datos nunca vistos en el WF
    - SIMULACIÓN Feb-Mar 2026 : datos reales de mercado

Resultados guardados en: resultados/momentum_strategy/

Uso:
    python -m gestion_cuantitativa.main_momentum
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestion_cuantitativa.config import (
    BACKTEST_START, BACKTEST_END, OOS_START, OOS_END,
    SIMULATION_START, SIMULATION_END,
    INITIAL_WEALTH, ETF_CONFIGS, TRADING_DAYS_PER_YEAR,
    DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_M
)
from gestion_cuantitativa.data.download import download_all_data
from gestion_cuantitativa.backtest.engine import BacktestEngine
from gestion_cuantitativa.strategies.benchmark_merton import BenchmarkMerton
from gestion_cuantitativa.strategies.e2_dn_momentum import E2_DN_Momentum
from gestion_cuantitativa.strategies.buyhold import BuyAndHold
from gestion_cuantitativa.analysis.metrics import compute_metrics, compare_strategies
from gestion_cuantitativa.analysis.plots import generate_all_plots

# Parámetros óptimos obtenidos por Walk-Forward Analysis
# Score combinado: 0.40×Sharpe + 0.30×Calmar + 0.20×Consistencia + 0.10×(-Vol)
# Resultado: score=0.90, consistencia=100%, Sharpe medio validación=0.782
WF_LOOKBACK  = 252   # días
WF_SKIP      = 21    # días
WF_ALPHA_MIN = 0.10


def compute_effective_backtest_start(risky_data, backtest_start, warmup_days=252):
    """
    Calcula la fecha efectiva de inicio del backtest, asegurando que hay
    suficientes datos de warm-up para los estimadores rolling.

    Args:
        risky_data: pd.DataFrame con datos del ETF
        backtest_start: str, fecha deseada de inicio
        warmup_days: int, días mínimos de warm-up necesarios

    Returns:
        str, fecha efectiva de inicio
    """
    first_data_date = risky_data.index[0]
    desired_start = pd.Timestamp(backtest_start)

    # Necesitamos warmup_days de datos antes del backtest_start
    min_start = risky_data.index[warmup_days] if len(risky_data) > warmup_days else risky_data.index[-1]

    if min_start > desired_start:
        effective_start = min_start.strftime('%Y-%m-%d')
        print(f"  ⚠ Datos insuficientes para warm-up desde {backtest_start}")
        print(f"    Primer dato: {first_data_date.date()}, warm-up={warmup_days}d")
        print(f"    → Backtest ajustado a: {effective_start}")
        return effective_start

    return backtest_start


def run_backtest_for_etf(ticker, etf_config, base_output_dir,
                         start_date=None, end_date=None,
                         period_label=None, data=None):
    """
    Ejecuta el backtesting completo para un ETF dado.

    Args:
        ticker: str, ticker del ETF (e.g., "MSE.PA")
        etf_config: dict con 'name', 'description', 'fallback_tickers'
        base_output_dir: str, directorio base para resultados
        start_date: str, inicio del backtest (default: BACKTEST_START)
        end_date: str, fin del backtest (default: BACKTEST_END)
        period_label: str, etiqueta del periodo (e.g., "OOS 2025")
        data: dict pre-descargado (opcional, evita re-descargar)

    Returns:
        tuple (results, comparison, etf_name, effective_start, data)
    """
    start_date = start_date or BACKTEST_START
    end_date = end_date or BACKTEST_END
    period_label = period_label or "BACKTEST"
    etf_name = etf_config['name']

    print("\n" + "=" * 70)
    print(f"  {period_label}: {etf_name} ({ticker})")
    print(f"  {etf_config['description']}")
    print(f"  Periodo: {start_date} → {end_date}")
    print("=" * 70)

    # ============================================================
    # 1. DESCARGA DE DATOS (o reutilizar si ya se descargaron)
    # ============================================================
    if data is None:
        print(f"\n[1/5] Descargando datos para {ticker}...")
        data = download_all_data(ticker=ticker)
    else:
        print(f"\n[1/5] Reutilizando datos ya descargados para {ticker}")

    risky_data = data['risky']
    risk_free = data['risk_free']
    vix = data['vix']
    actual_ticker = data.get('actual_ticker', ticker)

    # ============================================================
    # 2. AJUSTAR PERIODO Y COSTES
    # ============================================================
    effective_start = compute_effective_backtest_start(risky_data, start_date)

    print("\n[2/5] Costes de transacción...")
    lambda_L = etf_config.get('lambda_L', DEFAULT_LAMBDA_L)
    lambda_M = etf_config.get('lambda_M', DEFAULT_LAMBDA_M)
    print(f"  λ_L (compra) = {lambda_L:.5f} ({lambda_L*100:.3f}%)")
    print(f"  λ_M (venta)  = {lambda_M:.5f} ({lambda_M*100:.3f}%)")
    print(f"  Fuente: config.py (manual)")

    # ============================================================
    # 3. DEFINIR ESTRATEGIAS
    # ============================================================
    print("\n[3/5] Inicializando estrategias...")
    strategies = [
        BuyAndHold(etf_name=etf_name),
        BenchmarkMerton(),
        E2_DN_Momentum(
            momentum_lookback = WF_LOOKBACK,
            momentum_skip     = WF_SKIP,
            alpha_min         = WF_ALPHA_MIN,
        ),
    ]

    for s in strategies:
        print(f"  ✓ {s.name}")

    # ============================================================
    # 4. EJECUTAR BACKTESTS
    # ============================================================
    print(f"\n[4/5] Ejecutando backtests ({effective_start} → {end_date})...")
    print(f"  Patrimonio inicial: {INITIAL_WEALTH:,.0f} EUR")
    print(f"  Costes: λ_L={lambda_L:.4f}, λ_M={lambda_M:.4f}")

    # Calcular tasa libre de riesgo media para Sharpe
    avg_rf = risk_free.loc[effective_start:end_date].mean()
    if np.isnan(avg_rf):
        avg_rf = 0.01
    print(f"  Tasa libre de riesgo media: {avg_rf*100:.2f}%")

    results = []
    for strategy in strategies:
        print(f"\n  → {strategy.name}...", end=" ", flush=True)

        engine = BacktestEngine(
            risky_data=risky_data,
            risk_free_series=risk_free,
            vix_series=vix,
            start_date=effective_start,
            end_date=end_date,
            lambda_L=lambda_L,
            lambda_M=lambda_M,
            initial_wealth=INITIAL_WEALTH,
            sigma_method='ewma'
        )

        result = engine.run(strategy)
        results.append(result)

        # Resumen rápido
        metrics = compute_metrics(result, avg_risk_free=avg_rf)
        print(f"CAGR={metrics['CAGR (%)']:.1f}%, "
              f"Sharpe={metrics['Sharpe Ratio']:.2f}, "
              f"MaxDD={metrics['Max Drawdown (%)']:.1f}%, "
              f"Trades={metrics['Nº Operaciones']}")

    # ============================================================
    # 5. COMPARATIVA DE RESULTADOS
    # ============================================================
    print("\n[5/5] Generando comparativa...")

    # Tabla de métricas
    comparison = compare_strategies(results, avg_risk_free=avg_rf)
    print("\n" + "=" * 70)
    print(f"  RESULTADOS {period_label}: {etf_name} ({actual_ticker})")
    print("=" * 70)
    print()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(comparison.T.to_string())

    # Guardar resultados por ETF
    safe_name = etf_name.replace(" ", "_").replace("&", "and")
    output_dir = os.path.join(base_output_dir, safe_name)
    os.makedirs(output_dir, exist_ok=True)

    comparison.T.to_csv(os.path.join(output_dir, 'metricas_comparativas.csv'))
    print(f"\n  Tabla guardada en: {output_dir}/metricas_comparativas.csv")

    # Gráficas
    print("\n  Generando gráficas...")
    etf_prices = risky_data['Close']
    figs = generate_all_plots(results, save_dir=output_dir, etf_name=etf_name,
                               prices=etf_prices)
    print(f"  Gráficas guardadas en: {output_dir}/")

    # ============================================================
    # ANÁLISIS COMPARATIVO
    # ============================================================
    print("\n" + "-" * 70)
    print(f"  ANÁLISIS {period_label}: {etf_name}")
    print(f"  Parámetros E2: lookback={WF_LOOKBACK}d / skip={WF_SKIP}d / alpha_min={WF_ALPHA_MIN}")
    print("-" * 70)

    m_buyhold   = compute_metrics(results[0], avg_risk_free=avg_rf)
    m_benchmark = compute_metrics(results[1], avg_risk_free=avg_rf)
    m_e2        = compute_metrics(results[2], avg_risk_free=avg_rf)

    # E2 vs Buy & Hold
    print(f"\n  1) E2 Momentum vs Buy & Hold:")
    print(f"     CAGR  : {m_buyhold['CAGR (%)']:+.2f}%  →  {m_e2['CAGR (%)']:+.2f}%  "
          f"({m_e2['CAGR (%)'] - m_buyhold['CAGR (%)']:+.2f}%)")
    print(f"     Sharpe: {m_buyhold['Sharpe Ratio']:.3f}  →  {m_e2['Sharpe Ratio']:.3f}  "
          f"({m_e2['Sharpe Ratio'] - m_buyhold['Sharpe Ratio']:+.3f})")
    print(f"     MaxDD : {m_buyhold['Max Drawdown (%)']:.1f}%  →  {m_e2['Max Drawdown (%)']:.1f}%  "
          f"({m_e2['Max Drawdown (%)'] - m_buyhold['Max Drawdown (%)']:+.1f}%)")
    print(f"     Calmar: {m_buyhold['Calmar Ratio']:.3f}  →  {m_e2['Calmar Ratio']:.3f}")
    print(f"     Costes: {m_buyhold['Costes (% patrimonio)']:.2f}%  →  "
          f"{m_e2['Costes (% patrimonio)']:.2f}%")

    # E2 vs Benchmark Merton
    print(f"\n  2) E2 Momentum vs Benchmark Merton:")
    print(f"     CAGR  : {m_benchmark['CAGR (%)']:+.2f}%  →  {m_e2['CAGR (%)']:+.2f}%  "
          f"({m_e2['CAGR (%)'] - m_benchmark['CAGR (%)']:+.2f}%)")
    print(f"     Sharpe: {m_benchmark['Sharpe Ratio']:.3f}  →  {m_e2['Sharpe Ratio']:.3f}  "
          f"({m_e2['Sharpe Ratio'] - m_benchmark['Sharpe Ratio']:+.3f})")
    print(f"     MaxDD : {m_benchmark['Max Drawdown (%)']:.1f}%  →  {m_e2['Max Drawdown (%)']:.1f}%  "
          f"({m_e2['Max Drawdown (%)'] - m_benchmark['Max Drawdown (%)']:+.1f}%)")
    print(f"     Calmar: {m_benchmark['Calmar Ratio']:.3f}  →  {m_e2['Calmar Ratio']:.3f}")
    print(f"     Trades: {m_benchmark['Nº Operaciones']}  →  {m_e2['Nº Operaciones']}  "
          f"({m_e2['Nº Operaciones'] - m_benchmark['Nº Operaciones']:+d})")
    print(f"     Costes: {m_benchmark['Costes (% patrimonio)']:.2f}%  →  "
          f"{m_e2['Costes (% patrimonio)']:.2f}%  "
          f"({m_e2['Costes (% patrimonio)'] - m_benchmark['Costes (% patrimonio)']:+.2f}%)")

    # Ranking
    print(f"\n  RANKING {period_label} ({etf_name}) — por Sharpe Ratio:")
    estrategias_ranked = sorted(
        [(m_e2,        f"E2 DN+Momentum (LK={WF_LOOKBACK}d/SK={WF_SKIP}d/AM={WF_ALPHA_MIN})"),
         (m_benchmark, "Benchmark Merton"),
         (m_buyhold,   f"Buy & Hold ({etf_name})")],
        key=lambda x: x[0]["Sharpe Ratio"], reverse=True
    )
    for rank, (m, name) in enumerate(estrategias_ranked, 1):
        print(f"     {rank}. {name:<50} "
              f"Sharpe={m['Sharpe Ratio']:.3f}  "
              f"CAGR={m['CAGR (%)']:+.2f}%  "
              f"MaxDD={m['Max Drawdown (%)']:.1f}%  "
              f"Calmar={m['Calmar Ratio']:.3f}")

    return results, comparison, etf_name, effective_start, data


def print_global_summary(all_results, period_label=""):
    """Imprime resumen global comparando ETFs para un periodo."""
    label = f" ({period_label})" if period_label else ""
    print("\n" + "=" * 70)
    print(f"  RESUMEN GLOBAL: COMPARACIÓN ENTRE ETFs{label}")
    print("=" * 70)

    for ticker, info in all_results.items():
        etf_name = info['etf_name']
        comp = info['comparison']
        comp_t = comp.T
        print(f"\n  {etf_name} ({ticker}), desde {info['effective_start']}:")

        for col in comp_t.columns:
            cagr = comp_t.loc['CAGR (%)', col]
            sharpe = comp_t.loc['Sharpe Ratio', col]
            maxdd = comp_t.loc['Max Drawdown (%)', col]
            print(f"    {col:<35} CAGR={cagr:+.2f}%  Sharpe={sharpe:.3f}  MaxDD={maxdd:.1f}%")


def main():
    print("=" * 70)
    print("  BACKTESTING E2 DN+MOMENTUM — Parámetros optimizados por Walk-Forward")
    print("  Máster en Finanzas Cuantitativas — AFI")
    print(f"  Parámetros WF: lookback={WF_LOOKBACK}d / skip={WF_SKIP}d / alpha_min={WF_ALPHA_MIN}")
    print("  ETFs: " + ", ".join(f"{c['name']} ({t})" for t, c in ETF_CONFIGS.items()))
    print("=" * 70)

    base_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'resultados', 'momentum_strategy'
    )

    # ============================================================
    # FASE 1: BACKTEST IN-SAMPLE (2010-2024)
    # ============================================================
    all_results_is = {}
    downloaded_data = {}  # Cache de datos descargados para reutilizar en OOS

    for ticker, etf_config in ETF_CONFIGS.items():
        results, comparison, etf_name, eff_start, data = run_backtest_for_etf(
            ticker, etf_config, base_output_dir,
            start_date=BACKTEST_START, end_date=BACKTEST_END,
            period_label="IN-SAMPLE"
        )
        all_results_is[ticker] = {
            'results': results,
            'comparison': comparison,
            'etf_name': etf_name,
            'effective_start': eff_start,
        }
        downloaded_data[ticker] = data  # Guardar para OOS

    print_global_summary(all_results_is, "In-Sample")

    # ============================================================
    # FASE 2: VALIDACIÓN OUT-OF-SAMPLE (2025)
    # ============================================================
    print("\n\n" + "#" * 70)
    print("#" * 70)
    print("  VALIDACIÓN OUT-OF-SAMPLE: {start} → {end}".format(
        start=OOS_START, end=OOS_END))
    print("  Los estimadores solo usan datos pasados → sin data leakage")
    print("#" * 70)
    print("#" * 70)

    oos_output_dir = os.path.join(base_output_dir, 'OOS_2025')

    all_results_oos = {}
    for ticker, etf_config in ETF_CONFIGS.items():
        safe_name = etf_config['name'].replace(" ", "_").replace("&", "and")
        etf_oos_dir = os.path.join(oos_output_dir, safe_name)

        results, comparison, etf_name, eff_start, _ = run_backtest_for_etf(
            ticker, etf_config, oos_output_dir,
            start_date=OOS_START, end_date=OOS_END,
            period_label="OUT-OF-SAMPLE 2025",
            data=downloaded_data[ticker]  # Reutilizar datos ya descargados
        )
        all_results_oos[ticker] = {
            'results': results,
            'comparison': comparison,
            'etf_name': etf_name,
            'effective_start': eff_start,
        }

    print_global_summary(all_results_oos, "Out-of-Sample 2025")

    # ============================================================
    # FASE 3: SIMULACIÓN 2 MESES (Feb-Mar 2026)
    # ============================================================
    print("\n\n" + "#" * 70)
    print("#" * 70)
    print("  SIMULACIÓN CON DATOS REALES: {start} → {end}".format(
        start=SIMULATION_START, end=SIMULATION_END))
    print("  Estimadores usan todo el histórico previo (2008-2026)")
    print("#" * 70)
    print("#" * 70)

    sim_output_dir = os.path.join(base_output_dir, 'SIM_2026')

    all_results_sim = {}
    for ticker, etf_config in ETF_CONFIGS.items():
        results, comparison, etf_name, eff_start, _ = run_backtest_for_etf(
            ticker, etf_config, sim_output_dir,
            start_date=SIMULATION_START, end_date=SIMULATION_END,
            period_label="SIMULACIÓN FEB-MAR 2026",
            data=downloaded_data[ticker]
        )
        all_results_sim[ticker] = {
            'results': results,
            'comparison': comparison,
            'etf_name': etf_name,
            'effective_start': eff_start,
        }

    print_global_summary(all_results_sim, "Simulación Feb-Mar 2026")

    # ============================================================
    # COMPARACIÓN IS vs OOS vs SIMULACIÓN
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  COMPARACIÓN IN-SAMPLE vs OOS vs SIMULACIÓN")
    print("=" * 70)

    for ticker in ETF_CONFIGS:
        etf_name = all_results_is[ticker]['etf_name']
        is_comp = all_results_is[ticker]['comparison'].T
        oos_comp = all_results_oos[ticker]['comparison'].T
        sim_comp = all_results_sim[ticker]['comparison'].T

        print(f"\n  {etf_name} ({ticker}):")
        print(f"  {'Estrategia':<35} {'Sharpe IS':>10} {'Sharpe OOS':>11} {'Sharpe SIM':>11} {'CAGR IS':>9} {'CAGR OOS':>10} {'CAGR SIM':>10}")
        print(f"  {'-' * 96}")

        for col in is_comp.columns:
            s_is = is_comp.loc['Sharpe Ratio', col]
            c_is = is_comp.loc['CAGR (%)', col]
            s_oos = oos_comp.loc['Sharpe Ratio', col] if col in oos_comp.columns else float('nan')
            c_oos = oos_comp.loc['CAGR (%)', col] if col in oos_comp.columns else float('nan')
            s_sim = sim_comp.loc['Sharpe Ratio', col] if col in sim_comp.columns else float('nan')
            c_sim = sim_comp.loc['CAGR (%)', col] if col in sim_comp.columns else float('nan')
            print(f"  {col:<35} {s_is:>10.3f} {s_oos:>11.3f} {s_sim:>11.3f} {c_is:>+8.2f}% {c_oos:>+9.2f}% {c_sim:>+9.2f}%")

    # ============================================================
    # CONCLUSIÓN
    # ============================================================
    print("\n" + "=" * 70)
    print("  BACKTEST + OOS + SIMULACIÓN COMPLETADOS")
    print("=" * 70)

    return all_results_is, all_results_oos, all_results_sim


if __name__ == "__main__":
    all_results_is, all_results_oos, all_results_sim = main()