# P5 — Tu tarea: Backtesting, Métricas, Gráficos y Walk-Forward
### VERSIÓN DEFINITIVA
### Entrega: Miércoles por la noche

---

## Lo que tienes que hacer, en una frase

Cuando P4 avise que el backtest simple ha terminado, ejecutar el análisis
de métricas y gráficos. Después ejecutar el walk-forward para validar
que la estrategia no está sobreajustada. Ambas cosas el miércoles.

---

## Cuándo empieza tu trabajo

**Tu trabajo empieza cuando P4 escriba: "P4: ✓ BACKTEST COMPLETADO".**

Antes de eso, prepara todos tus scripts con datos sintéticos.
El miércoles tienes que ejecutar dos cosas en orden:
1. `python run_analysis.py` — análisis del backtest simple
2. `python walk_forward_bl.py` — validación walk-forward

El walk-forward tarda entre 1 y 3 horas dependiendo del ordenador.
**Empiézalo en cuanto P4 avise, no esperes al final del día.**

---

## Lo que recibes de P4

```
outputs/bl_omega/
├── wealth_history.csv   ← patrimonio diario + pesos de todos los ETFs + XEON.DE
└── trades.csv           ← registro de todas las órdenes ejecutadas
```

Y la función `run_backtest()` de `main_black_litterman.py` para llamarla
en bucle desde tu walk-forward.

---

## Lo que entregas

```
outputs/bl_omega/
├── metricas_comparativas.csv     ← tabla backtest simple vs Buy & Hold
├── metricas_comparativas.png     ← tabla visual para presentación
├── grafico_wealth.png            ← curvas de patrimonio
├── grafico_drawdown.png          ← drawdown histórico
├── grafico_pesos.png             ← composición cartera con XEON.DE
├── grafico_xeon.png              ← peso de XEON.DE a lo largo del tiempo

outputs/walk_forward/
├── resultados_wf.csv             ← Sharpe IS y OOS por ventana y combinación
├── mejores_params_por_ventana.csv← parámetros óptimos en cada ventana IS
├── wealth_oos_concatenado.csv    ← curva de patrimonio OOS concatenada
├── grafico_wf_wealth.png         ← backtest simple vs curva OOS
└── grafico_wf_sharpe.png         ← Sharpe IS vs OOS por ventana
```

---

## Archivo 1: `metrics_bl.py`

```python
"""
Métricas de evaluación para la estrategia BL-Omega.
Incluye métricas específicas de la gestión de XEON.DE.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(wealth_series, trades_df=None, history_df=None,
                    strategy_name="Estrategia", avg_risk_free=0.025,
                    initial_wealth=10_000_000, xeon_col=None):
    """
    Calcula todas las métricas de una estrategia.

    Args:
        wealth_series:  pd.Series con patrimonio día a día
        trades_df:      pd.DataFrame con los trades (opcional)
        history_df:     pd.DataFrame con historial completo (para métricas de XEON.DE)
        strategy_name:  nombre para la tabla
        avg_risk_free:  tasa BCE media del periodo
        initial_wealth: patrimonio inicial
        xeon_col:       nombre de la columna de peso de XEON.DE en history_df

    Returns:
        dict con todas las métricas
    """
    wealth   = wealth_series.dropna()
    n_days   = len(wealth)
    n_years  = n_days / TRADING_DAYS

    if n_years < 0.1:
        return {'Estrategia': strategy_name, 'Error': 'Muy pocos datos'}

    daily_returns = wealth.pct_change().dropna()
    total_return  = wealth.iloc[-1] / wealth.iloc[0]
    cagr          = total_return ** (1 / n_years) - 1
    vol           = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe        = (cagr - avg_risk_free) / vol if vol > 0 else 0.0

    rolling_max   = wealth.cummax()
    drawdown      = (wealth - rolling_max) / rolling_max
    max_drawdown  = drawdown.min()
    calmar        = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else 0.0

    neg_rets      = daily_returns[daily_returns < 0]
    downside_vol  = neg_rets.std() * np.sqrt(TRADING_DAYS)
    sortino       = (cagr - avg_risk_free) / downside_vol if downside_vol > 1e-6 else 0.0

    # Métricas de trades
    n_trades      = 0
    total_costs   = 0
    turnover_anual= 0

    if trades_df is not None and len(trades_df) > 0:
        n_trades = len(trades_df)
        if 'cost' in trades_df.columns:
            total_costs = trades_df['cost'].sum()
        if 'delta_eur' in trades_df.columns and n_years > 0:
            total_traded  = trades_df['delta_eur'].abs().sum()
            avg_wealth    = wealth.mean()
            turnover_anual= (total_traded / avg_wealth) / n_years if avg_wealth > 0 else 0

    # Métricas específicas de XEON.DE
    tiempo_en_riesgo  = float('nan')
    xeon_medio        = float('nan')
    dias_crisis       = float('nan')

    if history_df is not None and xeon_col and xeon_col in history_df.columns:
        xeon_weights     = history_df[xeon_col].dropna()
        xeon_medio       = float(xeon_weights.mean())
        # Tiempo con peso en riesgo alto (XEON.DE < 20%)
        tiempo_en_riesgo = float((xeon_weights < 0.20).mean())

    return {
        'Estrategia':            strategy_name,
        'CAGR (%)':              round(cagr * 100, 2),
        'Volatilidad (%)':       round(vol * 100, 2),
        'Sharpe Ratio':          round(sharpe, 3),
        'Sortino Ratio':         round(sortino, 3),
        'Max Drawdown (%)':      round(max_drawdown * 100, 2),
        'Calmar Ratio':          round(calmar, 3),
        'Nº Rebalanceos':        n_trades,
        'Turnover Anual':        round(turnover_anual, 3),
        'Costes (% patrimonio)': round(total_costs / initial_wealth * 100, 3),
        'XEON.DE medio (%)':     round(xeon_medio * 100, 1) if not np.isnan(xeon_medio) else 'N/A',
        'Tiempo max riesgo (%)': round(tiempo_en_riesgo * 100, 1) if not np.isnan(tiempo_en_riesgo) else 'N/A',
        'Retorno Total (%)':     round((total_return - 1) * 100, 1),
        'Riqueza Final (EUR)':   round(wealth.iloc[-1], 0),
    }


def compute_buyhold_wealth(prices_df, ticker, initial_wealth=10_000_000):
    """Buy & Hold en un ETF: comprar el día 1 y no vender nunca."""
    prices = prices_df[ticker].dropna()
    return initial_wealth * (prices / prices.iloc[0])


def compare_strategies(results_dict, avg_risk_free=0.025):
    """
    Tabla comparativa de todas las estrategias.

    Args:
        results_dict: dict {nombre: {'wealth': pd.Series, 'trades': df, 'history': df}}
    """
    rows = []
    for name, data in results_dict.items():
        m = compute_metrics(
            data['wealth'],
            trades_df  = data.get('trades'),
            history_df = data.get('history'),
            strategy_name = name,
            avg_risk_free = avg_risk_free,
            xeon_col   = data.get('xeon_col'),
        )
        rows.append(m)
    return pd.DataFrame(rows).set_index('Estrategia')
```

---

## Archivo 2: `plots_bl.py`

```python
"""
Gráficos para la estrategia BL-Omega.
Incluye gráfico específico del peso de XEON.DE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

COLORES = {
    'BL-Omega':    '#2196F3',
    'Buy & Hold':  '#4CAF50',
    'Merton Puro': '#FF9800',
    'XEON.DE':     '#9E9E9E',  # Gris para el activo monetario
}


def _color(name):
    for key, c in COLORES.items():
        if key.lower() in name.lower():
            return c
    return '#9C27B0'


def plot_wealth_curves(results_dict, save_path=None, title=None):
    """Curvas de evolución del patrimonio."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, data in results_dict.items():
        w = data['wealth']
        ax.plot(w.index, w.values / 1e6, label=name,
                color=_color(name), linewidth=2)
    ax.set_xlabel('Fecha'); ax.set_ylabel('Patrimonio (M EUR)')
    ax.set_title(title or 'Evolución del patrimonio'); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_drawdown(results_dict, save_path=None):
    """Drawdown histórico de todas las estrategias."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, data in results_dict.items():
        w  = data['wealth']
        dd = (w - w.cummax()) / w.cummax() * 100
        c  = _color(name)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=c, label=name)
        ax.plot(dd.index, dd.values, color=c, linewidth=1)
    ax.set_xlabel('Fecha'); ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown histórico'); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_weights_over_time(history_df, todos_tickers, xeon_ticker='XEON.DE',
                            save_path=None, top_n=6):
    """
    Área apilada con la composición de la cartera.
    XEON.DE aparece en gris en la parte superior para diferenciarlo.
    """
    weight_cols = [f'weight_{t}' for t in todos_tickers if f'weight_{t}' in history_df.columns]
    if not weight_cols:
        print("  [AVISO P5] No hay columnas de pesos en el historial")
        return None

    weights_df = history_df[weight_cols].copy()
    weights_df.columns = [c.replace('weight_', '') for c in weight_cols]

    # Separar XEON.DE del resto
    risk_cols = [c for c in weights_df.columns if c != xeon_ticker]
    mean_risk = weights_df[risk_cols].mean().sort_values(ascending=False)
    top_risk  = list(mean_risk.head(top_n).index)

    # Orden de la pila: ETFs de riesgo abajo, XEON.DE arriba
    plot_cols = top_risk + ([xeon_ticker] if xeon_ticker in weights_df.columns else [])
    data_plot = weights_df[plot_cols].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_risk))) if top_risk else []
    all_colors = list(colors) + ['#BDBDBD']  # Gris para XEON.DE

    ax.stackplot(data_plot.index, data_plot.T.values,
                 labels=plot_cols, colors=all_colors[:len(plot_cols)], alpha=0.8)

    ax.set_xlabel('Fecha'); ax.set_ylabel('Peso en cartera')
    ax.set_title('Composición de la cartera (gris = XEON.DE monetario)')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.set_ylim(0, 1); ax.grid(alpha=0.2, axis='y')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(); plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_xeon_weight(history_df, xeon_ticker='XEON.DE', save_path=None):
    """
    Gráfico específico del peso de XEON.DE a lo largo del tiempo.
    Muestra también los umbrales de régimen (20% y 30% de volatilidad).
    """
    xeon_col = f'weight_{xeon_ticker}'
    if xeon_col not in history_df.columns:
        print("  [AVISO P5] No hay columna de XEON.DE en el historial")
        return None

    xeon_w = history_df[xeon_col].dropna()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(xeon_w.index, xeon_w.values * 100, 0,
                    alpha=0.5, color='#9E9E9E', label='XEON.DE')
    ax.plot(xeon_w.index, xeon_w.values * 100, color='#616161', linewidth=1)

    # Líneas de referencia
    ax.axhline(30, color='#FF9800', linestyle='--', alpha=0.7,
               label='30% (régimen caution)')
    ax.axhline(60, color='#F44336', linestyle='--', alpha=0.7,
               label='60% (régimen crisis)')

    ax.set_xlabel('Fecha'); ax.set_ylabel('Peso XEON.DE (%)')
    ax.set_title('Peso del activo monetario (XEON.DE) a lo largo del tiempo')
    ax.legend(); ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(); plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_metrics_table(metrics_df, save_path=None, title=None):
    """Tabla visual de métricas (PNG para presentación)."""
    cols = ['CAGR (%)', 'Volatilidad (%)', 'Sharpe Ratio', 'Sortino Ratio',
            'Max Drawdown (%)', 'Calmar Ratio', 'Nº Rebalanceos',
            'XEON.DE medio (%)', 'Costes (% patrimonio)', 'Retorno Total (%)']
    cols_ok = [c for c in cols if c in metrics_df.columns]
    df_plot = metrics_df[cols_ok]

    fig, ax = plt.subplots(figsize=(16, max(3, len(df_plot) + 2)))
    ax.axis('off')
    table = ax.table(cellText=df_plot.values, colLabels=df_plot.columns,
                     rowLabels=df_plot.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.8)
    for j in range(len(df_plot.columns)):
        table[0, j].set_facecolor('#1F4E79')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(df_plot) + 1):
        c = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(-1, len(df_plot.columns)):
            if j >= 0: table[i, j].set_facecolor(c)
    ax.set_title(title or 'Métricas Comparativas', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
```

---

## Archivo 3: `run_analysis.py`

```python
"""
run_analysis.py — Análisis completo de resultados.
Ejecutar DESPUÉS de que P4 haya completado el backtest:
    python run_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from metrics_bl import compute_metrics, compare_strategies, compute_buyhold_wealth
from plots_bl import (plot_wealth_curves, plot_drawdown, plot_weights_over_time,
                       plot_xeon_weight, plot_metrics_table)

OUTPUT_DIR     = "outputs/bl_omega"
XEON_TICKER    = "XEON.DE"
INITIAL_WEALTH = 10_000_000


def main():
    print("=" * 60)
    print("ANÁLISIS DE RESULTADOS — BL-Omega")
    print("=" * 60)

    # 1) Cargar resultados
    wealth_file = os.path.join(OUTPUT_DIR, "wealth_history.csv")
    trades_file = os.path.join(OUTPUT_DIR, "trades.csv")

    if not os.path.exists(wealth_file):
        print(f"ERROR: {wealth_file} no existe.")
        print("Asegúrate de que P4 ha ejecutado main_black_litterman.py")
        return

    history_df = pd.read_csv(wealth_file, index_col=0, parse_dates=True)
    trades_df  = pd.read_csv(trades_file) if os.path.exists(trades_file) else pd.DataFrame()

    bl_wealth = history_df['wealth']
    xeon_col  = f'weight_{XEON_TICKER}'

    print(f"  Historial: {len(history_df)} días")
    print(f"  Trades: {len(trades_df)}")
    xeon_medio = history_df[xeon_col].mean() if xeon_col in history_df.columns else float('nan')
    print(f"  Peso medio XEON.DE: {xeon_medio:.1%}" if not np.isnan(xeon_medio) else "  XEON.DE no en historial")

    # 2) Buy & Hold de referencia
    prices_cache = "data/cache/universe_prices.csv"
    if os.path.exists(prices_cache):
        prices_df  = pd.read_csv(prices_cache, index_col=0, parse_dates=True)
        bh_ticker  = [c for c in prices_df.columns if c != XEON_TICKER][0]
        bh_wealth  = compute_buyhold_wealth(prices_df, bh_ticker, INITIAL_WEALTH)
        bh_wealth  = bh_wealth.reindex(bl_wealth.index, method='ffill')
    else:
        print("  ⚠ Sin caché de precios → Buy & Hold sintético")
        n = len(bl_wealth)
        bh_ticker = "Referencia"
        bh_wealth = pd.Series(
            INITIAL_WEALTH * (1.08 ** (np.arange(n) / 252)),
            index=bl_wealth.index
        )

    # 3) Métricas
    avg_rf = 0.025
    results_dict = {
        'BL-Omega (nuestra estrategia)': {
            'wealth':   bl_wealth,
            'trades':   trades_df,
            'history':  history_df,
            'xeon_col': xeon_col,
        },
        f'Buy & Hold ({bh_ticker})': {
            'wealth': bh_wealth,
        },
    }

    comparison = compare_strategies(results_dict, avg_risk_free=avg_rf)
    print("\nMÉTRICAS:")
    print(comparison.to_string())

    csv_path = os.path.join(OUTPUT_DIR, "metricas_comparativas.csv")
    comparison.to_csv(csv_path)
    print(f"\n  CSV: {csv_path}")

    # 4) Gráficos
    print("\nGenerando gráficos...")
    plot_wealth_curves(results_dict,
                       save_path=os.path.join(OUTPUT_DIR, "grafico_wealth.png"))
    plot_drawdown(results_dict,
                  save_path=os.path.join(OUTPUT_DIR, "grafico_drawdown.png"))

    todos_tickers = [c.replace('weight_', '') for c in history_df.columns
                     if c.startswith('weight_')]
    if todos_tickers:
        plot_weights_over_time(history_df, todos_tickers, xeon_ticker=XEON_TICKER,
                               save_path=os.path.join(OUTPUT_DIR, "grafico_pesos.png"))
        plot_xeon_weight(history_df, xeon_ticker=XEON_TICKER,
                         save_path=os.path.join(OUTPUT_DIR, "grafico_xeon.png"))

    plot_metrics_table(comparison,
                       save_path=os.path.join(OUTPUT_DIR, "metricas_comparativas.png"))

    # 5) Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    for nombre, row in comparison.iterrows():
        print(f"  {nombre:<40} "
              f"Sharpe={row.get('Sharpe Ratio', 'N/A'):>6}  "
              f"CAGR={row.get('CAGR (%)', 'N/A'):>+6}%  "
              f"MaxDD={row.get('Max Drawdown (%)', 'N/A'):>5}%")

    print(f"\n  Archivos en {OUTPUT_DIR}/")
    print("  ¡Análisis completado!")


if __name__ == "__main__":
    main()
```

---

---

## Archivo 4: `walk_forward_bl.py` — LA PARTE NUEVA

```python
"""
Walk-Forward Validation de la estrategia BL-Omega.

Por qué es necesario:
  El backtest simple usa parámetros fijos en todo el periodo. El problema
  es que no sabemos si esos parámetros funcionan bien porque son buenos
  de verdad, o porque están ajustados a los datos históricos que ya vimos
  (sobreajuste / overfitting).

  El walk-forward resuelve esto dividiendo el histórico en ventanas:
    - IS (in-sample, 3 años): buscamos los mejores parámetros
    - OOS (out-of-sample, 1 año): medimos con parámetros que NO vimos

  Si el Sharpe OOS es razonablemente parecido al IS → la estrategia es robusta.
  Si el Sharpe OOS es mucho peor → estamos sobreajustados.

Qué optimizamos:
  Solo 3 parámetros, que son los que más impactan y son defendibles:
    - omega_window_fast: 21, 42, 63 días
    - gamma:             -1, -2, -3
    - recalib_freq:      5, 10, 21 días
  Grid total: 3 × 3 × 3 = 27 combinaciones por ventana.

Ventanas:
  IS = 3 años, OOS = 1 año, paso = 1 año.
  Con datos desde ~2015: 7 ventanas, 7 años de OOS.

Tiempo estimado: 1-3 horas (27 backtests × 7 ventanas = 189 backtests).
"""

import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from main_black_litterman import run_backtest

WF_OUTPUT_DIR = "outputs/walk_forward"

# ============================================================
# PARÁMETROS DEL WALK-FORWARD
# ============================================================

IS_YEARS   = 3    # Años de in-sample para optimizar
OOS_YEARS  = 1    # Años de out-of-sample para medir
PASO_YEARS = 1    # Años de desplazamiento de ventana

# Grid de parámetros a optimizar
PARAM_GRID = {
    'omega_window_fast': [21, 42, 63],
    'gamma':             [-1, -2, -3],
    'recalib_freq':      [5, 10, 21],
}

# Métrica de optimización IS (qué maximizamos en cada ventana)
METRIC_IS = 'sharpe'   # Alternativas: 'cagr', 'calmar'


# ============================================================
# GENERADOR DE VENTANAS
# ============================================================

def generate_windows(data_start, data_end,
                     is_years=IS_YEARS, oos_years=OOS_YEARS, paso=PASO_YEARS):
    """
    Genera las ventanas IS y OOS del walk-forward.

    Args:
        data_start: primera fecha disponible en los datos
        data_end:   última fecha disponible
        is_years:   tamaño ventana IS en años
        oos_years:  tamaño ventana OOS en años
        paso:       desplazamiento en años

    Returns:
        lista de dicts {'is_start', 'is_end', 'oos_start', 'oos_end', 'ventana'}
    """
    windows = []
    current = pd.Timestamp(data_start)
    v = 1

    while True:
        is_start  = current
        is_end    = current + pd.DateOffset(years=is_years) - pd.DateOffset(days=1)
        oos_start = is_end  + pd.DateOffset(days=1)
        oos_end   = oos_start + pd.DateOffset(years=oos_years) - pd.DateOffset(days=1)

        if oos_end > pd.Timestamp(data_end):
            break

        windows.append({
            'ventana':   v,
            'is_start':  is_start,
            'is_end':    is_end,
            'oos_start': oos_start,
            'oos_end':   oos_end,
        })

        current += pd.DateOffset(years=paso)
        v += 1

    print(f"  {len(windows)} ventanas walk-forward generadas:")
    for w in windows:
        print(f"    V{w['ventana']}: IS={w['is_start'].date()}→{w['is_end'].date()} | "
              f"OOS={w['oos_start'].date()}→{w['oos_end'].date()}")

    return windows


# ============================================================
# GRID SEARCH EN UNA VENTANA IS
# ============================================================

def optimize_is(is_start, is_end, param_grid, ventana_num):
    """
    Prueba todas las combinaciones de parámetros en la ventana IS
    y devuelve la combinación con mejor Sharpe.

    Args:
        is_start:   inicio del periodo IS
        is_end:     fin del periodo IS
        param_grid: dict con listas de valores por parámetro
        ventana_num: número de ventana (para logs)

    Returns:
        tuple (best_params, best_metric, all_results)
          best_params: dict con los mejores parámetros
          best_metric: valor del Sharpe del mejor
          all_results: lista de dicts con todos los resultados IS
    """
    # Generar todas las combinaciones
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"\n  V{ventana_num} IS: probando {len(combos)} combinaciones "
          f"({is_start.date()} → {is_end.date()})...")

    best_params = None
    best_metric = -np.inf
    all_results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        try:
            result = run_backtest(
                start=is_start,
                end=is_end,
                params=params,
                save_results=False,   # No guardar cada IS individual
            )
            metric = result[METRIC_IS]
        except Exception as e:
            print(f"    Combo {i+1}/{len(combos)} falló: {e}")
            metric = -np.inf

        all_results.append({**params, METRIC_IS: metric})

        if metric > best_metric:
            best_metric = metric
            best_params = params.copy()

        # Log de progreso cada 9 combos
        if (i + 1) % 9 == 0:
            print(f"    {i+1}/{len(combos)} combinaciones | "
                  f"mejor hasta ahora: Sharpe={best_metric:.3f} "
                  f"con {best_params}")

    print(f"  V{ventana_num} IS: mejor combinación → "
          f"Sharpe={best_metric:.3f} | {best_params}")

    return best_params, best_metric, all_results


# ============================================================
# FUNCIÓN PRINCIPAL DEL WALK-FORWARD
# ============================================================

def run_walk_forward():
    """
    Pipeline completo del walk-forward:
      1. Detectar fechas disponibles
      2. Generar ventanas IS/OOS
      3. Para cada ventana: optimizar IS → evaluar OOS
      4. Concatenar resultados OOS
      5. Guardar todo y generar gráficos
    """
    print("=" * 60)
    print("WALK-FORWARD VALIDATION — BL-Omega")
    print("=" * 60)
    print(f"Grid: {PARAM_GRID}")
    print(f"IS={IS_YEARS} años, OOS={OOS_YEARS} año, paso={PASO_YEARS} año")
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print(f"Combinaciones por ventana: {n_combos}")

    os.makedirs(WF_OUTPUT_DIR, exist_ok=True)

    # Detectar fechas disponibles (carga los datos via caché)
    from main_black_litterman import _load_shared_data
    data, _ = _load_shared_data()
    data_start = data['backtest_start']
    data_end   = data['returns'].index[-1]

    print(f"\nDatos disponibles: {data_start.date()} → {data_end.date()}")

    # Generar ventanas
    windows = generate_windows(data_start, data_end)

    if len(windows) == 0:
        print("\nERROR: No hay suficientes datos para generar ventanas.")
        print(f"  Necesitas al menos {IS_YEARS + OOS_YEARS} años de datos.")
        return

    # ---- Bucle principal ----
    resultados_wf        = []
    mejores_params       = []
    wealth_oos_list      = []

    t_inicio = datetime.now()

    for w in windows:
        v = w['ventana']
        print(f"\n{'─'*50}")
        print(f"VENTANA {v}/{len(windows)}")

        # 1) Optimizar en IS
        best_params, best_sharpe_is, all_is = optimize_is(
            w['is_start'], w['is_end'], PARAM_GRID, v
        )

        # 2) Evaluar en OOS con los mejores parámetros del IS
        print(f"\n  V{v} OOS: evaluando con params óptimos "
              f"({w['oos_start'].date()} → {w['oos_end'].date()})...")
        try:
            oos_result = run_backtest(
                start=w['oos_start'],
                end=w['oos_end'],
                params=best_params,
                save_results=False,
            )
            sharpe_oos = oos_result['sharpe']
            cagr_oos   = oos_result['cagr']
            print(f"  V{v} OOS: Sharpe={sharpe_oos:.3f} | CAGR={cagr_oos:.2%}")

            # Guardar curva OOS para concatenar
            wealth_oos_list.append(oos_result['history']['wealth'])

        except Exception as e:
            print(f"  V{v} OOS falló: {e}")
            sharpe_oos = np.nan
            cagr_oos   = np.nan

        # Guardar resultados de esta ventana
        resultados_wf.append({
            'ventana':     v,
            'is_start':    w['is_start'].date(),
            'is_end':      w['is_end'].date(),
            'oos_start':   w['oos_start'].date(),
            'oos_end':     w['oos_end'].date(),
            'sharpe_is':   round(best_sharpe_is, 4),
            'sharpe_oos':  round(sharpe_oos, 4) if not np.isnan(sharpe_oos) else np.nan,
            'cagr_oos':    round(cagr_oos, 4) if not np.isnan(cagr_oos) else np.nan,
            **{f'param_{k}': v for k, v in best_params.items()},
        })

        mejores_params.append({
            'ventana': v,
            **best_params,
            'sharpe_is': round(best_sharpe_is, 4),
        })

        # Tiempo transcurrido y estimación de tiempo restante
        elapsed = (datetime.now() - t_inicio).seconds / 60
        avg_por_ventana = elapsed / v
        restante = avg_por_ventana * (len(windows) - v)
        print(f"  Tiempo transcurrido: {elapsed:.1f} min | "
              f"Estimado restante: {restante:.1f} min")

    # ---- Guardar resultados ----
    print(f"\n{'='*60}")
    print("GUARDANDO RESULTADOS WALK-FORWARD")

    df_wf = pd.DataFrame(resultados_wf)
    df_wf.to_csv(os.path.join(WF_OUTPUT_DIR, 'resultados_wf.csv'), index=False)

    df_params = pd.DataFrame(mejores_params)
    df_params.to_csv(
        os.path.join(WF_OUTPUT_DIR, 'mejores_params_por_ventana.csv'), index=False
    )

    # Concatenar curvas OOS
    if wealth_oos_list:
        wealth_oos = pd.concat(wealth_oos_list).sort_index()
        # Re-escalar para que empiece en 10M (cada ventana empieza desde 10M)
        # La concatenación preserva los valores absolutos de cada ventana
        wealth_oos.to_csv(
            os.path.join(WF_OUTPUT_DIR, 'wealth_oos_concatenado.csv'), header=['wealth']
        )

    print(f"\n{'='*60}")
    print("RESUMEN WALK-FORWARD")
    print(f"{'='*60}")
    print(df_wf[['ventana', 'is_start', 'oos_start', 'sharpe_is', 'sharpe_oos',
                  'cagr_oos']].to_string(index=False))

    sharpe_is_medio  = df_wf['sharpe_is'].mean()
    sharpe_oos_medio = df_wf['sharpe_oos'].dropna().mean()
    ratio_is_oos     = sharpe_oos_medio / sharpe_is_medio if sharpe_is_medio > 0 else 0

    print(f"\n  Sharpe IS medio:   {sharpe_is_medio:.3f}")
    print(f"  Sharpe OOS medio:  {sharpe_oos_medio:.3f}")
    print(f"  Ratio OOS/IS:      {ratio_is_oos:.2f}")
    print()
    if ratio_is_oos >= 0.6:
        print("  ✓ La estrategia es ROBUSTA (OOS ≥ 60% del IS)")
    elif ratio_is_oos >= 0.4:
        print("  ~ La estrategia es MODERADAMENTE robusta (OOS ≥ 40% del IS)")
    else:
        print("  ✗ Posible sobreajuste (OOS < 40% del IS)")

    print(f"\n  Resultados en: {WF_OUTPUT_DIR}/")

    # Generar gráficos del walk-forward
    _plot_walk_forward(df_wf, wealth_oos if wealth_oos_list else None)

    return df_wf


# ============================================================
# GRÁFICOS DEL WALK-FORWARD
# ============================================================

def _plot_walk_forward(df_wf, wealth_oos=None):
    """Genera los gráficos de resultados del walk-forward."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Gráfico 1: Sharpe IS vs OOS por ventana
    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_wf['ventana']
    ax.bar(x - 0.2, df_wf['sharpe_is'],  0.35, label='IS (optimización)', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, df_wf['sharpe_oos'], 0.35, label='OOS (validación)',   color='#4CAF50', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Ventana'); ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Walk-Forward: Sharpe IS vs OOS por ventana\n'
                 '(OOS = lo que realmente importa)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"V{i}\n({r['oos_start']})" for i, r in df_wf.iterrows()],
                       fontsize=8)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WF_OUTPUT_DIR, 'grafico_wf_sharpe.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico guardado: grafico_wf_sharpe.png")

    # Gráfico 2: Curva OOS concatenada vs backtest simple
    if wealth_oos is not None:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(wealth_oos.index, wealth_oos.values / 1e6,
                label='Walk-Forward OOS (real)', color='#4CAF50', linewidth=2)

        # Intentar cargar el backtest simple para comparar
        bs_file = 'outputs/bl_omega/wealth_history.csv'
        if os.path.exists(bs_file):
            bs = pd.read_csv(bs_file, index_col=0, parse_dates=True)
            ax.plot(bs.index, bs['wealth'].values / 1e6,
                    label='Backtest simple (referencia)', color='#2196F3',
                    linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_xlabel('Fecha'); ax.set_ylabel('Patrimonio (M EUR)')
        ax.set_title('Walk-Forward OOS vs Backtest Simple\n'
                     '(OOS usa parámetros que NO se vieron en esa ventana)')
        ax.legend(); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate(); plt.tight_layout()
        plt.savefig(os.path.join(WF_OUTPUT_DIR, 'grafico_wf_wealth.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Gráfico guardado: grafico_wf_wealth.png")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    run_walk_forward()
```

---

## Cómo prepararte antes de que P4 esté listo

```python
# test_analysis_sintetico.py — ejecuta esto ahora mismo para probar tus scripts
import numpy as np
import pandas as pd
import os
from metrics_bl import compute_metrics, compare_strategies
from plots_bl import plot_wealth_curves, plot_drawdown, plot_xeon_weight

np.random.seed(42)
dates  = pd.bdate_range('2015-01-01', '2024-12-31')
n      = len(dates)

# Simular historial con columna de XEON.DE
returns_bl = np.random.randn(n) * 0.01 + 0.00040
wealth_bl  = pd.Series(10_000_000 * np.cumprod(1 + returns_bl), index=dates)
returns_bh = np.random.randn(n) * 0.012 + 0.00030
wealth_bh  = pd.Series(10_000_000 * np.cumprod(1 + returns_bh), index=dates)

# Historial con pesos incluyendo XEON.DE
xeon_weights = np.random.beta(2, 5, n) * 0.5  # Entre 0 y 50%, sesgado hacia abajo
history_df   = pd.DataFrame({
    'wealth':         wealth_bl.values,
    'weight_ETF_A':   0.4 - xeon_weights * 0.4,
    'weight_ETF_B':   0.35 - xeon_weights * 0.35,
    'weight_XEON.DE': xeon_weights,
}, index=dates)

results_dict = {
    'BL-Omega':   {'wealth': wealth_bl, 'history': history_df, 'xeon_col': 'weight_XEON.DE'},
    'Buy & Hold': {'wealth': wealth_bh},
}

comparison = compare_strategies(results_dict, avg_risk_free=0.025)
print("Métricas (datos sintéticos):")
print(comparison[['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'XEON.DE medio (%)']].to_string())

os.makedirs('outputs/test', exist_ok=True)
plot_wealth_curves(results_dict, save_path='outputs/test/wealth_test.png')
plot_drawdown(results_dict, save_path='outputs/test/drawdown_test.png')
plot_xeon_weight(history_df, xeon_ticker='XEON.DE', save_path='outputs/test/xeon_test.png')

print("\n✓ Scripts funcionan correctamente con datos sintéticos.")
print("  Gráficos en outputs/test/")
print("  Cuando P4 avise, ejecutar: python run_analysis.py")
```

---

## Checklist antes de avisar al grupo

**Antes de que P4 esté listo (puedes hacer esto ya):**
- [ ] `metrics_bl.py` creado y test sintético pasa
- [ ] `plots_bl.py` crea los 4 gráficos sin errores (incluido `grafico_xeon.png`)
- [ ] `run_analysis.py` preparado
- [ ] `walk_forward_bl.py` creado y la lógica de ventanas funciona con datos sintéticos

**Cuando P4 avise "BACKTEST COMPLETADO":**
- [ ] `python run_analysis.py` sin errores → outputs en `outputs/bl_omega/`
- [ ] `python walk_forward_bl.py` lanzado inmediatamente (tarda 1-3 horas)
- [ ] Mientras corre el walk-forward, preparar la presentación de métricas

**Cuando termine el walk-forward:**
- [ ] `resultados_wf.csv` generado con Sharpe IS y OOS por ventana
- [ ] `grafico_wf_sharpe.png` generado
- [ ] `grafico_wf_wealth.png` generado
- [ ] Ratio OOS/IS reportado en el WhatsApp

---

## Cuándo avisar por WhatsApp

| Momento | Qué escribir |
|---|---|
| Scripts listos con datos sintéticos | "P5: scripts listos, esperando backtest de P4" |
| P4 avisa → lanzas run_analysis.py | "P5: análisis en curso..." |
| run_analysis.py terminado | "P5: ✓ análisis listo, lanzando walk-forward (tarda ~2h)" |
| Walk-forward terminado | "P5: ✓ WF LISTO — Sharpe IS medio=X.XX, OOS medio=X.XX, ratio=X.XX" |
| Sharpe OOS muy negativo | "ATENCIÓN: WF da Sharpe OOS negativo, revisar con P1/P2" |

---

## Lo que NO tienes que hacer

- No ejecutar el backtest simple tú mismo (lo hace P4 con `main_black_litterman.py`)
- No descargar datos (los tiene P4)
- No tocar ningún archivo de P1, P2, P3 o P4

---

## Dependencias

```
pip install numpy pandas matplotlib openpyxl
```
