"""
Visualizaciones para el backtesting comparativo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_equity_curves(results_list, title=None, etf_name=None):
    """
    Gráfica de equity curves (riqueza normalizada) de todas las estrategias.
    Buy & Hold actúa como referencia del ETF (total return).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, result in enumerate(results_list):
        history = result['history']
        wealth = history['wealth']
        # Normalizar a base 100
        wealth_norm = wealth / wealth.iloc[0] * 100
        color = colors[i % len(colors)]
        lw = 2.0 if i == len(results_list) - 1 else 1.5  # E3 más grueso
        ax.plot(wealth_norm.index, wealth_norm.values,
                label=result['strategy_name'], linewidth=lw, color=color)

    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Valor del Portfolio (base 100)', fontsize=12)
    default_title = f"Comparativa de Estrategias — {etf_name}" if etf_name else "Comparativa de Estrategias"
    ax.set_title(title or default_title, fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_drawdowns(results_list, title=None, etf_name=None):
    """
    Gráfica de drawdowns de todas las estrategias.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, result in enumerate(results_list):
        history = result['history']
        wealth = history['wealth']
        cummax = wealth.cummax()
        dd = (wealth - cummax) / cummax * 100
        color = colors[i % len(colors)]
        ax.fill_between(dd.index, dd.values, 0, alpha=0.1, color=color)
        ax.plot(dd.index, dd.values, label=result['strategy_name'],
                linewidth=1, color=color)

    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    default_title = f"Drawdowns — {etf_name}" if etf_name else "Drawdowns"
    ax.set_title(title or default_title, fontsize=14)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_alpha_evolution(results_list, title=None, etf_name=None):
    """
    Gráfica de la evolución de α en el tiempo para cada estrategia.
    """
    n = len(results_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, result in enumerate(results_list):
        ax = axes[i]
        history = result['history']
        alpha = history['alpha']
        color = colors[i % len(colors)]

        ax.plot(alpha.index, alpha.values, linewidth=0.8, color=color)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='α=1 (sin apalancamiento)')
        ax.set_ylabel('α')
        ax.set_title(result['strategy_name'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Marcar trades
        trades = result['trades']
        if len(trades) > 0:
            buys = trades[trades['type'] == 'buy']
            sells = trades[trades['type'] == 'sell']
            if len(buys) > 0:
                ax.scatter(pd.to_datetime(buys['date']),
                          buys['alpha_after'], marker='^', color='green',
                          s=10, alpha=0.5, zorder=5)
            if len(sells) > 0:
                ax.scatter(pd.to_datetime(sells['date']),
                          sells['alpha_after'], marker='v', color='red',
                          s=10, alpha=0.5, zorder=5)

    axes[-1].set_xlabel('Fecha')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    default_title = f"Evolución de α — {etf_name}" if etf_name else "Evolución de α (fracción en riesgo)"
    fig.suptitle(title or default_title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_cumulative_costs(results_list, title=None, etf_name=None):
    """
    Gráfica de costes de transacción acumulados como % del patrimonio inicial.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, result in enumerate(results_list):
        history = result['history']
        costs_pct = history['total_costs'] / result['initial_wealth'] * 100
        color = colors[i % len(colors)]
        ax.plot(costs_pct.index, costs_pct.values,
                label=result['strategy_name'], linewidth=1.5, color=color)

    ax.set_xlabel('Fecha')
    ax.set_ylabel('Costes Acumulados (% patrimonio inicial)')
    default_title = f"Costes de Transacción — {etf_name}" if etf_name else "Costes de Transacción Acumulados"
    ax.set_title(title or default_title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_trade_distribution(results_list, title=None, etf_name=None):
    """
    Gráfico de barras con número de compras y ventas por estrategia.
    """
    names = []
    buys = []
    sells = []

    for result in results_list:
        names.append(result['strategy_name'].replace(': ', '\n'))
        trades = result['trades']
        if len(trades) > 0:
            buys.append(len(trades[trades['type'] == 'buy']))
            sells.append(len(trades[trades['type'] == 'sell']))
        else:
            buys.append(0)
            sells.append(0)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, buys, width, label='Compras', color='#2ecc71')
    bars2 = ax.bar(x + width / 2, sells, width, label='Ventas', color='#e74c3c')

    ax.set_ylabel('Número de operaciones')
    default_title = f"Distribución de Operaciones — {etf_name}" if etf_name else "Distribución de Operaciones"
    ax.set_title(title or default_title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir etiquetas
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_dn_bands_detail(results_list, prices=None, title=None, etf_name=None):
    """
    Gráfica consolidada: precio del ETF (eje izq.) + α de cada estrategia
    con sus fronteras DN (eje der.) en un solo plot.

    Args:
        results_list: lista de resultados de backtest (cada uno con 'bands' DataFrame)
        prices: pd.Series con precios de cierre del ETF
        title: título general (opcional)
        etf_name: nombre del ETF para títulos
    """
    if not results_list:
        return None

    fig, ax_price = plt.subplots(figsize=(16, 8))
    ax_alpha = ax_price.twinx()

    # Colores por estrategia
    strategy_colors = {
        'Buy & Hold':   '#7f8c8d',
        'Benchmark':    '#e74c3c',
        'E1':           '#3498db',
        'E2':           '#2ecc71',
        'E3':           '#f39c12',
    }

    def _get_color(name):
        for key, color in strategy_colors.items():
            if key in name:
                return color
        return '#9b59b6'

    # --- Eje izquierdo: precio del ETF ---
    dates_ref = results_list[0]['history'].index
    if prices is not None:
        price_range = prices.loc[dates_ref[0]:dates_ref[-1]]
        ax_price.plot(price_range.index, price_range.values,
                       '-', color='#2c3e50', linewidth=1.2, alpha=0.4,
                       label=f'Precio {etf_name or "ETF"}')
        ax_price.set_ylabel(f'Precio {etf_name or "ETF"} (EUR)', fontsize=11,
                             color='#2c3e50')
        ax_price.tick_params(axis='y', labelcolor='#2c3e50')

    # --- Eje derecho: α de cada estrategia + bandas DN ---
    for result in results_list:
        name = result['strategy_name']
        color = _get_color(name)
        history = result['history']
        alpha = history['alpha']
        dates = alpha.index

        # Nombre corto para leyenda
        short_name = name.split(':')[0].strip() if ':' in name else name.split('(')[0].strip()

        # α actual
        ax_alpha.plot(dates, alpha.values, '-', color=color,
                       linewidth=1.0, label=f'α {short_name}')

        # Bandas DN (solo estrategias que las tienen)
        bands = result.get('bands')
        if bands is not None and len(bands) > 0 and bands['alpha_lower'].notna().any():
            lower = bands['alpha_lower']
            upper = bands['alpha_upper']

            ax_alpha.fill_between(dates, lower.values, upper.values,
                                   alpha=0.10, color=color)
            ax_alpha.plot(dates, lower.values, '--', color=color,
                           alpha=0.5, linewidth=0.6)
            ax_alpha.plot(dates, upper.values, '--', color=color,
                           alpha=0.5, linewidth=0.6)

    ax_alpha.set_ylabel('α (fracción en activo de riesgo)', fontsize=11)
    ax_alpha.set_ylim(bottom=-0.05)
    ax_alpha.grid(True, alpha=0.2)

    # Leyenda combinada
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_alpha, labels_alpha = ax_alpha.get_legend_handles_labels()
    ax_alpha.legend(lines_price + lines_alpha, labels_price + labels_alpha,
                     loc='upper left', fontsize=9, ncol=2)

    # Formato eje X
    ax_price.set_xlabel('Fecha', fontsize=11)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    default_title = (f"Fronteras Davis-Norman y Asignación — {etf_name}"
                     if etf_name else "Fronteras Davis-Norman y Asignación")
    ax_price.set_title(title or default_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_strategy_detail(result, prices=None, etf_name=None):
    """
    Gráfico individual por estrategia: precio ETF (eje izq.) + α con bandas DN
    (eje der.) + líneas verticales verdes (compras) y rojas (ventas).

    Args:
        result: dict de resultado de una estrategia (con 'history', 'bands', 'trades')
        prices: pd.Series con precios de cierre del ETF
        etf_name: nombre del ETF para títulos
    """
    strategy_name = result['strategy_name']
    history = result['history']
    alpha = history['alpha']
    dates = alpha.index

    fig, ax_price = plt.subplots(figsize=(16, 7))
    ax_alpha = ax_price.twinx()

    # --- Eje izquierdo: precio del ETF ---
    if prices is not None:
        price_range = prices.loc[dates[0]:dates[-1]]
        ax_price.plot(price_range.index, price_range.values,
                       '-', color='#2c3e50', linewidth=1.2, alpha=0.4,
                       label=f'Precio {etf_name or "ETF"}')
        ax_price.set_ylabel(f'Precio {etf_name or "ETF"} (EUR)', fontsize=11,
                             color='#2c3e50')
        ax_price.tick_params(axis='y', labelcolor='#2c3e50')

    # --- Eje derecho: α + bandas DN ---
    # Bandas DN
    bands = result.get('bands')
    has_bands = (bands is not None and len(bands) > 0 and
                 bands['alpha_lower'].notna().any())

    if has_bands:
        lower = bands['alpha_lower']
        upper = bands['alpha_upper']
        ax_alpha.fill_between(dates, lower.values, upper.values,
                               alpha=0.15, color='#3498db',
                               label='Zona no-transacción')
        ax_alpha.plot(dates, lower.values, '--', color='#3498db',
                       alpha=0.6, linewidth=0.7, label='Frontera compra (α_L)')
        ax_alpha.plot(dates, upper.values, '--', color='#3498db',
                       alpha=0.6, linewidth=0.7, label='Frontera venta (α_U)')

    # α actual
    ax_alpha.plot(dates, alpha.values, '-', color='#2c3e50',
                   linewidth=1.0, label='α actual')

    # --- Líneas verticales: compras (verde) y ventas (rojo) ---
    trades = result['trades']
    buy_label_added = False
    sell_label_added = False
    if len(trades) > 0:
        buys = trades[trades['type'] == 'buy']
        sells = trades[trades['type'] == 'sell']
        for _, row in buys.iterrows():
            label = 'Compra' if not buy_label_added else None
            ax_alpha.axvline(x=pd.to_datetime(row['date']), color='green',
                              alpha=0.3, linewidth=0.5, label=label)
            buy_label_added = True
        for _, row in sells.iterrows():
            label = 'Venta' if not sell_label_added else None
            ax_alpha.axvline(x=pd.to_datetime(row['date']), color='red',
                              alpha=0.3, linewidth=0.5, label=label)
            sell_label_added = True

    ax_alpha.set_ylabel('α (fracción en activo de riesgo)', fontsize=11)
    ax_alpha.set_ylim(bottom=-0.05)
    ax_alpha.grid(True, alpha=0.2)

    # Leyenda combinada
    lines_p, labels_p = ax_price.get_legend_handles_labels()
    lines_a, labels_a = ax_alpha.get_legend_handles_labels()
    ax_alpha.legend(lines_p + lines_a, labels_p + labels_a,
                     loc='upper left', fontsize=9, ncol=2)

    ax_price.set_xlabel('Fecha', fontsize=11)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    title = f"{strategy_name} — {etf_name}" if etf_name else strategy_name
    ax_price.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_all_plots(results_list, save_dir=None, etf_name=None, prices=None):
    """
    Genera todas las gráficas y opcionalmente las guarda.

    Args:
        results_list: lista de resultados de backtest
        save_dir: directorio donde guardar las gráficas (None = solo mostrar)
        etf_name: nombre del ETF para títulos

    Returns:
        dict de figuras
    """
    figs = {}

    figs['equity'] = plot_equity_curves(results_list, etf_name=etf_name)
    figs['drawdown'] = plot_drawdowns(results_list, etf_name=etf_name)
    figs['alpha'] = plot_alpha_evolution(results_list, etf_name=etf_name)
    figs['costs'] = plot_cumulative_costs(results_list, etf_name=etf_name)
    figs['trades'] = plot_trade_distribution(results_list, etf_name=etf_name)

    # Fronteras DN + α + precio ETF (consolidado)
    dn_fig = plot_dn_bands_detail(results_list, prices=prices, etf_name=etf_name)
    if dn_fig is not None:
        figs['dn_bands'] = dn_fig

    # Gráfico individual por estrategia (bandas + α + precio + compras/ventas)
    for result in results_list:
        sname = result['strategy_name']
        safe = sname.replace(':', '').replace(' ', '_').replace('&', 'and')
        fig_s = plot_strategy_detail(result, prices=prices, etf_name=etf_name)
        figs[f'detail_{safe}'] = fig_s

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f'{name}.png'),
                       dpi=150, bbox_inches='tight')
            print(f"  Guardado: {name}.png")

    return figs
