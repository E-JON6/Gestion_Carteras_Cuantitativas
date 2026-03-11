"""
Estimación de costes de transacción.
- Estimador de Corwin-Schultz (2012) a partir de High/Low
- Fórmula de las normas del proyecto (con BID/ASK reales)
"""

import numpy as np
import pandas as pd

import os

# Valores por defecto si no hay archivo BID/ASK disponible
DEFAULT_LAMBDA_L = 0.002
DEFAULT_LAMBDA_M = 0.002

def corwin_schultz_spread(high, low, window=21):
    """
    Estimador de spread de Corwin-Schultz (2012).
    Estima el spread bid-ask a partir de precios High y Low diarios.

    S = 2(e^α - 1) / (1 + e^α)

    donde α depende de los ratios High/Low de días consecutivos.

    Args:
        high: pd.Series de precios High
        low: pd.Series de precios Low
        window: ventana de suavizado (default 21 días)

    Returns:
        pd.Series con el spread estimado (en proporción, ej. 0.002 = 0.2%)
    """
    log_hl = np.log(high / low)

    # β = E[ln(H/L)]² para un día
    beta = log_hl ** 2

    # γ = ln(H_2d / L_2d)² para dos días consecutivos
    high_2d = high.rolling(2).max()
    low_2d = low.rolling(2).min()
    gamma = np.log(high_2d / low_2d) ** 2

    # Suavizar con ventana rolling
    beta_avg = beta.rolling(window).mean()
    gamma_avg = gamma.rolling(window).mean()

    # Cálculo del alpha
    k = 2 * np.sqrt(2) - 1
    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    term1 = (np.sqrt(2 * beta_avg) - np.sqrt(beta_avg)) / (3 - 2 * np.sqrt(2))
    term2 = np.sqrt(gamma_avg / (3 - 2 * np.sqrt(2)))

    alpha = term1 - term2
    # Asegurar alpha no negativo (spread ≥ 0)
    alpha = alpha.clip(lower=0)

    # Spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Asegurar spread no negativo y razonable
    spread = spread.clip(lower=0, upper=0.05)

    return spread


def compute_ct_from_bid_ask(bid, ask):
    """
    Calcula CT según la fórmula de las normas del proyecto.

    CT_i = (ASK_i - BID_i) / ((ASK_i + BID_i) * 0.5) * 0.5 + 0.0001

    Args:
        bid: pd.Series o array de precios bid
        ask: pd.Series o array de precios ask

    Returns:
        float: CT promedio
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)

    mid = (ask + bid) / 2.0
    ct_i = (ask - bid) / mid * 0.5 + 0.0001

    return float(np.mean(ct_i))


def estimate_transaction_costs(risky_data, method='corwin_schultz', window=21):
    """
    Estima λ_L y λ_M a partir de datos del activo.

    Args:
        risky_data: pd.DataFrame con columnas High, Low, Close
        method: 'corwin_schultz' o 'fixed'
        window: ventana para estimador C-S

    Returns:
        dict con 'lambda_L', 'lambda_M', 'spread_series'
    """
    if method == 'corwin_schultz':
        spread = corwin_schultz_spread(
            risky_data['High'], risky_data['Low'], window=window
        )
        # λ = spread/2 + 0.0001 (comisión mínima)
        lambda_series = spread / 2 + 0.0001

        # Usar la mediana como estimación robusta
        lambda_est = float(lambda_series.median())

        # Floor: ETFs líquidos Euro Stoxx 50 tienen spread ~0.02% + comisión ~0.01%
        # Mínimo realista: ~0.03% (3 bps por lado)
        lambda_est = max(lambda_est, 0.0003)

        return {
            'lambda_L': lambda_est,
            'lambda_M': lambda_est,
            'spread_series': spread,
            'lambda_series': lambda_series
        }

    elif method == 'fixed':
        return {
            'lambda_L': DEFAULT_LAMBDA_L,
            'lambda_M': DEFAULT_LAMBDA_M,
            'spread_series': None,
            'lambda_series': None
        }

    else:
        raise ValueError(f"Método desconocido: {method}")


def load_lambda_from_bidask(bidask_dir, etf_ticker):
    """
    Lee el CSV de BID/ASK de Bloomberg y calcula el λ según
    la fórmula de las normas, usando solo enero 2026.

    Args:
        bidask_dir: ruta a la carpeta con los CSVs
        etf_ticker: ticker del ETF (ej. 'MSE.PA')

    Returns:
        float: λ calculado (se usa igual para compra y venta)
    """

    # Mapeo de ticker a nombre de archivo
    ticker_to_file = {
        'MSE.PA':  'EUROSTOXX50-bid_ask.csv',
        'IUSE.L':  'SP500-bid_ask.csv',
        'IEMA.L':  'EM-bid_ask.csv',
        'IUSN.DE': 'WorldSmallCap-bid_ask.csv',
        'IS3Q.DE': 'IS3Q.DE-bid_ask.csv',
    }

    filename = ticker_to_file.get(etf_ticker)
    if filename is None:
        print(f"  ⚠ No hay archivo BID/ASK para {etf_ticker}, usando λ por defecto")
        return DEFAULT_LAMBDA_L

    filepath = os.path.join(bidask_dir, filename)
    if not os.path.exists(filepath):
        print(f"  ⚠ Archivo no encontrado: {filepath}, usando λ por defecto")
        return DEFAULT_LAMBDA_L

    # Leer CSV de Bloomberg (6 líneas de cabecera)
    # usecols=[0,1,2] ignora columnas vacías extra que Excel añade al exportar
    df = pd.read_csv(filepath, sep=';', skiprows=6, decimal=',',
                     usecols=[0, 1, 2], header=0)
    df.columns = ['Date', 'BID', 'ASK']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.dropna()

    # Filtrar solo enero 2026
    enero = df[
        (df['Date'] >= '2026-01-01') &
        (df['Date'] <= '2026-01-31')
    ]

    if len(enero) == 0:
        print(f"  ⚠ Sin datos de enero 2026 para {etf_ticker}, usando λ por defecto")
        return DEFAULT_LAMBDA_L

    # Fórmula de las normas
    lam = compute_ct_from_bid_ask(enero['BID'], enero['ASK'])
    print(f"  λ calculado desde BID/ASK ({len(enero)} días enero 2026): {lam:.6f} ({lam*100:.4f}%)")

    return lam