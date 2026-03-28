"""
Merton Multivariante: pesos óptimos de cartera para ETFs de riesgo.

XEON.DE se excluye del cálculo. Su peso = 1 - suma(pesos_riesgo).
Los pesos de riesgo pueden sumar entre 0 y 1 según el régimen de volatilidad.

Mejoras sobre la versión básica:
  1. Régimen de volatilidad (cap dinámico en riesgo total)
  2. Restricción sectorial (máx 25% por categoría)
  3. Condición de salida de congelados (< 2% → liquidar)
  4. XEON.DE excluido explícitamente del cálculo

Referencia: Merton (1969, 1971).
"""

import numpy as np

# ============================================================
# PARÁMETROS
# ============================================================

GAMMA             = -2      # Parámetro de utilidad. RRA = 1 - GAMMA = 3.
N_TOP_ASSETS      = 5       # Máximo de ETFs de riesgo en cartera simultáneamente
MAX_WEIGHT        = 0.40    # Peso máximo por ETF individual (40%)
MIN_WEIGHT        = 0.01    # Peso mínimo para considerar que hay posición (1%)
MAX_SECTOR_WEIGHT = 0.25    # Peso máximo por categoría sectorial (25%)
FREEZE_EXIT_THR   = 0.02    # Si un congelado cae por debajo del 2% → liquidar

# Umbrales de régimen de volatilidad
VOL_CAUTION_THR = 0.20   # sigma > 20% → régimen caution, cap 70% en riesgo
VOL_CRISIS_THR  = 0.30   # sigma > 30% → régimen crisis, cap 40% en riesgo


# ============================================================
# DETECTAR RÉGIMEN DE VOLATILIDAD
# ============================================================

def detect_regime(sigma_mercado):
    """
    Determina el régimen de volatilidad del mercado y el cap máximo en riesgo.

    Args:
        sigma_mercado: float, volatilidad media anualizada del universo de ETFs
                       (calculada por P4 como media de las volatilidades individuales)

    Returns:
        tuple (regime, max_risk_weight)
          regime: 'normal', 'caution' o 'crisis'
          max_risk_weight: peso máximo total en ETFs de riesgo (el resto va a XEON.DE)
    """
    if sigma_mercado > VOL_CRISIS_THR:
        return 'crisis', 0.40
    elif sigma_mercado > VOL_CAUTION_THR:
        return 'caution', 0.70
    else:
        return 'normal', 1.00


# ============================================================
# PESOS BRUTOS DE MERTON
# ============================================================

def merton_weights_risk_only(mu_BL, Sigma, risk_free_rate, tickers,
                              xeon_ticker='XEON.DE', gamma=GAMMA):
    """
    Calcula los pesos óptimos de Merton SOLO para los ETFs de riesgo.
    XEON.DE se excluye antes de calcular.

    Fórmula: w* = (1 / gamma_eff) × Sigma_riesgo^(-1) × (mu_riesgo - r)

    Con gamma = -2:  gamma_eff = 1 - gamma = 3
    → w* = (1/3) × Sigma_riesgo^(-1) × (mu_riesgo - r)

    Args:
        mu_BL:           np.array (N,) — retornos esperados, incluyendo XEON.DE
        Sigma:           np.array (N,N) — covarianza, incluyendo XEON.DE
        risk_free_rate:  float
        tickers:         list de N strings, incluyendo XEON.DE
        xeon_ticker:     string del ticker monetario
        gamma:           parámetro de utilidad (default: -2)

    Returns:
        tuple (w_raw_risk, risk_tickers, risk_indices)
          w_raw_risk:   np.array con pesos brutos de los ETFs de riesgo
          risk_tickers: list de tickers de riesgo (sin XEON.DE)
          risk_indices: list de índices de riesgo en el array original
    """
    # Identificar índices de ETFs de riesgo (excluir XEON.DE)
    risk_indices = [i for i, t in enumerate(tickers) if t != xeon_ticker]
    risk_tickers = [tickers[i] for i in risk_indices]
    N_risk = len(risk_indices)

    if N_risk == 0:
        return np.array([]), [], []

    # Extraer mu y Sigma solo para los ETFs de riesgo
    mu_risk    = mu_BL[risk_indices]
    Sigma_risk = Sigma[np.ix_(risk_indices, risk_indices)]

    # Exceso de retorno sobre la tasa libre de riesgo
    excess_return = mu_risk - risk_free_rate

    # Aversión al riesgo efectiva
    gamma_eff = 1.0 - gamma   # Con gamma=-2 → 3

    # Invertir Sigma (ya viene regularizada de P1 con Ledoit-Wolf)
    try:
        Sigma_inv = np.linalg.inv(Sigma_risk)
    except np.linalg.LinAlgError:
        print("  [AVISO P2] Usando pseudo-inversa para Sigma_risk")
        Sigma_inv = np.linalg.pinv(Sigma_risk)

    w_raw = (1.0 / gamma_eff) * Sigma_inv @ excess_return

    return w_raw, risk_tickers, risk_indices


# ============================================================
# RESTRICCIONES
# ============================================================

def apply_constraints(w_raw, risk_tickers, categoria_por_ticker=None,
                      n_top=N_TOP_ASSETS, max_weight=MAX_WEIGHT,
                      min_weight=MIN_WEIGHT,
                      max_sector=MAX_SECTOR_WEIGHT,
                      max_risk_total=1.0):
    """
    Aplica todas las restricciones a los pesos brutos de Merton.

    Orden de restricciones:
      1. Sin posiciones cortas (w >= 0)
      2. Solo top N_TOP_ASSETS por peso
      3. Peso máximo individual (MAX_WEIGHT = 40%)
      4. Restricción sectorial (MAX_SECTOR_WEIGHT = 25% por categoría)
      5. Cap total de riesgo según régimen (max_risk_total)
      6. Normalizar para que la suma de riesgo ≤ max_risk_total

    Args:
        w_raw:              np.array — pesos brutos (pueden ser negativos o >1)
        risk_tickers:       list — nombres de los ETFs de riesgo
        categoria_por_ticker: dict {ticker: categoria} para restricción sectorial
        n_top:              máximo ETFs en cartera
        max_weight:         peso máximo por ETF
        min_weight:         peso mínimo para considerar posición
        max_sector:         peso máximo por categoría
        max_risk_total:     peso máximo total en riesgo (del régimen)

    Returns:
        np.array — pesos finales (suma ≤ max_risk_total, todos >= 0)
    """
    N = len(w_raw)
    w = w_raw.copy()

    # 1. Sin shorts
    w = np.maximum(w, 0.0)

    if w.sum() < 1e-10:
        # Todos cero: distribuir equitativamente entre los primeros n_top
        w = np.zeros(N)
        w[:min(n_top, N)] = 1.0 / min(n_top, N)

    # 2. Solo los top N_TOP_ASSETS
    sorted_idx  = np.argsort(w)[::-1]
    w_filtered  = np.zeros(N)
    w_filtered[sorted_idx[:n_top]] = w[sorted_idx[:n_top]]
    w = w_filtered

    # 3. Cap individual al MAX_WEIGHT
    for _ in range(10):
        if not (w > max_weight).any():
            break
        w = np.minimum(w, max_weight)

    # 4. Restricción sectorial (si se proporcionan categorías)
    if categoria_por_ticker and len(categoria_por_ticker) > 0:
        from collections import defaultdict
        categoria_idx = defaultdict(list)
        for i, ticker in enumerate(risk_tickers):
            cat = categoria_por_ticker.get(ticker, f'unknown_{ticker}')
            categoria_idx[cat].append(i)

        for _ in range(10):
            changed = False
            for cat, indices in categoria_idx.items():
                sector_total = w[indices].sum()
                if sector_total > max_sector + 1e-6:
                    scale = max_sector / sector_total
                    for i in indices:
                        w[i] *= scale
                    changed = True
            if not changed:
                break

    # 5. Cap total según régimen
    total = w.sum()
    if total > max_risk_total + 1e-6:
        w = w * (max_risk_total / total)

    # 6. Eliminar posiciones por debajo del mínimo
    w[w < min_weight] = 0.0

    # Re-normalizar si la suma supera el cap (puede pasar tras eliminar pequeños)
    if w.sum() > max_risk_total + 1e-6:
        w = w * (max_risk_total / w.sum())

    return w


# ============================================================
# DETECTAR CONGELADOS A LIQUIDAR
# ============================================================

def check_frozen_exits(current_weights, tickers, frozen_tickers,
                       xeon_ticker='XEON.DE', threshold=FREEZE_EXIT_THR):
    """
    Identifica qué congelados deben liquidarse porque su peso ha caído demasiado.

    Un ETF congelado es uno que salió del top-5 de Merton pero ya estaba en
    cartera. Se mantiene hasta que su peso cae por debajo del umbral (2%).
    Cuando eso ocurre, no tiene sentido seguir manteniéndolo — los costes
    de transacción de mantenerlo superan el beneficio de la diversificación.

    Args:
        current_weights:  np.array — pesos actuales de la cartera completa
        tickers:          list — tickers en el mismo orden que current_weights
        frozen_tickers:   list — tickers actualmente congelados
        xeon_ticker:      string del ticker monetario
        threshold:        float — umbral de peso para liquidar (0.02 = 2%)

    Returns:
        list — tickers congelados que deben liquidarse en el próximo rebalanceo
    """
    liquidar = []
    for ticker in frozen_tickers:
        if ticker == xeon_ticker:
            continue
        if ticker in tickers:
            idx = tickers.index(ticker)
            if idx < len(current_weights) and current_weights[idx] < threshold:
                liquidar.append(ticker)
    return liquidar


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def run_merton_v0(bl_result, risk_free_rate,
                             xeon_ticker='XEON.DE',
                             sigma_mercado=0.15,
                             categoria_por_ticker=None,
                             current_weights=None,
                             frozen_tickers=None,
                             gamma=GAMMA):
    """
    Pipeline completo: mu_BL + Sigma → pesos óptimos con todas las restricciones.

    Esta es la función que P4 llama desde el motor (y P3 desde la estrategia).

    Args:
        bl_result:            dict de run_black_litterman() — de P1
        risk_free_rate:       float — tasa BCE
        xeon_ticker:          string — ticker del ETF monetario
        sigma_mercado:        float — volatilidad media anualizada del universo
                              (P4 la calcula como media de sqrt(diag(Sigma)))
        categoria_por_ticker: dict {ticker: categoria} para restricción sectorial
        current_weights:      np.array — pesos actuales (para detectar congelados a liquidar)
        frozen_tickers:       list — ETFs actualmente congelados
        gamma:                parámetro de aversión al riesgo

    Returns:
        dict con las 7 keys del contrato:
          'weights_array'    np.array (N,) — pesos riesgo + peso XEON.DE, suma ≈ 1
          'weights_dict'     dict — solo ETFs de riesgo con peso > MIN_WEIGHT
          'selected_tickers' list — ETFs de riesgo en cartera (≤ N_TOP_ASSETS)
          'weight_xeon'      float — complemento: 1 - suma(pesos_riesgo)
          'w_raw'            np.array (N,) — pesos brutos antes de restricciones
          'regime'           str — 'normal', 'caution' o 'crisis'
          'liquidar'         list — congelados con peso < FREEZE_EXIT_THR → liquidar
    """
    mu_BL   = bl_result['mu_BL']
    Sigma   = bl_result['Sigma']
    tickers = bl_result['tickers']
    N       = len(tickers)

    # --- Régimen de volatilidad ---
    regime, max_risk_total = detect_regime(sigma_mercado)

    # --- Pesos brutos de Merton (solo riesgo) ---
    w_raw_risk, risk_tickers, risk_indices = merton_weights_risk_only(
        mu_BL, Sigma, risk_free_rate, tickers,
        xeon_ticker=xeon_ticker, gamma=gamma
    )

    # --- Aplicar restricciones ---
    w_risk_final = apply_constraints(
        w_raw_risk,
        risk_tickers,
        categoria_por_ticker=categoria_por_ticker,
        max_risk_total=max_risk_total,
    )

    # --- Construir array completo (N elementos, XEON.DE = 0 aquí) ---
    weights_full = np.zeros(N)
    for i, orig_idx in enumerate(risk_indices):
        weights_full[orig_idx] = w_risk_final[i]

    # --- Peso de XEON.DE = complemento ---
    xeon_idx    = tickers.index(xeon_ticker) if xeon_ticker in tickers else -1
    weight_xeon = max(0.0, 1.0 - weights_full.sum())
    if xeon_idx >= 0:
        weights_full[xeon_idx] = weight_xeon

    # --- Detectar congelados a liquidar ---
    liquidar = []
    if current_weights is not None and frozen_tickers:
        liquidar = check_frozen_exits(
            current_weights, tickers, frozen_tickers, xeon_ticker
        )

    # --- Construir resultados ---
    selected     = [risk_tickers[i] for i in range(len(risk_tickers))
                    if w_risk_final[i] > MIN_WEIGHT]
    weights_dict = {risk_tickers[i]: float(w_risk_final[i])
                    for i in range(len(risk_tickers))
                    if w_risk_final[i] > MIN_WEIGHT}

    # Array bruto completo para debug
    w_raw_full = np.zeros(N)
    for i, orig_idx in enumerate(risk_indices):
        w_raw_full[orig_idx] = w_raw_risk[i] if i < len(w_raw_risk) else 0.0

    print(f"  [Merton] Régimen: {regime} | "
          f"Cap riesgo: {max_risk_total:.0%} | "
          f"XEON.DE: {weight_xeon:.1%} | "
          f"Cartera: {weights_dict}")

    return {
        'weights_array':    weights_full,
        'weights':     weights_dict,
        'selected_etfs': selected,
        'weight_xeon':      weight_xeon,
        'w_raw':            w_raw_full,
        'regime':           regime,
        'liquidar':         liquidar,
    }