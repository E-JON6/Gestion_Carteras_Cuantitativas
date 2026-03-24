# P2 — Tu tarea: Merton Multivariante (pesos óptimos)
### VERSIÓN DEFINITIVA
### Entrega: Martes por la noche

---

## Lo que tienes que hacer, en una frase

Crear el archivo `models/merton_multivariate.py`.
Dados los retornos esperados de P1, calcula **cuánto poner en cada ETF de riesgo**.
XEON.DE no entra en el cálculo — su peso es el complemento que determina P3.

---

## Cambios respecto a la versión anterior — léelos primero

1. **XEON.DE no entra en Merton.** Los pesos que calculas son solo para ETFs de riesgo y pueden sumar menos de 1. El complemento va a XEON.DE. Explicado en detalle abajo.
2. **Régimen de volatilidad.** Si la volatilidad del mercado es alta, se reduce el peso máximo total en riesgo. Esto fuerza más peso en XEON.DE automáticamente.
3. **Restricción sectorial.** Máximo 25% en ETFs de la misma categoría.
4. **Condición de salida del congelado.** Si un ETF congelado tiene peso < 2%, se liquida en el siguiente rebalanceo. Esto lo registras en el resultado para que P3 lo sepa.
5. **Ledoit-Wolf ya viene hecho en Sigma.** No necesitas regularizar tú — P1 ya lo hace.

---

## Por qué XEON.DE no entra en Merton

Merton calcula:
```
w* = (1 / gamma_eff) × Sigma^(-1) × (mu_BL - r)
```

El `mu_BL` de XEON.DE es aproximadamente igual a `r` (la tasa libre de riesgo),
porque XEON.DE replica exactamente esa tasa. Entonces:

```
mu_BL_XEON - r ≈ 0
```

Si metieras XEON.DE en el cálculo, Merton le asignaría peso ≈ 0 de todas formas,
porque su exceso de retorno sobre la tasa libre de riesgo es prácticamente cero.

Pero además hay un problema numérico: si la estimación de `mu_BL` de XEON.DE
tiene cualquier error numérico pequeño (puede ser positivo o negativo), Merton
podría asignarle un peso positivo o negativo enorme al invertir Sigma, porque
la volatilidad de XEON.DE es muy pequeña.

Por eso la solución más limpia es: **excluir XEON.DE del cálculo de Merton
y asignar su peso como complemento al final.**

---

## El régimen de volatilidad

La volatilidad del mercado no es constante. En 2020 (COVID) o en 2008 (crisis
financiera) la volatilidad de los ETFs se disparó al doble o triple de lo normal.

Si en esos momentos Merton sigue calculando pesos normales, el sistema puede
terminar con mucho riesgo justo cuando el mercado está más peligroso.

La solución es limitar el peso máximo total en ETFs de riesgo según la volatilidad:

```
Volatilidad media del universo   Peso máximo en riesgo   Mínimo en XEON.DE
─────────────────────────────    ─────────────────────   ─────────────────
< 20% anualizada (normal)        100%                    0% (puede ser 0)
20% - 30% (caution)              70%                     30%
> 30% (crisis)                   40%                     60%
```

Esto no significa que XEON.DE siempre tenga ese mínimo — significa que el sistema
no puede poner más del máximo en riesgo aunque Merton lo quiera.

---

## Lo que recibes (de P1)

```python
bl_result = {
  'mu_BL':   np.array([0.08, 0.12, 0.05, 0.001, ...]),  # N valores, incluyendo XEON.DE
  'Sigma':   np.array([[...]]),                           # N×N, incluyendo XEON.DE
  'tickers': ['IWDA.L', 'XLK', 'GLD', ..., 'XEON.DE'],  # N tickers
  ...
}
```

También recibes:
- `risk_free_rate`: float, tasa BCE
- `xeon_ticker`: string, siempre `'XEON.DE'`
- `sigma_mercado`: float, volatilidad media del universo (la calcula P4 y la pasa)
- `categoria_por_ticker`: dict `{ticker: categoria}` para la restricción sectorial

---

## Lo que entregas (a P3 y P4)

```python
{
  'weights_array':    np.array shape (N,),  # pesos de riesgo + 0.0 en XEON.DE
                                             # suma ≤ 1 (el resto va a XEON.DE via P3)
  'weights_dict':     {'IWDA.L': 0.28, ...},# solo ETFs de riesgo con peso > 0
  'selected_tickers': ['IWDA.L', 'XLK',...],# ETFs de riesgo en cartera (≤5)
  'weight_xeon':      0.22,                 # complemento: 1 - suma(pesos_riesgo)
  'w_raw':            np.array shape (N,),  # pesos brutos antes de restricciones
  'regime':           'normal',             # 'normal', 'caution' o 'crisis'
  'liquidar':         ['ETF_X'],            # congelados con peso < 2% → liquidar
}
```

---

## El código que tienes que escribir

```python
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

GAMMA            = -2     # Parámetro de utilidad. RRA = 1 - GAMMA = 3.
N_TOP_ASSETS     = 5      # Máximo de ETFs de riesgo en cartera simultáneamente
MAX_WEIGHT       = 0.40   # Peso máximo por ETF individual (40%)
MIN_WEIGHT       = 0.01   # Peso mínimo para considerar que hay posición (1%)
MAX_SECTOR_WEIGHT= 0.25   # Peso máximo por categoría sectorial (25%)
FREEZE_EXIT_THR  = 0.02   # Si un congelado cae por debajo del 2% → liquidar

# Umbrales de régimen de volatilidad
VOL_CAUTION_THR  = 0.20   # sigma > 20% → régimen caution, cap 70% en riesgo
VOL_CRISIS_THR   = 0.30   # sigma > 30% → régimen crisis, cap 40% en riesgo


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
    mu_risk  = mu_BL[risk_indices]
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
    sorted_idx = np.argsort(w)[::-1]
    w_filtered = np.zeros(N)
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
                    # Reducir proporcionalmente los ETFs de este sector
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

def compute_optimal_weights(bl_result, risk_free_rate,
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
        dict con las keys exactas del contrato (ver overview)
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
    xeon_idx = tickers.index(xeon_ticker) if xeon_ticker in tickers else -1
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
    selected = [risk_tickers[i] for i in range(len(risk_tickers))
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
        'weights_dict':     weights_dict,
        'selected_tickers': selected,
        'weight_xeon':      weight_xeon,
        'w_raw':            w_raw_full,
        'regime':           regime,
        'liquidar':         liquidar,
    }
```

---

## Cómo probar que tu código funciona

```python
"""
test_merton_multivariate.py
Ejecutar: python test_merton_multivariate.py
"""
import numpy as np
from models.merton_multivariate import (
    detect_regime, merton_weights_risk_only,
    apply_constraints, compute_optimal_weights
)

print("=== TEST 1: Régimen de volatilidad ===")
regime, cap = detect_regime(0.15)
assert regime == 'normal' and cap == 1.00, f"ERROR: esperaba normal/1.0, got {regime}/{cap}"
regime, cap = detect_regime(0.25)
assert regime == 'caution' and cap == 0.70, f"ERROR: esperaba caution/0.7"
regime, cap = detect_regime(0.35)
assert regime == 'crisis' and cap == 0.40, f"ERROR: esperaba crisis/0.4"
print("  ✓ Regímenes OK\n")

print("=== TEST 2: XEON.DE excluido de Merton ===")
tickers = ['ETF_A', 'ETF_B', 'XEON.DE']
mu_BL = np.array([0.10, 0.07, 0.025])  # XEON.DE tiene mu ≈ tasa libre de riesgo
Sigma = np.array([
    [0.04,  0.01,  0.0],
    [0.01,  0.035, 0.0],
    [0.0,   0.0,   0.000025],  # Sigma XEON muy pequeña
])
w_raw, risk_t, risk_idx = merton_weights_risk_only(
    mu_BL, Sigma, 0.025, tickers, xeon_ticker='XEON.DE'
)
assert 'XEON.DE' not in risk_t, "ERROR: XEON.DE no debe estar en risk_tickers"
assert len(w_raw) == 2, f"ERROR: debe haber 2 pesos de riesgo, hay {len(w_raw)}"
print(f"  Pesos brutos riesgo: {dict(zip(risk_t, w_raw.round(3)))}")
print("  ✓ XEON.DE excluido OK\n")

print("=== TEST 3: Pipeline completo con régimen crisis ===")
np.random.seed(42)
N = 6
tickers6 = [f'ETF_{i}' for i in range(5)] + ['XEON.DE']
bl_mock = {
    'mu_BL':   np.array([0.12, 0.09, 0.08, 0.06, 0.05, 0.025]),
    'Sigma':   np.eye(N) * 0.04 + np.random.uniform(0, 0.005, (N,N)),
    'tickers': tickers6,
}
bl_mock['Sigma'] = (bl_mock['Sigma'] + bl_mock['Sigma'].T) / 2

# En régimen crisis, máximo 40% en riesgo → mínimo 60% en XEON.DE
result = compute_optimal_weights(bl_mock, risk_free_rate=0.025,
                                 xeon_ticker='XEON.DE',
                                 sigma_mercado=0.35)  # > 30% = crisis

assert result['regime'] == 'crisis', f"ERROR: esperaba crisis, got {result['regime']}"
assert result['weight_xeon'] >= 0.60 - 0.01, \
    f"ERROR: en crisis XEON.DE debe ser ≥ 60%, got {result['weight_xeon']:.1%}"
total = result['weights_array'].sum()
assert abs(total - 1.0) < 0.01, f"ERROR: pesos no suman 1, suman {total:.3f}"
print(f"  Régimen: {result['regime']}")
print(f"  XEON.DE: {result['weight_xeon']:.1%}  → debe ser ≥ 60%")
print(f"  Suma total: {total:.4f}  → debe ser ≈ 1.0")
print("  ✓ Pipeline crisis OK\n")

print("=== TEST 4: Condición de salida de congelados ===")
current_w = np.array([0.01, 0.30, 0.30, 0.30, 0.08, 0.01])  # ETF_0 y XEON.DE con peso < 2%
from models.merton_multivariate import check_frozen_exits
liquidar = check_frozen_exits(
    current_weights=current_w,
    tickers=tickers6,
    frozen_tickers=['ETF_0'],  # ETF_0 está congelado con peso 1%
    xeon_ticker='XEON.DE',
    threshold=0.02
)
assert 'ETF_0' in liquidar, "ERROR: ETF_0 con peso 1% debería liquidarse"
print(f"  Congelados a liquidar: {liquidar}  → debe incluir ETF_0")
print("  ✓ Salida congelados OK\n")

print("=== TODOS LOS TESTS PASARON ===")
```

---

## Checklist antes de avisar al grupo

- [ ] `models/merton_multivariate.py` creado
- [ ] Los 4 tests pasan sin errores
- [ ] `compute_optimal_weights()` devuelve las 7 keys del contrato
- [ ] XEON.DE tiene peso = `1 - suma(pesos_riesgo)` en `weights_array`
- [ ] En régimen crisis, `weight_xeon` ≥ 0.60
- [ ] La suma de `weights_array` es 1.0 ± 0.01
- [ ] Nunca hay más de 5 ETFs de riesgo en `selected_tickers`

---

## Cuándo avisar por WhatsApp

| Momento | Qué escribir |
|---|---|
| Archivo creado | "P2: archivo creado, testeando" |
| Tests OK | "P2: ✓ listo para integración con P1 y P3" |
| Error raro | Pega el traceback completo |

---

## Lo que NO tienes que hacer

- No descargar datos (eso es P4)
- No tocar `black_litterman.py` ni `davis_norman.py`
- No calcular las bandas de no-transacción (eso es P3)
- No generar el Excel de órdenes (eso es P3)

---

## Dependencias

```
pip install numpy
```
