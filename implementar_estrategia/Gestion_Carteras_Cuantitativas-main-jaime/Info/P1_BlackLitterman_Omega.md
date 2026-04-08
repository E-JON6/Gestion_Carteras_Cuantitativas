

Si usas la media histórica directamente, el número es muy ruidoso — cambia mucho de
un mes a otro y da señales falsas. **Black-Litterman** soluciona esto mezclando dos fuentes:

- **Prior de equilibrio**: "sin información especial, cada activo debería rendir según
  su peso en el mercado y su correlación con el resto"
- **Views** (nuestras opiniones): "los 3 ETFs con mejor asimetría reciente por Omega
  deberían rendir por encima de lo que dice el prior"

El resultado, `mu_BL`, es un vector de retornos esperados más estable y más defensible
que una media histórica cruda.

---

## El papel de XEON.DE en este módulo — MUY IMPORTANTE

XEON.DE (ETF monetario €STR) entra en `returns_df` como una columna más.
Pero se trata diferente al resto:

**XEON.DE NO genera views de Omega.** No tiene sentido económico calcular su Omega
y poner una view sobre él. Su retorno viene de la tasa BCE, no del mercado de renta
variable. Si un mes el BCE sube tipos y XEON.DE tiene Omega alto, no significa que
"vaya bien" como activo de inversión — significa que el BCE subió tipos.

**XEON.DE SÍ entra en Sigma.** La correlación entre XEON.DE y el resto de ETFs
es información valiosa. En periodos de stress, XEON.DE tiene baja correlación con
renta variable, y eso mejora el cálculo de diversificación.

**XEON.DE SÍ entra en el prior.** Su volatilidad histórica (muy baja) hace que el
prior le asigne un retorno esperado bajo, que es correcto.

En el código, la exclusión se hace con una sola línea:
```python
omega_scores = omega_scores.drop(xeon_ticker, errors='ignore')
```

---

## Lo que recibes (de P4)

`returns_df`: DataFrame donde cada columna es un ETF y cada fila es un día.
Las columnas incluyen XEON.DE. Los valores son retornos diarios.
**CRÍTICO: P4 garantiza que este DataFrame solo contiene datos hasta la fecha
actual. Nunca datos futuros. Tú no tienes que preocuparte de esto.**

```
            IWDA.L   XLK    GLD   XEON.DE  ...
2024-01-02  0.008   0.012  -0.003  0.0001
2024-01-03 -0.002   0.004   0.001  0.0001
...
```

`risk_free_rate`: float. Tasa BCE anualizada (ej: 0.025 = 2.5%).

`xeon_ticker`: string. Siempre `'XEON.DE'`. Pasarlo explícitamente para que
el código sepa cuál excluir de las views.

---

## Lo que entregas (a P2 y P3)

Un dict con EXACTAMENTE estas keys (si cambias alguna, rompes el trabajo de los demás):

```python
{
  'mu_BL':        np.array shape (N,)    # N = todos los ETFs incluyendo XEON.DE
  'Sigma':        np.array shape (N, N)  # incluyendo XEON.DE
  'tickers':      list de N strings      # incluyendo XEON.DE
  'top_tickers':  list de ≤3 strings     # NUNCA incluye XEON.DE
  'omega_scores': pd.Series             # solo ETFs de riesgo, sin XEON.DE
  'Q':            np.array shape (K,)   # K ≤ 3
  'confidence':   np.array shape (K,)
}
```

---

## El código que tienes que escribir

```python
"""
Black-Litterman con Omega Ratio como generador de views.

Mejoras sobre la versión básica:
  1. Prior con volatilidad inversa (más robusto que pesos iguales)
  2. Omega con doble ventana de confirmación (fast + slow)
  3. Q calculado con ventana lenta (más estable)
  4. Ledoit-Wolf para Sigma (mejor condicionamiento numérico)
  5. XEON.DE excluido de views pero incluido en Sigma y prior

Referencias: Black & Litterman (1992), He & Litterman (1999),
             Ledoit & Wolf (2004).
"""

import numpy as np
import pandas as pd

# ============================================================
# PARÁMETROS
# ============================================================

OMEGA_WINDOW_FAST = 42    # Ventana corta para Omega (~2 meses). Señal reciente.
OMEGA_WINDOW_SLOW = 84    # Ventana larga para Omega (~4 meses). Confirmación.
                          # Un ETF solo genera view si está en top-3 en AMBAS.
OMEGA_THRESHOLD   = 0.0   # Umbral: 0 = distinguir días positivos de negativos
N_TOP_VIEWS       = 3     # ETFs que generan views (siempre ETFs de riesgo)
TAU               = 0.05  # Escalar de incertidumbre del prior. Estándar en la literatura.
DELTA             = 2.5   # Aversión al riesgo del mercado implícito (He & Litterman 1999)
EWMA_LAMBDA       = 0.94  # Factor de decaimiento para EWMA de Sigma (RiskMetrics)


# ============================================================
# PASO 1: OMEGA RATIO
# ============================================================

def compute_omega_ratio(returns_series, window, threshold=OMEGA_THRESHOLD):
    """
    Omega = sum(ganancias por encima del umbral) / sum(pérdidas por debajo del umbral)

    Si Omega > 1: las ganancias recientes superan las pérdidas → buena asimetría
    Si Omega < 1: las pérdidas dominan → mala asimetría

    Caputamos el valor en 10.0 para evitar infinitos cuando no hay pérdidas.
    """
    recent = returns_series.dropna().iloc[-window:]

    if len(recent) < window // 2:
        return np.nan  # No hay suficientes datos todavía

    gains  = (recent[recent > threshold] - threshold).sum()
    losses = (threshold - recent[recent < threshold]).sum()

    if losses < 1e-10:
        return 10.0 if gains > 1e-10 else 1.0

    return min(gains / losses, 10.0)  # Cap en 10 para estabilidad


def rank_etfs_by_omega(returns_df, window, xeon_ticker='XEON.DE'):
    """
    Calcula Omega de todos los ETFs de riesgo y los ordena.
    XEON.DE se excluye antes de calcular — no tiene sentido su Omega.

    Returns:
        pd.Series con Omega de cada ETF de riesgo, ordenado de mayor a menor.
    """
    scores = {}
    for ticker in returns_df.columns:
        if ticker == xeon_ticker:
            continue  # XEON.DE nunca genera view
        scores[ticker] = compute_omega_ratio(returns_df[ticker], window)

    return pd.Series(scores).dropna().sort_values(ascending=False)


def get_confirmed_top(returns_df, xeon_ticker='XEON.DE', n_top=N_TOP_VIEWS):
    """
    Selecciona los top ETFs con confirmación en DOS ventanas temporales.

    Lógica:
      - Calcula Omega con ventana rápida (42 días) y ventana lenta (84 días).
      - Un ETF entra en las views SOLO si está en top-N en AMBAS ventanas.
      - Si no hay suficientes confirmados, usa solo la ventana rápida como fallback.

    Por qué dos ventanas:
      Con una sola ventana de 42 días, un ETF puede entrar en el top-3 por un
      buen mes puntual y salir a la semana siguiente. Eso genera views inestables
      y muchos rebalanceos innecesarios. La confirmación en la ventana lenta
      asegura que la señal lleva al menos 4 meses siendo consistente.

    Returns:
        tuple (top_tickers, omega_fast, omega_slow)
    """
    omega_fast = rank_etfs_by_omega(returns_df, OMEGA_WINDOW_FAST, xeon_ticker)
    omega_slow = rank_etfs_by_omega(returns_df, OMEGA_WINDOW_SLOW, xeon_ticker)

    top_fast = set(omega_fast.head(n_top).index)
    top_slow = set(omega_slow.head(n_top).index)

    # Intersección: en ambos rankings
    confirmed = top_fast & top_slow

    if len(confirmed) >= max(1, n_top // 2):
        # Suficientes confirmados → ordenar por Omega rápido (señal más reciente)
        top_tickers = [t for t in omega_fast.index if t in confirmed][:n_top]
    else:
        # Fallback: solo ventana rápida (puede pasar al principio del backtest)
        top_tickers = list(omega_fast.head(n_top).index)

    return top_tickers, omega_fast, omega_slow


# ============================================================
# PASO 2: MATRIZ DE COVARIANZAS (Ledoit-Wolf + EWMA)
# ============================================================

def compute_covariance_matrix(returns_df, ewma_lambda=EWMA_LAMBDA):
    """
    Estima la matriz de covarianzas con EWMA + shrinkage de Ledoit-Wolf.

    Dos pasos:
      1. EWMA: da más peso a los datos recientes, captura volatilidad actual.
      2. Ledoit-Wolf: mezcla la covarianza EWMA con la identidad escalada para
         evitar que la matriz sea "inestable" al invertirla en Merton.
         Sin shrinkage, matrices con muchos activos tienen eigenvalores
         muy pequeños que al invertirlas dan números enormes e inestables.

    La covarianza resultante incluye XEON.DE — es correcto porque su correlación
    con el resto es información valiosa para el prior y para Merton.

    Returns:
        np.array (N × N) — covarianza ANUALIZADA, simétrica, invertible.
    """
    T, N = returns_df.shape

    # --- EWMA ---
    # pandas ewm calcula la covarianza con pesos exponenciales
    # alpha = 1 - lambda (pandas usa alpha, nosotros tenemos lambda)
    cov_ewma = returns_df.ewm(
        alpha=1 - ewma_lambda,
        min_periods=min(21, T // 4)
    ).cov()

    # ewm().cov() devuelve un DataFrame MultiIndex (fecha, ticker) × ticker
    # Cogemos el último snapshot (el de hoy)
    last_date = cov_ewma.index.get_level_values(0)[-1]
    Sigma_daily = cov_ewma.loc[last_date].values.astype(float)

    # Anualizar: multiplicar por 252 días hábiles
    Sigma = Sigma_daily * 252

    # Simetrizar (pequeños errores numéricos de EWMA)
    Sigma = (Sigma + Sigma.T) / 2.0

    # --- Ledoit-Wolf (shrinkage hacia identidad escalada) ---
    # El factor óptimo de shrinkage minimiza el error cuadrático medio esperado.
    # Ref: Ledoit & Wolf (2004), "A well-conditioned estimator for
    # large-dimensional covariance matrices."
    if T > N:  # Solo si tenemos más observaciones que activos
        mu_shrink = np.trace(Sigma) / N           # Escalar: varianza media
        F = mu_shrink * np.eye(N)                  # Target: identidad escalada
        delta_sq = np.sum((Sigma - F) ** 2)

        if delta_sq > 1e-12:
            # Estimar la varianza de los elementos de la covarianza muestral
            # (Chen et al. 2010, fórmula simplificada)
            X = returns_df.values * np.sqrt(252)   # Retornos anualizados para consistencia
            X = X - X.mean(axis=0)
            beta_sq = 0.0
            for i in range(T):
                xi = X[i, :].reshape(-1, 1)
                beta_sq += np.sum((xi @ xi.T - Sigma) ** 2)
            beta_sq /= (T ** 2)

            alpha_lw = min(beta_sq / delta_sq, 1.0)
            Sigma = (1 - alpha_lw) * Sigma + alpha_lw * F

    # --- Proyección a semidefinida positiva ---
    # Por si algún eigenvalor quedó negativo por errores numéricos
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-6)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma = (Sigma + Sigma.T) / 2.0  # Re-simetrizar

    return Sigma


# ============================================================
# PASO 3: PRIOR DE EQUILIBRIO (Volatilidad Inversa)
# ============================================================

def compute_equilibrium_prior(Sigma, risk_free=0.0, delta=DELTA):
    """
    Calcula el prior de equilibrio de Black-Litterman usando pesos de
    volatilidad inversa en lugar de pesos iguales.

    Fórmula: pi = r + delta × Sigma × w_mkt

    Diferencia con pesos iguales:
      Pesos iguales: w_mkt_i = 1/N para todos
      Volatilidad inversa: w_mkt_i = (1/sigma_i) / sum(1/sigma_j)

    Por qué es mejor:
      - Los activos menos volátiles (como XEON.DE, bonos) tienen mayor
        capitalización de mercado implícita en equilibrio.
      - Si GLD tiene sigma=12% y QQQ tiene sigma=22%, en equilibrio el mercado
        pone más dinero en GLD (menos riesgo por unidad de retorno).
      - Con pesos iguales estamos diciendo que el mercado pone lo mismo en
        un ETF de semiconductores ultra-volátil que en un bono del tesoro.
        Eso no es realista.
      - La volatilidad inversa es la aproximación estándar cuando no se tienen
        datos de capitalización de mercado real.

    XEON.DE tiene sigma muy baja (~0.5% anualizada), así que tendrá un peso
    alto en el prior, lo que da un prior conservador. Eso es correcto.
    """
    N = Sigma.shape[0]
    vol = np.sqrt(np.diag(Sigma))
    vol = np.maximum(vol, 1e-6)  # Evitar división por cero

    inv_vol = 1.0 / vol
    w_mkt = inv_vol / inv_vol.sum()

    pi = risk_free + delta * (Sigma @ w_mkt)
    return pi


# ============================================================
# PASO 4: CONSTRUIR VIEWS
# ============================================================

def build_views(top_tickers, omega_fast, omega_slow, returns_df, all_tickers):
    """
    Construye P, Q y confidence para Black-Litterman.

    P: matriz (K × N). Fila k tiene un 1 en el ETF k y 0 en el resto.
       XEON.DE nunca aparece aquí.

    Q: vector (K,). Retorno esperado de cada view.
       Se usa la ventana LENTA (84 días) para ser más conservador y estable.
       Con la ventana rápida (42 días) Q es más ruidoso y puede exagerar.

    confidence: vector (K,). Confianza en cada view.
       Se calcula como el promedio geométrico del Omega rápido y lento.
       Un ETF que está en el top en AMBAS ventanas tiene más confianza
       que uno que solo lo está en la rápida.

    Args:
        top_tickers:  lista de tickers que generan views (sin XEON.DE)
        omega_fast:   pd.Series con Omega de ventana 42d
        omega_slow:   pd.Series con Omega de ventana 84d
        returns_df:   DataFrame completo de retornos
        all_tickers:  lista con TODOS los tickers en el orden de returns_df

    Returns:
        P: np.array (K × N)
        Q: np.array (K,)
        confidence: np.array (K,)
    """
    N = len(all_tickers)
    K = len(top_tickers)

    P = np.zeros((K, N))
    Q = np.zeros(K)
    confidence = np.zeros(K)

    for k, ticker in enumerate(top_tickers):
        if ticker not in all_tickers:
            continue

        # Columna en la matriz P
        col_idx = all_tickers.index(ticker)
        P[k, col_idx] = 1.0

        # Q: retorno anualizado de la ventana LENTA (más estable)
        # La ventana lenta de 84 días reduce el ruido respecto a los 42 días
        recent = returns_df[ticker].dropna().iloc[-OMEGA_WINDOW_SLOW:]
        Q[k] = recent.mean() * 252

        # Confianza: promedio geométrico de Omega fast y slow
        # Si el ETF solo está en el top por la ventana rápida, la confianza es menor
        cf = omega_fast.get(ticker, 1.0)
        cs = omega_slow.get(ticker, cf)  # Si no hay slow, usar fast
        confidence[k] = np.sqrt(cf * cs)  # Promedio geométrico

    # Normalizar confianza para que el máximo sea 1.0
    if confidence.max() > 1e-10:
        confidence = confidence / confidence.max()

    return P, Q, confidence


# ============================================================
# PASO 5: POSTERIOR DE BLACK-LITTERMAN
# ============================================================

def compute_bl_posterior(pi, Sigma, P, Q, confidence, tau=TAU):
    """
    Calcula mu_BL: el vector de retornos esperados posterior.

    Fórmula estándar de He & Litterman (1999):
      mu_BL = [(tau*Sigma)^-1 + P' * Omega_BL^-1 * P]^-1
              [(tau*Sigma)^-1 * pi + P' * Omega_BL^-1 * Q]

    Donde Omega_BL es la matriz de incertidumbre de las views:
      Omega_BL[k,k] = (1/confidence[k]) × tau × (P[k,:] @ Sigma @ P[k,:].T)

    Interpretación del resultado:
      - Si confidence es 0 en todas las views → mu_BL ≈ pi (nos quedamos en el prior)
      - Si confidence es alta → mu_BL se acerca a Q (confiamos en las views)
      - XEON.DE recibe mu_BL del prior directamente (nadie opina sobre él)
    """
    N = Sigma.shape[0]
    K = P.shape[0]

    # Regularización numérica
    reg = np.eye(N) * 1e-8

    tau_Sigma = tau * Sigma
    tau_Sigma_inv = np.linalg.inv(tau_Sigma + reg)

    # Construir Omega_BL: incertidumbre de las views
    Omega_BL = np.zeros((K, K))
    for k in range(K):
        var_view = float(P[k, :] @ tau_Sigma @ P[k, :].T)
        Omega_BL[k, k] = (1.0 / max(confidence[k], 0.01)) * var_view

    Omega_BL_inv = np.diag(1.0 / np.diag(Omega_BL))

    # Posterior
    A = tau_Sigma_inv + P.T @ Omega_BL_inv @ P
    b = tau_Sigma_inv @ pi + P.T @ Omega_BL_inv @ Q
    A_inv = np.linalg.inv(A + reg)
    mu_BL = A_inv @ b

    return mu_BL


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def run_black_litterman(returns_df, risk_free_rate, xeon_ticker='XEON.DE', date=None):
    """
    Pipeline completo: Omega (con confirmación) → views → BL → mu_BL.

    Esta es la función que P4 llama desde el motor.

    IMPORTANTE sobre returns_df:
      - Debe incluir XEON.DE como columna.
      - P4 garantiza que solo contiene datos hasta la fecha actual (sin look-ahead).
      - Debe tener al menos OMEGA_WINDOW_SLOW + 10 = 94 días de datos.

    Args:
        returns_df:      pd.DataFrame (días × tickers), incluye XEON.DE
        risk_free_rate:  float, tasa BCE anualizada (ej: 0.025)
        xeon_ticker:     str, nombre del ticker monetario (default 'XEON.DE')
        date:            pd.Timestamp, fecha actual (solo para logging)

    Returns:
        dict con las keys exactas que esperan P2 y P3 (ver contrato en el overview)
    """
    tickers = list(returns_df.columns)
    N = len(tickers)

    # Verificación de datos mínimos
    min_days = OMEGA_WINDOW_SLOW + 10
    if len(returns_df) < min_days:
        raise ValueError(
            f"Necesitas al menos {min_days} días de datos. "
            f"Solo tienes {len(returns_df)}. "
            f"P4 debe pasar más historia en el warm-up."
        )

    # --- Paso 1: Omega con confirmación doble ---
    # XEON.DE se excluye automáticamente dentro de get_confirmed_top
    top_tickers, omega_fast, omega_slow = get_confirmed_top(
        returns_df, xeon_ticker=xeon_ticker, n_top=N_TOP_VIEWS
    )

    # --- Paso 2: Sigma (incluye XEON.DE) ---
    Sigma = compute_covariance_matrix(returns_df)

    # --- Paso 3: Views (sin XEON.DE) ---
    P, Q, confidence = build_views(top_tickers, omega_fast, omega_slow,
                                   returns_df, tickers)

    # --- Paso 4: Prior (incluye XEON.DE, con volatilidad inversa) ---
    pi = compute_equilibrium_prior(Sigma, risk_free=risk_free_rate)

    # --- Paso 5: Posterior ---
    mu_BL = compute_bl_posterior(pi, Sigma, P, Q, confidence)

    if date:
        date_str = date.date() if hasattr(date, 'date') else date
        xeon_idx = tickers.index(xeon_ticker) if xeon_ticker in tickers else -1
        xeon_mu = f"{mu_BL[xeon_idx]:.3f}" if xeon_idx >= 0 else "N/A"
        print(f"  [BL] {date_str} | Top-3: {top_tickers} | "
              f"mu_BL medio (riesgo): {np.mean([mu_BL[tickers.index(t)] for t in top_tickers]):.3f} | "
              f"mu_BL XEON: {xeon_mu}")

    return {
        'mu_BL':        mu_BL,
        'Sigma':        Sigma,
        'tickers':      tickers,
        'top_tickers':  top_tickers,
        'omega_scores': omega_fast,   # Solo ETFs de riesgo
        'Q':            Q,
        'confidence':   confidence,
    }
```

---

## Cómo probar que tu código funciona

```python
"""
test_black_litterman.py
Ejecutar: python test_black_litterman.py
"""
import numpy as np
import pandas as pd
from models.black_litterman import (
    compute_omega_ratio, rank_etfs_by_omega,
    get_confirmed_top, run_black_litterman
)

print("=== TEST 1: Omega Ratio básico ===")
retornos_buenos = pd.Series([0.01, 0.02, -0.005, 0.015, -0.003, 0.012])
omega = compute_omega_ratio(retornos_buenos, window=6)
assert omega > 1.0, f"ERROR: Omega debería ser > 1, got {omega}"
retornos_malos = pd.Series([-0.02, -0.015, 0.005, -0.018, 0.003, -0.01])
omega_malo = compute_omega_ratio(retornos_malos, window=6)
assert omega_malo < 1.0, f"ERROR: Omega debería ser < 1, got {omega_malo}"
print("  ✓ Omega básico OK\n")

print("=== TEST 2: XEON.DE excluido de views ===")
np.random.seed(42)
n_days = 100
tickers = ['ETF_A', 'ETF_B', 'ETF_C', 'XEON.DE']
returns = np.random.randn(n_days, 4) * 0.01
# Hacer que XEON.DE tenga el Omega más alto artificialmente
returns[:, 3] = np.abs(returns[:, 3]) * 0.1  # solo subidas, muy bueno
returns_df = pd.DataFrame(returns, columns=tickers)

omega_scores = rank_etfs_by_omega(returns_df, window=42, xeon_ticker='XEON.DE')
assert 'XEON.DE' not in omega_scores.index, "ERROR: XEON.DE no debería aparecer en omega_scores"
print(f"  Omega scores: {omega_scores.to_dict()}")
print("  ✓ XEON.DE excluido de Omega OK\n")

print("=== TEST 3: Pipeline completo con XEON.DE ===")
np.random.seed(42)
n_days = 100
tickers_full = ['ETF_A', 'ETF_B', 'ETF_C', 'ETF_D', 'XEON.DE']
returns_full = np.random.randn(n_days, 5) * 0.01
returns_full[:, 0] += 0.001   # ETF_A tiene buen drift
returns_full[:, 4]  = 0.0001  # XEON.DE casi sin variación (ETF monetario)
returns_df_full = pd.DataFrame(returns_full, columns=tickers_full)

result = run_black_litterman(returns_df_full, risk_free_rate=0.025,
                             xeon_ticker='XEON.DE')

assert result['mu_BL'].shape == (5,), "ERROR: mu_BL debe tener forma (5,)"
assert result['Sigma'].shape == (5, 5), "ERROR: Sigma debe tener forma (5, 5)"
assert 'XEON.DE' not in result['top_tickers'], "ERROR: XEON.DE no debe estar en top_tickers"
assert np.allclose(result['Sigma'], result['Sigma'].T, atol=1e-6), "ERROR: Sigma no simétrica"
assert 'XEON.DE' in result['tickers'], "ERROR: XEON.DE debe estar en la lista de tickers"

xeon_idx = result['tickers'].index('XEON.DE')
print(f"  mu_BL XEON.DE: {result['mu_BL'][xeon_idx]:.4f}  → debe ser bajo (~tasa BCE)")
print(f"  Top tickers: {result['top_tickers']}  → no debe incluir XEON.DE")
print("  ✓ Pipeline completo OK\n")

print("=== TODOS LOS TESTS PASARON ===")
```

---

## Checklist antes de avisar al grupo

- [ ] `models/black_litterman.py` creado con el código de arriba
- [ ] Los 3 tests pasan sin errores
- [ ] `run_black_litterman()` devuelve el dict con las 7 keys exactas del contrato
- [ ] XEON.DE nunca aparece en `top_tickers` ni en `omega_scores`
- [ ] XEON.DE sí aparece en `tickers`, `mu_BL` y `Sigma`
- [ ] `mu_BL` tiene shape `(N,)` — no `(N,1)`
- [ ] `Sigma` es simétrica (test lo verifica)

---

## Cuándo avisar por WhatsApp

| Momento | Qué escribir |
|---|---|
| Archivo creado aunque no testado | "P1: archivo creado, testeando" |
| Tests OK | "P1: ✓ listo, entrego a P2" |
| Error raro | Pega el traceback completo, no esperes más de 30 min |

---

## Lo que NO tienes que hacer

- No tocar `davis_norman.py` ni `engine_multiasset.py`
- No descargar datos tú mismo (eso es P4)
- No calcular pesos de cartera (eso es P2)
- No preocuparte del look-ahead bias — P4 lo garantiza pasando solo datos hasta `date`

---

## Dependencias

```
pip install numpy pandas
```
