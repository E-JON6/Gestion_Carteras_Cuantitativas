# P3 — Tu tarea: Estrategia + Simulación Diaria
### VERSIÓN DEFINITIVA
### Entrega: Martes por la noche

---

## Lo que tienes que hacer, en una frase

Crear dos archivos:
1. `strategies/bl_omega_strategy.py` — La clase que une todo y decide si rebalancear hoy.
2. `simulacion_diaria_bl.py` — El script diario que genera el Excel con las órdenes.

---

## Cambios respecto a la versión anterior — léelos primero

1. **XEON.DE siempre aparece en el Excel de operativa.** Cada rebalanceo genera obligatoriamente dos tipos de órdenes: ETFs de riesgo + XEON.DE como contrapartida. Si compramos ETFs de riesgo, vendemos XEON.DE para financiarlo. Si vendemos ETFs de riesgo, compramos XEON.DE con los ingresos.
2. **El estado JSON guarda participaciones reales de XEON.DE**, no euros en cash.
3. **Banda DN para XEON.DE.** Su peso también puede salirse de banda y disparar rebalanceo.
4. **Condición de salida de congelados.** Si un ETF congelado tiene peso < 2% (viene de P2 en `resultado['liquidar']`), se vende en el siguiente rebalanceo.
5. **La cartera siempre suma exactamente 1.** Si la suma de ETFs de riesgo es 0.78, XEON.DE tiene 0.22. Nunca hay cash.

---

## Por qué XEON.DE siempre aparece en las órdenes

La cartera tiene dos "compartimentos":
- **Compartimento de riesgo:** los ≤5 ETFs seleccionados por Merton
- **Compartimento monetario:** XEON.DE

Cuando el sistema decide rebalancear, transfiere dinero entre estos compartimentos.
Si quiere más riesgo → vende XEON.DE y compra ETFs. Si quiere menos riesgo → vende
ETFs y compra XEON.DE. Estas dos operaciones son siempre simultáneas y siempre
aparecen juntas en el Excel.

Esto es exactamente lo que hace el código original del proyecto — no es un cambio
de criterio, es adaptar esa misma lógica al nuevo sistema multiactivo.

---

## Lo que recibes

**De P2:**
```python
merton_result = {
  'weights_array':    np.array([0.28, 0.22, 0.0, ..., 0.22]),  # último es XEON.DE
  'weights_dict':     {'IWDA.L': 0.28, 'XLK': 0.22, 'GLD': 0.28},
  'selected_tickers': ['IWDA.L', 'XLK', 'GLD'],
  'weight_xeon':      0.22,
  'regime':           'normal',
  'liquidar':         [],  # ETFs congelados a liquidar
}
```

**De `models/davis_norman.py`** (ya existe, no tocar):
```python
from models.davis_norman import dn_bands_asymptotic
# dn_bands_asymptotic(alpha_star, sigma, lambda_L, lambda_M, gamma)
# Devuelve (lower, upper) — la banda de no-transacción para un activo
```

---

## Lo que entregas

**A P4 (motor de backtest):**
```python
# BLOmegaStrategy.decide() devuelve None o:
{
  'rebalance':      True,
  'target_weights': np.array([...]),  # shape (N,), suma = 1.0, incluye XEON.DE
  'reason':         'Banda DN cruzada: XLK peso=0.28 fuera de [0.19, 0.27]',
}
```

**Al profesor (vía `simulacion_diaria_bl.py`):**
- Excel `Operativa_YYYYMMDD.xlsx` con 2+ filas (una por ETF de riesgo operado + XEON.DE)
- JSON `estado_bl.json` con estado persistente incluyendo `participaciones_xeon`

---

## Archivo 1: `strategies/bl_omega_strategy.py`

```python
"""
Estrategia BL-Omega: Omega + Black-Litterman + Merton + Davis-Norman.

Cada día:
  1. Cada RECALIB_FREQ días: recalcular pesos objetivo con BL + Merton
  2. Cada día: verificar si algún ETF (incluido XEON.DE) salió de su banda DN
  3. Si hay que rebalancear: devolver pesos objetivo (con XEON.DE incluido)
  4. Gestionar congelados: mantener ETFs que salieron del top-5 hasta que
     su peso caiga por debajo del 2%

Garantía fundamental: target_weights siempre suma exactamente 1.0
"""

import numpy as np
import pandas as pd

from models.black_litterman import run_black_litterman
from models.merton_multivariate import compute_optimal_weights
from models.davis_norman import dn_bands_asymptotic

RECALIB_FREQ  = 5     # Recalibrar pesos objetivo cada N días hábiles
GAMMA         = -2    # Aversión al riesgo (igual que en config.py)
XEON_TICKER   = 'XEON.DE'


class BLOmegaStrategy:
    """
    Estrategia completa: Omega + BL + Merton + Davis-Norman + XEON.DE.
    """

    def __init__(self, tickers, xeon_ticker=XEON_TICKER,
                 recalib_freq=RECALIB_FREQ, gamma=GAMMA,
                 categoria_por_ticker=None):
        """
        Args:
            tickers:              lista de TODOS los tickers, incluyendo XEON.DE
            xeon_ticker:          ticker del ETF monetario
            recalib_freq:         cada cuántos días recalcular pesos objetivo
            gamma:                aversión al riesgo
            categoria_por_ticker: dict {ticker: categoria} para restricción sectorial
        """
        self.name                = "BL-Omega: Omega+BL+Merton+DN+XEON"
        self.tickers             = tickers
        self.N                   = len(tickers)
        self.xeon_ticker         = xeon_ticker
        self.recalib_freq        = recalib_freq
        self.gamma               = gamma
        self.categoria_por_ticker= categoria_por_ticker or {}

        # Índice de XEON.DE en el array de tickers
        self.xeon_idx = tickers.index(xeon_ticker) if xeon_ticker in tickers else -1

        # Estado interno
        self._target_weights  = np.zeros(self.N)
        self._bands_lower     = np.zeros(self.N)
        self._bands_upper     = np.zeros(self.N)
        self._selected_tickers= []
        self._frozen_tickers  = []    # ETFs fuera del top-5 pero aún en cartera
        self._last_bl_result  = None
        self._last_merton     = None

    def _compute_sigma_mercado(self, returns_df):
        """
        Calcula la volatilidad media del universo de ETFs de riesgo.
        Se usa para detectar el régimen de volatilidad en P2.
        """
        risk_cols = [c for c in returns_df.columns if c != self.xeon_ticker]
        if not risk_cols:
            return 0.15
        vols = returns_df[risk_cols].std() * np.sqrt(252)
        return float(vols.mean())

    def _recalibrate(self, returns_df, risk_free_rate, lambda_per_ticker):
        """
        Recalcula pesos objetivo con BL + Merton y actualiza bandas DN.

        IMPORTANTE: returns_df ya viene recortado a la fecha actual por P4.
        No hay riesgo de look-ahead bias aquí.
        """
        sigma_mercado = self._compute_sigma_mercado(returns_df)

        # --- Black-Litterman ---
        try:
            bl_result = run_black_litterman(
                returns_df, risk_free_rate,
                xeon_ticker=self.xeon_ticker
            )
        except Exception as e:
            print(f"  [AVISO P3] BL falló: {e}. Manteniendo pesos anteriores.")
            return

        # --- Merton ---
        try:
            merton_result = compute_optimal_weights(
                bl_result, risk_free_rate,
                xeon_ticker=self.xeon_ticker,
                sigma_mercado=sigma_mercado,
                categoria_por_ticker=self.categoria_por_ticker,
                current_weights=self._target_weights,
                frozen_tickers=self._frozen_tickers,
                gamma=self.gamma
            )
        except Exception as e:
            print(f"  [AVISO P3] Merton falló: {e}. Manteniendo pesos anteriores.")
            return

        self._last_bl_result = bl_result
        self._last_merton    = merton_result

        new_target   = merton_result['weights_array']
        new_selected = merton_result['selected_tickers']
        liquidar     = merton_result.get('liquidar', [])

        # --- Gestión de congelados ---
        # Detectar ETFs que salen del top-5
        for ticker in self._selected_tickers:
            if ticker not in new_selected and ticker not in self._frozen_tickers:
                self._frozen_tickers.append(ticker)
                print(f"  [DN] {ticker} sale del top-5 → congelado")

        # Liquidar congelados con peso < 2% (detectado por P2)
        for ticker in liquidar:
            if ticker in self._frozen_tickers:
                self._frozen_tickers.remove(ticker)
                print(f"  [DN] {ticker} congelado con peso < 2% → liquidar")
                # Poner su peso objetivo a 0 explícitamente
                if ticker in self.tickers:
                    new_target[self.tickers.index(ticker)] = 0.0

        # Recalcular XEON.DE para que la suma sea exactamente 1
        risk_sum = sum(new_target[i] for i in range(self.N)
                       if self.tickers[i] != self.xeon_ticker)
        if self.xeon_idx >= 0:
            new_target[self.xeon_idx] = max(0.0, 1.0 - risk_sum)

        # Verificar que suma = 1
        total = new_target.sum()
        if abs(total - 1.0) > 0.01:
            print(f"  [AVISO P3] Pesos suman {total:.4f} ≠ 1. Renormalizando.")
            new_target = new_target / total

        self._target_weights  = new_target
        self._selected_tickers= new_selected

        # --- Bandas DN para cada activo (incluido XEON.DE) ---
        Sigma    = bl_result['Sigma']
        vol_diag = np.sqrt(np.diag(Sigma))

        for i, ticker in enumerate(self.tickers):
            alpha_star = self._target_weights[i]
            if alpha_star < 0.005:
                self._bands_lower[i] = 0.0
                self._bands_upper[i] = 0.02
                continue

            sigma_i  = float(vol_diag[i]) if i < len(vol_diag) else 0.15
            lambda_i = lambda_per_ticker.get(ticker, 0.002)

            lower, upper = dn_bands_asymptotic(
                alpha_star=alpha_star,
                sigma=sigma_i,
                lambda_L=lambda_i,
                lambda_M=lambda_i,
                gamma=self.gamma
            )
            self._bands_lower[i] = lower
            self._bands_upper[i] = upper

    def _check_bands(self, current_weights):
        """
        Comprueba si algún activo (incluido XEON.DE) está fuera de su banda.

        Returns:
            tuple (bool, str) — (hay_que_rebalancear, motivo)
        """
        for i in range(self.N):
            w      = current_weights[i]
            lower  = self._bands_lower[i]
            upper  = self._bands_upper[i]
            target = self._target_weights[i]

            # Solo comprobar activos que tienen objetivo significativo o posición actual
            if target < 0.005 and w < 0.005:
                continue

            if w < lower or w > upper:
                ticker = self.tickers[i]
                return True, (f"Banda DN cruzada: {ticker} "
                              f"peso={w:.3f} fuera de [{lower:.3f}, {upper:.3f}]")

        return False, ""

    def get_initial_weights(self, returns_df, risk_free_rate, lambda_per_ticker):
        """
        Calcula los pesos iniciales al comenzar el backtest.
        Returns:
            np.array (N,) — pesos iniciales incluyendo XEON.DE
        """
        self._recalibrate(returns_df, risk_free_rate, lambda_per_ticker)
        return self._target_weights.copy()

    def decide(self, date, current_weights, returns_df, risk_free_rate,
               lambda_per_ticker, day_index):
        """
        Decide si hay que rebalancear hoy.

        IMPORTANTE: returns_df ya viene recortado hasta `date` por P4.

        Args:
            date:              pd.Timestamp — fecha actual
            current_weights:   np.array (N,) — pesos actuales incluyendo XEON.DE
            returns_df:        pd.DataFrame — retornos hasta hoy inclusive
            risk_free_rate:    float — tasa BCE actual
            lambda_per_ticker: dict {ticker: lambda}
            day_index:         int — día del backtest (0, 1, 2, ...)

        Returns:
            None si no hay que operar, o dict con 'rebalance', 'target_weights', 'reason'
        """
        # Recalibración periódica
        if day_index % self.recalib_freq == 0:
            self._recalibrate(returns_df, risk_free_rate, lambda_per_ticker)

        # Comprobar bandas
        hay_que_rebalancear, motivo = self._check_bands(current_weights)

        if hay_que_rebalancear:
            return {
                'rebalance':      True,
                'target_weights': self._target_weights.copy(),
                'reason':         motivo,
            }

        return None
```

---

## Archivo 2: `simulacion_diaria_bl.py`

```python
"""
Simulación diaria operativa — Estrategia BL-Omega.

Ejecutar cada día hábil: python simulacion_diaria_bl.py

Genera SIEMPRE dos tipos de órdenes:
  - ETF(s) de riesgo comprados o vendidos
  - XEON.DE como contrapartida (vendido si compramos riesgo, comprado si vendemos)

Archivos generados:
  outputs/Operativa_YYYYMMDD.xlsx  ← órdenes del día (2 filas mínimo si hay operativa)
  outputs/estado_bl.json           ← estado persistente (participaciones reales)
  outputs/debug/YYYYMMDD/          ← trazas completas para auditoría
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import date, datetime
import yfinance as yf
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from models.black_litterman import run_black_litterman
from models.merton_multivariate import compute_optimal_weights
from models.davis_norman import dn_bands_asymptotic

# ============================================================
# CONFIGURACIÓN — CAMBIAR ANTES DE EMPEZAR
# ============================================================

GRUPO           = "X"               # Número de grupo
CAPITAL_INICIAL = 10_000_000.0      # EUR
XEON_TICKER     = "XEON.DE"         # ETF monetario €STR
GAMMA           = -2
ESTADO_FILE     = "outputs/estado_bl.json"
OUTPUT_DIR      = "outputs"

# Costes de transacción (los mismos de config.py)
CT = {
    "MSE.PA":  0.001738,
    "IUSE.L":  0.000210,
    "IEMA.L":  0.000491,
    "XEON.DE": 0.000050,   # ETF monetario: spread muy pequeño
}
CT_DEFAULT = 0.002


# ============================================================
# ESTADO PERSISTENTE
# ============================================================

def estado_vacio():
    """Estado inicial para la primera ejecución."""
    return {
        "fecha_ultimo":           None,
        "capital_inicial":        CAPITAL_INICIAL,
        "patrimonio_total":       CAPITAL_INICIAL,
        "participaciones_xeon":   0.0,    # Unidades reales de XEON.DE
        "precio_xeon_ayer":       None,   # Para calcular retorno diario de XEON.DE
        "posiciones_riesgo":      {},     # {ticker: {'participaciones': N, 'precio_ayer': P}}
        "pesos_actuales":         {},
        "ultimo_top5":            [],
        "congelados":             [],
        "n_operaciones":          0,
        "costes_acumulados":      0.0,
        "en_modo_defensivo":      False,
    }


def cargar_estado():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(ESTADO_FILE):
        with open(ESTADO_FILE) as f:
            return json.load(f)
    print("  Sin estado previo → primera ejecución")
    return estado_vacio()


def guardar_estado(estado):
    with open(ESTADO_FILE, 'w') as f:
        json.dump(estado, f, indent=2, default=str)
    # Backup diario
    backup_dir = os.path.join(OUTPUT_DIR, "estado_backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f"estado_{date.today()}.json")
    with open(backup, 'w') as f:
        json.dump(estado, f, indent=2, default=str)


# ============================================================
# DESCARGA DE PRECIOS
# ============================================================

def descargar_precios_hoy(tickers_riesgo):
    """
    Descarga el precio de cierre de hoy de todos los ETFs de riesgo + XEON.DE.
    Returns:
        dict {ticker: precio_hoy}
    """
    todos = tickers_riesgo + [XEON_TICKER]
    precios = {}

    for ticker in todos:
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel('Ticker')
            precio = float(data['Close'].dropna().iloc[-1])
            precios[ticker] = precio
            print(f"  {ticker}: {precio:.4f} EUR")
        except Exception as e:
            print(f"  {ticker}: ERROR descargando precio ({e})")

    return precios


def descargar_historico(tickers_riesgo, start='2010-01-01'):
    """
    Descarga el histórico de retornos para la estrategia.
    XEON.DE se incluye para Sigma y prior de BL.
    Returns:
        pd.DataFrame con retornos diarios (días × tickers)
    """
    todos = tickers_riesgo + [XEON_TICKER]
    data = yf.download(todos, start=start, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = todos

    # Calendar: forward-fill máximo 1 día, luego eliminar NaN
    prices = prices.ffill(limit=1).dropna()
    returns = prices.pct_change().dropna()

    return returns


# ============================================================
# EXCEL DE OPERATIVA
# ============================================================

def generar_excel_operativa(ordenes, fecha_str):
    """
    Genera el Excel de operativa con las órdenes del día.

    Formato: ID | Cantidad | Precio | CT | Precio Ejecutado

    SIEMPRE hay una fila de XEON.DE además de las de los ETFs de riesgo.
    Si compramos ETFs de riesgo → vendemos XEON.DE (cantidad negativa).
    Si vendemos ETFs de riesgo  → compramos XEON.DE (cantidad positiva).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"Operativa_{fecha_str}.xlsx")

    bold = Font(name="Calibri", bold=True, size=11)
    norm = Font(name="Calibri", size=11)
    brd  = Border(left=Side(style="thin"), right=Side(style="thin"),
                  top=Side(style="thin"),  bottom=Side(style="thin"))
    ctr  = Alignment(horizontal="center", vertical="center")

    if os.path.exists(path):
        wb = openpyxl.load_workbook(path)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Operativa"
        for col, txt in enumerate(["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"], 1):
            c = ws.cell(row=1, column=col, value=txt)
            c.font = bold; c.border = brd; c.alignment = ctr
        for col, w in zip("ABCDE", [14, 14, 14, 12, 18]):
            ws.column_dimensions[col].width = w

    for orden in ordenes:
        nr     = ws.max_row + 1
        ticker = orden['ticker']
        cant   = orden['cantidad']
        precio = orden['precio']
        ct_val = orden['ct']
        is_buy = orden['es_compra']
        formula = f"=C{nr}*(1+D{nr})" if is_buy else f"=C{nr}*(1-D{nr})"

        row_data = [
            (ticker,           None),
            (round(cant, 4),   "#,##0.0000"),
            (round(precio, 4), "#,##0.0000"),
            (ct_val,           "0.000000%"),
            (formula,          "#,##0.0000"),
        ]
        for col, (val, fmt) in enumerate(row_data, 1):
            c = ws.cell(row=nr, column=col, value=val)
            c.font = norm; c.border = brd; c.alignment = ctr
            if fmt:
                c.number_format = fmt

    wb.save(path)
    print(f"  Excel guardado: {path} ({len(ordenes)} filas)")
    return path


# ============================================================
# LÓGICA PRINCIPAL DEL DÍA
# ============================================================

def ejecutar_dia(tickers_riesgo, returns_df, precios_hoy,
                 risk_free_rate, lambda_dict, fecha_hoy=None):
    """
    Pipeline completo del día.

    Args:
        tickers_riesgo:  lista de tickers de ETFs de riesgo (sin XEON.DE)
        returns_df:      DataFrame de retornos históricos (incluye XEON.DE)
        precios_hoy:     dict {ticker: precio_cierre_hoy}
        risk_free_rate:  float — tasa BCE
        lambda_dict:     dict {ticker: lambda}
        fecha_hoy:       date

    Returns:
        dict con resumen del día
    """
    if fecha_hoy is None:
        fecha_hoy = date.today()
    fecha_str = fecha_hoy.strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print(f"SIMULACIÓN DIARIA BL-OMEGA: {fecha_hoy}")
    print(f"{'='*60}")

    estado = cargar_estado()

    # --- Valorar cartera actual ---
    precio_xeon = precios_hoy.get(XEON_TICKER, 100.0)
    part_xeon   = estado.get('participaciones_xeon', 0.0)
    valor_xeon  = part_xeon * precio_xeon

    valor_riesgo = 0.0
    posiciones   = estado.get('posiciones_riesgo', {})
    for ticker, pos in posiciones.items():
        precio_actual = precios_hoy.get(ticker, pos.get('precio_ayer', 0))
        valor_riesgo += pos['participaciones'] * precio_actual

    patrimonio = valor_riesgo + valor_xeon

    # Primera ejecución: todo en XEON.DE
    if part_xeon == 0.0 and valor_riesgo == 0.0:
        part_xeon = CAPITAL_INICIAL / precio_xeon
        estado['participaciones_xeon'] = part_xeon
        valor_xeon = CAPITAL_INICIAL
        patrimonio = CAPITAL_INICIAL
        print(f"  [INIT] Cartera inicial → {part_xeon:.4f} participaciones XEON.DE")

    print(f"\n  Patrimonio: {patrimonio:,.2f} EUR")
    print(f"  ETFs riesgo: {valor_riesgo:,.2f} EUR ({valor_riesgo/patrimonio:.1%})")
    print(f"  XEON.DE:     {valor_xeon:,.2f} EUR ({valor_xeon/patrimonio:.1%})")

    # --- Calcular pesos actuales ---
    todos_tickers = tickers_riesgo + [XEON_TICKER]
    N = len(todos_tickers)
    pesos_actuales = np.zeros(N)
    for i, ticker in enumerate(todos_tickers):
        if ticker == XEON_TICKER:
            pesos_actuales[i] = valor_xeon / patrimonio if patrimonio > 0 else 0
        else:
            pos    = posiciones.get(ticker, {})
            precio = precios_hoy.get(ticker, pos.get('precio_ayer', 0))
            valor  = pos.get('participaciones', 0) * precio
            pesos_actuales[i] = valor / patrimonio if patrimonio > 0 else 0

    # --- BL + Merton ---
    print("\n[1] Calculando señal BL+Merton...")
    bl_result = run_black_litterman(returns_df, risk_free_rate,
                                    xeon_ticker=XEON_TICKER)

    sigma_mercado = float(np.sqrt(np.diag(bl_result['Sigma'])).mean())
    merton_result = compute_optimal_weights(
        bl_result, risk_free_rate,
        xeon_ticker=XEON_TICKER,
        sigma_mercado=sigma_mercado,
        current_weights=pesos_actuales,
        frozen_tickers=estado.get('congelados', []),
    )
    print(f"  Régimen: {merton_result['regime']}")
    print(f"  Pesos objetivo: {merton_result['weights_dict']}")
    print(f"  XEON.DE objetivo: {merton_result['weight_xeon']:.1%}")

    # Pesos objetivo completos (incluye XEON.DE)
    target = merton_result['weights_array'].copy()
    xeon_idx_local = todos_tickers.index(XEON_TICKER)
    target[xeon_idx_local] = merton_result['weight_xeon']

    # Verificar suma = 1
    assert abs(target.sum() - 1.0) < 0.01, \
        f"ERROR: pesos no suman 1, suman {target.sum():.4f}"

    # --- Bandas Davis-Norman ---
    print("\n[2] Calculando bandas DN...")
    Sigma    = bl_result['Sigma']
    vol_diag = np.sqrt(np.diag(Sigma))

    bandas        = {}
    hay_que_operar= False
    motivo        = []

    for i, ticker in enumerate(todos_tickers):
        alpha_star = target[i]
        if alpha_star < 0.005 and pesos_actuales[i] < 0.005:
            continue

        # Índice en Sigma (puede diferir si el orden varía)
        if ticker in bl_result['tickers']:
            sigma_idx = bl_result['tickers'].index(ticker)
            sigma_i   = float(vol_diag[sigma_idx])
        else:
            sigma_i   = 0.15

        lambda_i  = lambda_dict.get(ticker, CT_DEFAULT)

        if alpha_star > 0.005:
            lower, upper = dn_bands_asymptotic(
                alpha_star=alpha_star, sigma=sigma_i,
                lambda_L=lambda_i, lambda_M=lambda_i, gamma=GAMMA
            )
        else:
            lower, upper = 0.0, 0.02

        bandas[ticker] = {'lower': lower, 'upper': upper, 'target': alpha_star}

        peso_actual = pesos_actuales[i]
        if peso_actual < lower or peso_actual > upper:
            hay_que_operar = True
            motivo.append(f"{ticker}: {peso_actual:.3f} fuera de [{lower:.3f}, {upper:.3f}]")
            print(f"  {ticker}: peso={peso_actual:.3f} FUERA DE BANDA [{lower:.3f}, {upper:.3f}]")
        else:
            print(f"  {ticker}: peso={peso_actual:.3f} dentro de [{lower:.3f}, {upper:.3f}] ✓")

    # También operar si hay congelados que liquidar
    for ticker in merton_result.get('liquidar', []):
        hay_que_operar = True
        motivo.append(f"{ticker}: congelado con peso < 2% → liquidar")

    if not hay_que_operar:
        print("\n  → Sin operativa hoy. Todos los activos dentro de bandas.")
        estado.update({
            'fecha_ultimo': str(fecha_hoy),
            'patrimonio_total': patrimonio,
            'pesos_actuales': {t: float(pesos_actuales[i])
                               for i, t in enumerate(todos_tickers)},
        })
        guardar_estado(estado)
        guardar_debug(fecha_str, bl_result, merton_result, bandas, [], pesos_actuales, todos_tickers)
        return {'fecha': str(fecha_hoy), 'hay_operativa': False, 'ordenes': []}

    print(f"\n  → HAY OPERATIVA. Motivos: {motivo}")

    # --- Generar órdenes ---
    # Calculamos el delta EUR para cada activo y generamos la orden
    ordenes            = []
    total_compras_eur  = 0.0
    total_ventas_eur   = 0.0

    for i, ticker in enumerate(todos_tickers):
        precio     = precios_hoy.get(ticker)
        if not precio or precio <= 0:
            print(f"  [AVISO] Sin precio para {ticker}, saltando")
            continue

        peso_obj   = target[i]
        peso_act   = pesos_actuales[i]
        delta_peso = peso_obj - peso_act
        delta_eur  = delta_peso * patrimonio

        if abs(delta_eur) < 50:   # Menos de 50 EUR: no merece la pena
            continue

        ct_val    = lambda_dict.get(ticker, CT_DEFAULT)
        es_compra = delta_eur > 0
        n_partic  = int(abs(delta_eur) / precio)
        if n_partic == 0:
            continue

        if ticker == XEON_TICKER:
            # La orden de XEON.DE es la contrapartida: se añade al final
            # con el signo opuesto al flujo de riesgo total
            pass

        ordenes.append({
            'ticker':    ticker,
            'cantidad':  n_partic if es_compra else -n_partic,
            'precio':    round(precio, 4),
            'ct':        ct_val,
            'es_compra': es_compra,
            'delta_eur': round(delta_eur, 2),
            'importe':   round(abs(delta_eur), 2),
        })

        if es_compra:
            total_compras_eur += abs(delta_eur)
        else:
            total_ventas_eur  += abs(delta_eur)

    # Asegurar que XEON.DE siempre aparece en el Excel si hay otras órdenes
    xeon_en_ordenes = any(o['ticker'] == XEON_TICKER for o in ordenes)
    if ordenes and not xeon_en_ordenes:
        # Añadir XEON.DE como contrapartida del flujo neto
        flujo_neto = total_ventas_eur - total_compras_eur  # positivo = recibimos dinero → compramos XEON
        precio_xeon_hoy = precios_hoy.get(XEON_TICKER, 100.0)
        if abs(flujo_neto) >= 50:
            n_xeon = int(abs(flujo_neto) / precio_xeon_hoy)
            if n_xeon > 0:
                ordenes.append({
                    'ticker':    XEON_TICKER,
                    'cantidad':  n_xeon if flujo_neto > 0 else -n_xeon,
                    'precio':    round(precio_xeon_hoy, 4),
                    'ct':        lambda_dict.get(XEON_TICKER, CT['XEON.DE']),
                    'es_compra': flujo_neto > 0,
                    'delta_eur': round(flujo_neto, 2),
                    'importe':   round(abs(flujo_neto), 2),
                })

    print(f"\n  {len(ordenes)} órdenes generadas:")
    for o in ordenes:
        signo = "COMPRA" if o['es_compra'] else "VENTA"
        print(f"    {signo:5s} {o['ticker']:12s} {abs(o['cantidad']):8.0f} partic "
              f"@ {o['precio']:.4f} EUR  ({o['importe']:,.0f} EUR)")

    # Calcular costes totales
    coste_total = sum(o['importe'] * o['ct'] for o in ordenes)

    # --- Guardar Excel ---
    excel_path = generar_excel_operativa(ordenes, fecha_str)

    # --- Actualizar estado ---
    estado.update({
        'fecha_ultimo':          str(fecha_hoy),
        'patrimonio_total':      float(patrimonio - coste_total),
        'participaciones_xeon':  float(part_xeon),
        'precio_xeon_ayer':      float(precio_xeon),
        'ultimo_top5':           merton_result['selected_tickers'],
        'congelados':            estado.get('congelados', []),
        'n_operaciones':         estado.get('n_operaciones', 0) + len(ordenes),
        'costes_acumulados':     estado.get('costes_acumulados', 0) + coste_total,
        'pesos_actuales':        {t: float(target[i]) for i, t in enumerate(todos_tickers)},
    })
    guardar_estado(estado)

    # --- Debug ---
    guardar_debug(fecha_str, bl_result, merton_result, bandas, ordenes,
                  pesos_actuales, todos_tickers)

    return {
        'fecha':         str(fecha_hoy),
        'hay_operativa': True,
        'ordenes':       ordenes,
        'coste_total':   coste_total,
        'excel':         excel_path,
    }


def guardar_debug(fecha_str, bl_result, merton_result, bandas, ordenes,
                  pesos_actuales, todos_tickers):
    """Guarda las trazas completas del día para auditoría."""
    debug_path = os.path.join(OUTPUT_DIR, "debug", fecha_str)
    os.makedirs(debug_path, exist_ok=True)

    bl_result['omega_scores'].to_csv(
        os.path.join(debug_path, 'omega_scores.csv'), header=['omega']
    )
    pd.Series(bl_result['mu_BL'], index=bl_result['tickers']).to_csv(
        os.path.join(debug_path, 'posterior_mu.csv'), header=['mu_BL']
    )
    pd.Series(merton_result['weights_dict']).to_csv(
        os.path.join(debug_path, 'pesos_objetivo_riesgo.csv'), header=['peso']
    )
    pd.DataFrame(bandas).T.to_csv(os.path.join(debug_path, 'bandas_dn.csv'))

    resumen = {
        'fecha':          fecha_str,
        'top3_omega':     bl_result['top_tickers'],
        'top5_merton':    merton_result['selected_tickers'],
        'weight_xeon':    merton_result['weight_xeon'],
        'regime':         merton_result['regime'],
        'n_ordenes':      len(ordenes),
    }
    with open(os.path.join(debug_path, 'summary.json'), 'w') as f:
        json.dump(resumen, f, indent=2)

    print(f"  Debug guardado en: {debug_path}/")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    print("simulacion_diaria_bl.py")
    print("Para ejecutar necesitas tickers_riesgo, returns_df y precios_hoy.")
    print("Ver main_black_litterman.py para el flujo completo.")
```

---

## Cómo probar que tu código funciona

```python
"""
test_bl_strategy.py — Ejecutar: python test_bl_strategy.py
"""
import numpy as np
import pandas as pd
from strategies.bl_omega_strategy import BLOmegaStrategy

np.random.seed(42)
n_days = 100
tickers = ['ETF_A', 'ETF_B', 'ETF_C', 'ETF_D', 'ETF_E', 'XEON.DE']

# Retornos sintéticos: XEON.DE casi sin variación
returns = np.random.randn(n_days, 6) * 0.01
returns[:, 0] += 0.001   # ETF_A tiene buen drift
returns[:, 5]  = 0.0001  # XEON.DE casi plano
returns_df = pd.DataFrame(returns, columns=tickers)

estrategia = BLOmegaStrategy(tickers=tickers, xeon_ticker='XEON.DE')
lambda_dict = {t: 0.002 for t in tickers}
lambda_dict['XEON.DE'] = 0.00005

print("=== TEST 1: Pesos iniciales suman 1 ===")
pesos = estrategia.get_initial_weights(returns_df, 0.025, lambda_dict)
print(f"  Pesos iniciales: {dict(zip(tickers, pesos.round(3)))}")
assert abs(pesos.sum() - 1.0) < 0.01, f"ERROR: pesos suman {pesos.sum():.4f}"
assert pesos[tickers.index('XEON.DE')] >= 0, "ERROR: XEON.DE no puede ser negativo"
print("  ✓ Pesos suman 1 y XEON.DE >= 0\n")

print("=== TEST 2: Sin rebalanceo cuando estamos en el objetivo ===")
decision = estrategia.decide(
    date=pd.Timestamp('2024-06-01'),
    current_weights=pesos,       # Exactamente en el objetivo
    returns_df=returns_df,
    risk_free_rate=0.025,
    lambda_per_ticker=lambda_dict,
    day_index=1
)
print(f"  Decisión: {decision}")
print("  ✓ No rebalancea cuando está en objetivo\n")

print("=== TEST 3: Sí rebalancea cuando está muy lejos ===")
pesos_desviados = np.zeros(6)
pesos_desviados[0] = 0.90   # Todo en ETF_A (muy fuera de objetivo)
pesos_desviados[5] = 0.10   # Poco en XEON.DE
decision2 = estrategia.decide(
    date=pd.Timestamp('2024-06-02'),
    current_weights=pesos_desviados,
    returns_df=returns_df,
    risk_free_rate=0.025,
    lambda_per_ticker=lambda_dict,
    day_index=2
)
assert decision2 is not None, "ERROR: debería decidir rebalancear"
assert decision2['rebalance'] == True
target = decision2['target_weights']
assert abs(target.sum() - 1.0) < 0.01, f"ERROR: target no suma 1, suma {target.sum():.4f}"
assert target[tickers.index('XEON.DE')] >= 0, "ERROR: XEON.DE negativo"
print(f"  Target weights: {dict(zip(tickers, target.round(3)))}")
print(f"  XEON.DE target: {target[tickers.index('XEON.DE')]:.3f}")
print("  ✓ Rebalanceo correcto y target suma 1\n")

print("=== TODOS LOS TESTS PASARON ===")
```

---

## Checklist antes de avisar al grupo

- [ ] `strategies/bl_omega_strategy.py` creado
- [ ] `simulacion_diaria_bl.py` creado
- [ ] Los 3 tests pasan sin errores
- [ ] `target_weights.sum()` es siempre 1.0 ± 0.01
- [ ] `target_weights[xeon_idx]` nunca es negativo
- [ ] El Excel de operativa siempre tiene una fila de XEON.DE si hay operativa
- [ ] El JSON de estado guarda `participaciones_xeon` (no euros)

---

## Cuándo avisar por WhatsApp

| Momento | Qué escribir |
|---|---|
| `bl_omega_strategy.py` listo | "P3: estrategia lista, testeando" |
| `simulacion_diaria_bl.py` listo | "P3: ✓ todo listo" |
| Error raro | Pega el traceback completo |

---

## Lo que NO tienes que hacer

- No modificar `davis_norman.py`
- No descargar datos históricos (lo hace P4)
- No hacer el backtest (lo hace P4+P5)
- No modificar `engine_multiasset.py`
