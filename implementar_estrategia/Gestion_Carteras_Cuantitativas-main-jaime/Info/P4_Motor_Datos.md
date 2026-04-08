# P4 — Tu tarea: Motor de Datos + Engine Multiactivo
### VERSIÓN DEFINITIVA
### Entrega: **LUNES a mediodía** — eres el desbloqueante de todos

---

## ⚠️ TU TRABAJO ES EL MÁS URGENTE

Sin datos limpios el lunes, nadie puede trabajar.
Tu prioridad número 1 es tener el universo descargado y limpio cuanto antes.

---

## Cambios respecto a la versión anterior — léelos primero

1. **XEON.DE se descarga por separado** con su histórico real, igual que los otros ETFs.
2. **Look-ahead bias corregido.** En el loop diario, la estrategia solo recibe datos hasta la fecha actual. Esto es lo más importante técnicamente de todo tu trabajo.
3. **Calendario con forward-fill** (máximo 1 día) en lugar de intersección estricta. Evita perder cientos de días por festivos locales.
4. **`PortfolioMultiAsset` guarda participaciones reales de XEON.DE**, no euros abstractos.
5. **El historial exporta el peso de XEON.DE** como una columna más, para que P5 lo grafique.

---

## Qué es el look-ahead bias y por qué importa tanto

Es el error más grave en backtesting. Ocurre cuando usas datos del futuro para
tomar decisiones del pasado sin darte cuenta.

Ejemplo concreto del error que hay que evitar:

```python
# MAL — pasa TODOS los retornos a la estrategia
for date in trading_dates:
    decision = estrategia.decide(returns_df=returns_df_completo, ...)
    #                                       ^^^^^^^^^^^^^^^^^^^
    #                             Contiene retornos de fechas FUTURAS
    #                             La estrategia puede "ver el futuro" sin saberlo
```

```python
# BIEN — solo pasa datos hasta hoy
for date in trading_dates:
    returns_hasta_hoy = returns_df.loc[:date]   # Solo hasta la fecha actual
    decision = estrategia.decide(returns_df=returns_hasta_hoy, ...)
```

La diferencia en las métricas puede ser enorme: un Sharpe de 2.5 con look-ahead
puede caer a 0.8 sin él. Si no lo corriges, el backtest no es válido.

---

## Estructura de archivos que creas

```
gestion_cuantitativa/
├── config.py                ← MODIFICAR (añadir universo + XEON.DE)
├── data/
│   ├── universe_loader.py   ← CREAR TÚ (descarga todos los ETFs + XEON.DE)
│   └── cache/
│       └── universe_prices.csv  ← Se genera automáticamente
├── engine_multiasset.py     ← CREAR TÚ (motor de backtest)
└── main_black_litterman.py  ← CREAR TÚ (script principal)
```

---

## Archivo 1: Cambios en `config.py`

Añade esto al final del `config.py` existente (sin borrar nada):

```python
# ============================================================
# CONFIGURACIÓN NUEVA ESTRATEGIA BL-OMEGA
# ============================================================

XEON_TICKER = "XEON.DE"   # ETF monetario €STR — activo libre de riesgo invertible

# Universo de ETFs de riesgo {ticker: categoria}
# La categoría evita meter dos ETFs del mismo tema en cartera
ETF_UNIVERSE = {
    "IWDA.L":  "global_equity",
    "IEMA.L":  "emerging_markets",
    "IUSN.DE": "small_cap",
    "XLK":     "technology",
    "XLF":     "financials",
    "XLV":     "healthcare",
    "XLE":     "energy",
    "XLP":     "consumer_staples",
    "XLI":     "industrials",
    "XLU":     "utilities",
    "SOXX":    "semiconductors",
    "CIBR":    "cybersecurity",
    "BOTZ":    "robotics_ai",
    "LIT":     "clean_energy",
    "ITB":     "homebuilders",
    "GLD":     "gold",
    "SLV":     "silver",
    "DBA":     "agribusiness",
    "TLT":     "us_bonds_long",
    "IEF":     "us_bonds_medium",
}

BL_OMEGA_WINDOW_FAST = 42
BL_OMEGA_WINDOW_SLOW = 84
BL_N_TOP_VIEWS       = 3
BL_N_TOP_PORTFOLIO   = 5
BL_MAX_WEIGHT        = 0.40
BL_MAX_SECTOR_WEIGHT = 0.25
BL_RECALIB_FREQ      = 5
BL_VOL_CAUTION_THR   = 0.20
BL_VOL_CRISIS_THR    = 0.30
```

---

## Archivo 2: `data/universe_loader.py`

```python
"""
Descarga y limpieza del universo de ETFs.
Incluye XEON.DE como activo monetario (libre de riesgo invertible).

Estrategia de calendario — Opción C (fecha de inicio adaptada):
  No todos los ETFs tienen historia desde 2008. En lugar de eliminar los
  ETFs más recientes del universo (Opción A) o gestionar un universo que
  cambia de tamaño cada día (Opción B, muy compleja), detectamos
  automáticamente la primera fecha en que hay suficientes ETFs disponibles
  y empezamos el backtest desde ahí.

  Resultado típico: universo completo de 20 ETFs disponible desde ~2015-2016.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

MIN_ETFS_DISPONIBLES = 15   # Mínimo de ETFs de riesgo para empezar el backtest


def download_universe(tickers_riesgo, xeon_ticker, start, end, min_days=252):
    """
    Descarga precios de cierre de todos los ETFs de riesgo + XEON.DE.

    Solo elimina tickers que tengan menos de min_days en todo el periodo.
    Los que tienen historia parcial (ej. BOTZ desde 2016) se mantienen
    y la función detect_backtest_start se encarga de encontrar cuándo
    hay suficientes disponibles.

    Args:
        tickers_riesgo: lista de tickers de ETFs de riesgo
        xeon_ticker:    ticker del ETF monetario (XEON.DE)
        start:          fecha inicio str — descargar desde aquí para warm-up
        end:            fecha fin str
        min_days:       mínimo de días históricos para no eliminar el ticker

    Returns:
        pd.DataFrame (días × tickers) con precios de cierre.
        Puede haber NaN donde un ETF todavía no existía.
        XEON.DE es la última columna.
    """
    todos = tickers_riesgo + [xeon_ticker]
    print(f"Descargando {len(todos)} tickers ({len(tickers_riesgo)} riesgo + XEON.DE)...")

    data = yf.download(todos, start=start, end=end, progress=True, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = todos

    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Eliminar solo los tickers con historia muy corta (casi sin datos)
    tickers_ok = []
    for ticker in prices.columns:
        n = prices[ticker].dropna().count()
        if n < min_days:
            print(f"  ⚠ {ticker}: solo {n} días → ELIMINADO (mínimo {min_days})")
        else:
            tickers_ok.append(ticker)
            print(f"  ✓ {ticker}: {n} días "
                  f"(desde {prices[ticker].dropna().index[0].date()})")

    if xeon_ticker not in tickers_ok:
        print(f"\n  ⚠⚠ XEON.DE no tiene historia suficiente.")
        print(f"     Prueba con CSH2.PA o ERNE.L como fallback en config.py")

    prices = prices[tickers_ok]
    print(f"\n  → {len(tickers_ok)} tickers aceptados")
    return prices


def detect_backtest_start(prices_raw, xeon_ticker, min_etfs=MIN_ETFS_DISPONIBLES):
    """
    Detecta la primera fecha en que hay suficientes ETFs disponibles.

    Recorre el DataFrame fila a fila (de más antiguo a más reciente) y
    encuentra el primer día en que al menos min_etfs ETFs de RIESGO
    tienen precio (no NaN).

    Por qué esto funciona:
      - IWDA.L, GLD, TLT tienen historia desde 2008-2009 → disponibles desde el inicio
      - XLK, SOXX tienen historia desde ~2000 → disponibles desde el inicio
      - BOTZ, CIBR tienen historia desde 2015-2016 → no disponibles antes
      - Cuando hay 15+ ETFs con precio = hay suficiente diversificación
        para que BL y Merton funcionen bien

    Args:
        prices_raw:  DataFrame con NaN donde el ETF no existía todavía
        xeon_ticker: ticker del ETF monetario (no cuenta para el mínimo)
        min_etfs:    mínimo de ETFs de RIESGO que deben estar disponibles

    Returns:
        pd.Timestamp — primera fecha válida para empezar el backtest
    """
    risk_cols = [c for c in prices_raw.columns if c != xeon_ticker]

    for date, row in prices_raw[risk_cols].iterrows():
        n_disponibles = row.notna().sum()
        if n_disponibles >= min_etfs:
            print(f"  Fecha de inicio detectada: {date.date()} "
                  f"({n_disponibles} ETFs de riesgo disponibles)")
            return date

    raise ValueError(
        f"Nunca hay {min_etfs} ETFs disponibles simultáneamente. "
        f"Reduce MIN_ETFS_DISPONIBLES en universe_loader.py."
    )


def build_calendar(prices_raw, backtest_start, max_ffill_days=1):
    """
    Construye el calendario limpio desde la fecha de inicio detectada.

    Dos pasos:
      1. Recortar desde backtest_start (los ETFs recientes ya tienen datos)
      2. Forward-fill de máximo 1 día (festivos locales)
      3. Eliminar filas con NaN restantes (gaps de 2+ días)

    Args:
        prices_raw:      DataFrame completo con posibles NaN
        backtest_start:  primera fecha válida (de detect_backtest_start)
        max_ffill_days:  máximo días a rellenar hacia adelante

    Returns:
        pd.DataFrame limpio sin NaN, empezando desde backtest_start
    """
    # Desde la fecha de inicio todos los ETFs deberían tener datos
    prices = prices_raw.loc[backtest_start:].copy()
    prices = prices.ffill(limit=max_ffill_days)

    n_antes   = len(prices)
    prices    = prices.dropna()
    n_despues = len(prices)

    print(f"  Calendario final: {n_despues} días "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    if n_antes - n_despues > 0:
        print(f"  ({n_antes - n_despues} días eliminados por gaps residuales)")
    return prices


def compute_returns(prices):
    """Retornos simples diarios a partir de precios."""
    return prices.pct_change().dropna()


def load_universe_data(tickers_riesgo=None, xeon_ticker='XEON.DE',
                       start='2008-01-01', end='2024-12-31',
                       min_etfs=MIN_ETFS_DISPONIBLES,
                       cache_file=None):
    """
    Pipeline completo: descarga → detectar inicio → calendario → retornos.

    Si existe cache_file, carga desde disco (evita descargar cada vez).

    Flujo interno:
      1. Descargar todos los ETFs desde start (necesitamos history larga
         para el warm-up de los estimadores aunque el backtest empiece más tarde)
      2. Detectar la primera fecha con min_etfs ETFs disponibles
      3. Recortar y limpiar el calendario desde esa fecha
      4. Calcular retornos

    Returns:
        dict con:
            'prices':          DataFrame — precios limpios (sin NaN)
            'returns':         DataFrame — retornos diarios
            'tickers_riesgo':  list — ETFs de riesgo disponibles (sin XEON.DE)
            'todos_tickers':   list — todos incluyendo XEON.DE
            'backtest_start':  pd.Timestamp — primera fecha válida del backtest
    """
    from config import ETF_UNIVERSE, XEON_TICKER

    if tickers_riesgo is None:
        tickers_riesgo = list(ETF_UNIVERSE.keys())

    if cache_file and os.path.exists(cache_file):
        print(f"Cargando desde caché: {cache_file}")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        backtest_start = prices.index[0]
        print(f"  → {len(prices)} días desde {backtest_start.date()}, "
              f"{len(prices.columns)} tickers")
    else:
        # Descargar desde 2008 aunque el backtest empiece en 2015
        # Los datos anteriores sirven de warm-up para los estimadores
        prices_raw     = download_universe(tickers_riesgo, xeon_ticker, start, end)
        backtest_start = detect_backtest_start(prices_raw, xeon_ticker, min_etfs)
        prices         = build_calendar(prices_raw, backtest_start)

        if cache_file:
            os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
            prices.to_csv(cache_file)
            print(f"  Caché guardado: {cache_file}")

    returns = compute_returns(prices)

    t_riesgo = [t for t in prices.columns if t != xeon_ticker]
    t_todos  = list(prices.columns)

    return {
        'prices':          prices,
        'returns':         returns,
        'tickers_riesgo':  t_riesgo,
        'todos_tickers':   t_todos,
        'backtest_start':  prices.index[0],
    }


def build_lambda_dict(todos_tickers, default_lambda=0.002, xeon_lambda=0.00005):
    """
    Construye el diccionario de costes de transacción.
    XEON.DE tiene un lambda especial (spread muy pequeño).
    """
    from config import XEON_TICKER
    lambdas = {}
    for ticker in todos_tickers:
        if ticker == XEON_TICKER:
            lambdas[ticker] = xeon_lambda
        else:
            try:
                from data.transaction_costs import load_lambda_from_bidask
                from config import BIDASK_DIR
                lam = load_lambda_from_bidask(BIDASK_DIR, ticker)
                lambdas[ticker] = lam if lam and lam > 0 else default_lambda
            except Exception:
                lambdas[ticker] = default_lambda
    return lambdas
```

---

## Archivo 3: `engine_multiasset.py`

```python
"""
Motor de backtesting multiactivo.

Diferencias clave respecto al engine.py original:
  1. Gestiona múltiples ETFs + XEON.DE simultáneamente
  2. XEON.DE tiene participaciones reales (no euros abstractos)
  3. Look-ahead bias corregido: siempre pasa returns_df.loc[:date] a la estrategia
  4. El historial exporta weight_XEON como columna propia
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
INITIAL_WEALTH = 10_000_000


class PortfolioMultiAsset:
    """
    Portfolio con múltiples ETFs incluyendo XEON.DE.
    Mantiene participaciones reales de cada activo.
    """

    def __init__(self, todos_tickers, xeon_ticker, initial_wealth=INITIAL_WEALTH):
        self.todos_tickers = todos_tickers
        self.xeon_ticker   = xeon_ticker
        self.N             = len(todos_tickers)
        self.xeon_idx      = todos_tickers.index(xeon_ticker)
        self.wealth        = float(initial_wealth)
        self.initial_wealth= float(initial_wealth)

        # Empezamos 100% en XEON.DE (siempre invertidos desde el primer día)
        # Las participaciones reales se calculan cuando tenemos precio
        self.weights       = np.zeros(self.N)
        self.weights[self.xeon_idx] = 1.0

        # Historial
        self.history    = []
        self.trades     = []
        self.total_costs= 0.0

    def initialize_with_prices(self, prices_today):
        """
        Inicializa las participaciones con los precios del primer día.
        Todo empieza en XEON.DE.

        Args:
            prices_today: np.array (N,) con precios de cierre del día 0
        """
        # Al inicio todo en XEON.DE
        precio_xeon = prices_today[self.xeon_idx]
        if precio_xeon > 0:
            self.weights = np.zeros(self.N)
            self.weights[self.xeon_idx] = 1.0

    def update_prices(self, returns_today, rf_daily):
        """
        Actualiza el valor de la cartera con los retornos del día.

        XEON.DE se actualiza con su retorno de mercado real (no con rf_daily).
        rf_daily solo se usaría si tuviéramos cash real, que no tenemos.

        Args:
            returns_today: np.array (N,) — retornos diarios de hoy de todos los activos
            rf_daily:      float — tasa libre de riesgo diaria (solo para compatibilidad)
        """
        if isinstance(returns_today, pd.Series):
            returns_today = returns_today.reindex(self.todos_tickers).fillna(0).values

        # Valor por activo antes de hoy
        valor_por_activo = self.wealth * self.weights

        # Actualizar con retornos de hoy (XEON.DE usa su retorno real de mercado)
        valor_por_activo *= (1 + returns_today)

        self.wealth = valor_por_activo.sum()
        if self.wealth > 1e-6:
            self.weights = valor_por_activo / self.wealth
        else:
            self.weights = np.zeros(self.N)

    def execute_rebalance(self, target_weights, lambda_per_ticker, date=None):
        """
        Rebalancea la cartera hacia los pesos objetivo.

        SIEMPRE genera al menos un trade de XEON.DE junto con los de riesgo.
        Los costes se deducen del patrimonio.

        Args:
            target_weights:    np.array (N,) — pesos objetivo, suma = 1.0
            lambda_per_ticker: dict {ticker: lambda}
            date:              fecha del trade
        """
        target_weights = np.array(target_weights)

        # Verificar que suman 1
        if abs(target_weights.sum() - 1.0) > 0.01:
            print(f"  [AVISO motor] target_weights suman {target_weights.sum():.4f}, normalizando")
            target_weights = target_weights / target_weights.sum()

        coste_total = 0.0
        trades_hoy  = []

        for i, ticker in enumerate(self.todos_tickers):
            delta_weight = target_weights[i] - self.weights[i]
            delta_eur    = delta_weight * self.wealth

            if abs(delta_eur) < 10:
                continue

            lam  = lambda_per_ticker.get(ticker, 0.002)
            cost = lam * abs(delta_eur)
            coste_total += cost

            trades_hoy.append({
                'date':         date,
                'ticker':       ticker,
                'type':         'buy' if delta_eur > 0 else 'sell',
                'delta_eur':    delta_eur,
                'cost':         cost,
                'weight_before':self.weights[i],
                'weight_after': target_weights[i],
            })

        # Aplicar costes
        self.wealth      -= coste_total
        self.total_costs += coste_total

        # Actualizar pesos al objetivo
        if self.wealth > 1e-6:
            self.weights = target_weights * (1 - coste_total / (self.wealth + coste_total))
            s = self.weights.sum()
            if s > 1e-6:
                self.weights = self.weights / s

        self.trades.extend(trades_hoy)
        return {'cost': coste_total, 'n_trades': len(trades_hoy)}

    def record_state(self, date):
        """Registra el estado en el historial."""
        state = {
            'date':        date,
            'wealth':      self.wealth,
            'total_costs': self.total_costs,
        }
        for i, ticker in enumerate(self.todos_tickers):
            state[f'weight_{ticker}'] = self.weights[i]
        self.history.append(state)

    def get_history_df(self):
        return pd.DataFrame(self.history).set_index('date')

    def get_trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


class MultiAssetBacktestEngine:
    """
    Motor de backtest para la estrategia BL-Omega.
    Garantiza ausencia de look-ahead bias.
    """

    def __init__(self, prices_df, returns_df, risk_free_series,
                 xeon_ticker, lambda_per_ticker=None,
                 start_date=None, end_date=None,
                 initial_wealth=None, warmup_days=252):
        """
        Args:
            prices_df:         DataFrame (días × tickers) — precios cierre
            returns_df:        DataFrame (días × tickers) — retornos diarios
            risk_free_series:  pd.Series — tasa BCE anualizada
            xeon_ticker:       str — ticker del ETF monetario
            lambda_per_ticker: dict {ticker: lambda}
            start_date:        inicio del backtest
            end_date:          fin del backtest
            initial_wealth:    patrimonio inicial
            warmup_days:       días antes de start_date para warm-up de estimadores
        """
        self.prices_df        = prices_df
        self.returns_df       = returns_df
        self.risk_free_series = risk_free_series
        self.xeon_ticker      = xeon_ticker
        self.todos_tickers    = list(prices_df.columns)
        self.N                = len(self.todos_tickers)
        self.lambda_per_ticker= lambda_per_ticker or {t: 0.002 for t in self.todos_tickers}
        self.initial_wealth   = initial_wealth or INITIAL_WEALTH
        self.warmup_days      = warmup_days

        all_dates         = returns_df.index
        self.start_date   = pd.Timestamp(start_date) if start_date else all_dates[warmup_days]
        self.end_date     = pd.Timestamp(end_date)   if end_date   else all_dates[-1]

        print(f"Motor multiactivo:")
        print(f"  Tickers: {self.todos_tickers}")
        print(f"  XEON.DE: {xeon_ticker}")
        print(f"  Backtest: {self.start_date.date()} → {self.end_date.date()}")
        print(f"  Warm-up: {warmup_days} días")

    def run(self, strategy):
        """
        Ejecuta el backtest día a día.

        PUNTO CRÍTICO ANTI LOOK-AHEAD BIAS:
        En cada iteración pasamos returns_df.loc[:date] a la estrategia.
        Nunca el DataFrame completo.

        Args:
            strategy: instancia de BLOmegaStrategy (de P3)

        Returns:
            dict con history, trades, portfolio, tickers
        """
        trading_dates = self.returns_df.index[
            (self.returns_df.index >= self.start_date) &
            (self.returns_df.index <= self.end_date)
        ]

        if len(trading_dates) == 0:
            raise ValueError("No hay fechas en el rango del backtest.")

        print(f"\nEjecutando backtest: {len(trading_dates)} días...")

        # Tasa libre de riesgo alineada
        rf = self.risk_free_series.reindex(
            self.returns_df.index, method='ffill'
        ).fillna(0.02)

        # Inicializar portfolio (100% en XEON.DE al inicio)
        portfolio = PortfolioMultiAsset(
            todos_tickers=self.todos_tickers,
            xeon_ticker=self.xeon_ticker,
            initial_wealth=self.initial_wealth
        )

        # Pesos iniciales usando warm-up
        first_date    = trading_dates[0]
        # ANTI LOOK-AHEAD: solo datos hasta first_date
        returns_warmup= self.returns_df.loc[:first_date]
        rf_init       = float(rf.loc[:first_date].iloc[-1])

        print(f"  Inicializando con {len(returns_warmup)} días de warm-up...")
        initial_weights = strategy.get_initial_weights(
            returns_warmup, rf_init, self.lambda_per_ticker
        )
        portfolio.execute_rebalance(initial_weights, self.lambda_per_ticker, date=first_date)

        # Bucle diario
        for i, date in enumerate(trading_dates):
            if i % 100 == 0:
                xeon_w = portfolio.weights[portfolio.xeon_idx]
                print(f"  Día {i:4d}/{len(trading_dates)} | {date.date()} | "
                      f"Wealth: {portfolio.wealth:>12,.0f} EUR | "
                      f"XEON.DE: {xeon_w:.1%}")

            # 1) Actualizar precios con retornos de hoy
            returns_today = self.returns_df.loc[date]
            rf_daily      = float(rf.loc[date]) / TRADING_DAYS_PER_YEAR
            portfolio.update_prices(returns_today, rf_daily)

            # 2) *** ANTI LOOK-AHEAD BIAS ***
            # Solo pasamos datos hasta la fecha actual (inclusive)
            returns_hasta_hoy = self.returns_df.loc[:date]
            rf_hoy            = float(rf.loc[date])

            # 3) Decisión de la estrategia
            decision = strategy.decide(
                date              = date,
                current_weights   = portfolio.weights.copy(),
                returns_df        = returns_hasta_hoy,   # ← solo hasta hoy
                risk_free_rate    = rf_hoy,
                lambda_per_ticker = self.lambda_per_ticker,
                day_index         = i,
            )

            # 4) Ejecutar rebalanceo si procede
            if decision is not None and decision.get('rebalance', False):
                portfolio.execute_rebalance(
                    decision['target_weights'],
                    self.lambda_per_ticker,
                    date=date
                )

            # 5) Registrar estado
            portfolio.record_state(date)

        print(f"\nBacktest completado.")
        print(f"  Wealth final:    {portfolio.wealth:>12,.0f} EUR")
        print(f"  Costes totales:  {portfolio.total_costs:>10,.0f} EUR")
        print(f"  Nº rebalanceos:  {len(portfolio.trades)}")

        return {
            'history':      portfolio.get_history_df(),
            'trades':       portfolio.get_trades_df(),
            'portfolio':    portfolio,
            'strategy_name':strategy.name,
            'tickers':      self.todos_tickers,
            'xeon_ticker':  self.xeon_ticker,
            'start_date':   self.start_date,
            'end_date':     self.end_date,
            'initial_wealth':self.initial_wealth,
        }
```

---

## Archivo 4: `main_black_litterman.py`

```python
"""
Main: Backtest completo de la estrategia BL-Omega.

Dos modos de uso:

  1) Backtest simple (ejecutar directamente):
       python main_black_litterman.py

  2) Llamado desde walk_forward_bl.py (P5) con parámetros concretos:
       from main_black_litterman import run_backtest
       result = run_backtest(
           start='2015-01-01', end='2017-12-31',
           params={'omega_window_fast': 42, 'gamma': -2, 'recalib_freq': 5}
       )
"""

import os
import numpy as np
import pandas as pd

from data.universe_loader import load_universe_data, build_lambda_dict
from engine_multiasset import MultiAssetBacktestEngine
from strategies.bl_omega_strategy import BLOmegaStrategy
from config import (
    ETF_UNIVERSE, XEON_TICKER, INITIAL_WEALTH, GAMMA,
    BL_RECALIB_FREQ, BL_OMEGA_WINDOW_FAST,
    DATA_START, BACKTEST_END, RISK_FREE_FRED,
)

OUTPUT_DIR = "outputs/bl_omega"

# Datos cargados una sola vez y reutilizados por todas las llamadas
# (evita descargar de internet en cada ventana del walk-forward)
_DATA_CACHE = None
_RF_CACHE   = None


def _load_shared_data():
    """
    Carga datos y tasa libre de riesgo una sola vez.
    Las llamadas sucesivas de run_backtest() reutilizan estos datos.
    """
    global _DATA_CACHE, _RF_CACHE

    if _DATA_CACHE is None:
        print("Cargando datos del universo (solo se hace una vez)...")
        _DATA_CACHE = load_universe_data(
            tickers_riesgo=list(ETF_UNIVERSE.keys()),
            xeon_ticker=XEON_TICKER,
            start=DATA_START,
            end=BACKTEST_END,
            cache_file='data/cache/universe_prices.csv'
        )

    if _RF_CACHE is None:
        try:
            import pandas_datareader as pdr
            rf = pdr.get_data_fred(RISK_FREE_FRED, start=DATA_START, end=BACKTEST_END)
            rf = rf.iloc[:, 0] / 100.0
            _RF_CACHE = rf.reindex(
                _DATA_CACHE['returns'].index, method='ffill'
            ).fillna(0.02)
        except Exception as e:
            print(f"  FRED no disponible ({e}) → usando 2.5% fijo")
            _RF_CACHE = pd.Series(0.025, index=_DATA_CACHE['returns'].index)

    return _DATA_CACHE, _RF_CACHE


def run_backtest(start=None, end=None, params=None, save_results=True,
                 output_dir=OUTPUT_DIR):
    """
    Ejecuta un backtest de la estrategia BL-Omega en el periodo [start, end].

    Esta función es la que llama P5 desde walk_forward_bl.py en bucle
    para cada ventana IS y OOS. Los datos se cargan una sola vez gracias
    al caché interno (_DATA_CACHE).

    Args:
        start:        str o pd.Timestamp — inicio del backtest
                      Si None, usa la fecha detectada automáticamente
        end:          str o pd.Timestamp — fin del backtest
                      Si None, usa BACKTEST_END de config.py
        params:       dict con parámetros a usar. Keys válidas:
                        'omega_window_fast': int (default: BL_OMEGA_WINDOW_FAST)
                        'gamma':             float (default: GAMMA)
                        'recalib_freq':      int (default: BL_RECALIB_FREQ)
                      Cualquier key no reconocida se ignora silenciosamente.
        save_results: bool — si True guarda wealth_history.csv y trades.csv
        output_dir:   str — directorio donde guardar (solo si save_results=True)

    Returns:
        dict con:
            'history':      pd.DataFrame — patrimonio diario + pesos
            'trades':       pd.DataFrame — registro de todas las órdenes
            'sharpe':       float — Sharpe ratio del periodo
            'cagr':         float — CAGR del periodo
            'max_dd':       float — máximo drawdown
            'avg_rf':       float — tasa libre de riesgo media
            'start':        pd.Timestamp
            'end':          pd.Timestamp
            'params':       dict — parámetros usados
    """
    params = params or {}
    omega_window_fast = params.get('omega_window_fast', BL_OMEGA_WINDOW_FAST)
    gamma             = params.get('gamma',             GAMMA)
    recalib_freq      = params.get('recalib_freq',      BL_RECALIB_FREQ)

    # Cargar datos (del caché si ya se cargaron antes)
    data, rf = _load_shared_data()

    prices_df      = data['prices']
    returns_df     = data['returns']
    todos_tickers  = data['todos_tickers']

    # Fechas del backtest
    start = pd.Timestamp(start) if start else data['backtest_start']
    end   = pd.Timestamp(end)   if end   else returns_df.index[-1]

    avg_rf = float(rf.loc[start:end].mean())

    # Costes de transacción
    lambda_dict = build_lambda_dict(todos_tickers)

    # Estrategia con los parámetros de esta ventana
    # NOTA: omega_window_fast se pasa a BLOmegaStrategy que lo reenvía a BL
    estrategia = BLOmegaStrategy(
        tickers=todos_tickers,
        xeon_ticker=XEON_TICKER,
        recalib_freq=recalib_freq,
        gamma=gamma,
        categoria_por_ticker=ETF_UNIVERSE,
        omega_window_fast=omega_window_fast,   # parámetro nuevo que P3 añade al __init__
    )

    motor = MultiAssetBacktestEngine(
        prices_df=prices_df,
        returns_df=returns_df,
        risk_free_series=rf,
        xeon_ticker=XEON_TICKER,
        lambda_per_ticker=lambda_dict,
        start_date=start,
        end_date=end,
        initial_wealth=INITIAL_WEALTH,
        warmup_days=252,
    )

    result = motor.run(estrategia)
    history = result['history']
    wealth  = history['wealth']

    # Métricas básicas
    n_years = len(wealth) / 252
    cagr    = (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol     = wealth.pct_change().dropna().std() * np.sqrt(252)
    sharpe  = (cagr - avg_rf) / vol if vol > 0 else 0
    max_dd  = ((wealth - wealth.cummax()) / wealth.cummax()).min()

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        history.to_csv(os.path.join(output_dir, 'wealth_history.csv'))
        if len(result['trades']) > 0:
            result['trades'].to_csv(
                os.path.join(output_dir, 'trades.csv'), index=False
            )

    return {
        'history': history,
        'trades':  result['trades'],
        'sharpe':  round(sharpe, 4),
        'cagr':    round(cagr, 4),
        'max_dd':  round(max_dd, 4),
        'avg_rf':  avg_rf,
        'start':   start,
        'end':     end,
        'params':  {'omega_window_fast': omega_window_fast,
                    'gamma': gamma, 'recalib_freq': recalib_freq},
    }


def main():
    """Backtest simple con parámetros por defecto. Ejecutar directamente."""
    print("=" * 60)
    print("BACKTEST SIMPLE: BL-Omega + XEON.DE")
    print("=" * 60)

    result = run_backtest(save_results=True)

    print(f"\n{'='*60}")
    print(f"RESULTADOS")
    print(f"{'='*60}")
    print(f"  CAGR:           {result['cagr']:+.2%}")
    print(f"  Sharpe:         {result['sharpe']:.3f}")
    print(f"  Max Drawdown:   {result['max_dd']:.2%}")

    xeon_col  = f'weight_{XEON_TICKER}'
    xeon_mean = result['history'][xeon_col].mean() \
                if xeon_col in result['history'].columns else float('nan')
    print(f"  XEON.DE medio:  {xeon_mean:.1%}")
    print(f"  Periodo:        {result['start'].date()} → {result['end'].date()}")
    print(f"\n  Resultados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

### Nota importante para P3

Para que `run_backtest()` pueda pasar `omega_window_fast` a la estrategia,
`BLOmegaStrategy.__init__()` necesita aceptar ese parámetro y pasarlo
internamente a `run_black_litterman()`. Es un cambio pequeño en P3:

```python
# En bl_omega_strategy.py, añadir al __init__:
def __init__(self, tickers, xeon_ticker=XEON_TICKER,
             recalib_freq=RECALIB_FREQ, gamma=GAMMA,
             categoria_por_ticker=None,
             omega_window_fast=42):     # ← parámetro nuevo
    ...
    self.omega_window_fast = omega_window_fast

# Y en _recalibrate(), pasarlo a run_black_litterman:
bl_result = run_black_litterman(
    returns_df, risk_free_rate,
    xeon_ticker=self.xeon_ticker,
    omega_window_fast=self.omega_window_fast,   # ← pasarlo aquí
)
```

P3 no tiene que reescribir nada más — solo añadir este parámetro.

---

## Cómo probar que tu código funciona

```python
# test_universe_loader.py
from data.universe_loader import load_universe_data, build_lambda_dict

print("=== TEST: Descarga con XEON.DE ===")
data = load_universe_data(
    tickers_riesgo=['XLK', 'GLD', 'TLT'],
    xeon_ticker='XEON.DE',
    start='2020-01-01',
    end='2023-12-31'
)

assert 'XEON.DE' in data['todos_tickers'], "ERROR: XEON.DE no está en todos_tickers"
assert 'XEON.DE' not in data['tickers_riesgo'], "ERROR: XEON.DE no debe estar en tickers_riesgo"
assert data['returns'].isnull().sum().sum() == 0, "ERROR: hay NaN en retornos"
print(f"✓ Tickers riesgo: {data['tickers_riesgo']}")
print(f"✓ XEON.DE incluido: {data['todos_tickers']}")
print(f"✓ Sin NaN: OK")

# Verificar que la anti look-ahead está en el motor
# (test conceptual — no se puede verificar con una assertion simple)
print("\n=== RECORDATORIO ANTI LOOK-AHEAD ===")
print("En engine_multiasset.py, línea del bucle principal, verificar que dice:")
print("    returns_hasta_hoy = self.returns_df.loc[:date]")
print("y NO:")
print("    returns_df=self.returns_df  ← ESTO SERÍA LOOK-AHEAD BIAS")
```

---

## Checklist y orden de ejecución

Haz esto en orden. Cada paso desbloquea el siguiente:

- [ ] **Primero:** Actualizar `config.py` con `ETF_UNIVERSE`, `XEON_TICKER` y añadir `BL_OMEGA_WINDOW_FAST`
- [ ] **Segundo:** Crear `data/universe_loader.py` y ejecutar el test de descarga
  - Verificar que detecta la fecha de inicio automáticamente
  - Verificar que XEON.DE está en `todos_tickers` pero no en `tickers_riesgo`
  - Avisar al grupo con la fecha de inicio detectada y cuántos ETFs hay
- [ ] **Tercero:** Crear `engine_multiasset.py`
  - Verificar que el bucle usa `returns_df.loc[:date]` (anti look-ahead)
- [ ] **Cuarto:** Crear `main_black_litterman.py`
  - Verificar que `run_backtest()` funciona llamándola directamente
  - Verificar que acepta `params` con `omega_window_fast`, `gamma`, `recalib_freq`
- [ ] **Miércoles mañana:** Ejecutar `python main_black_litterman.py` (backtest simple)
  - Avisar al grupo cuando termine
- [ ] **Miércoles tarde:** Disponible para ayudar a P5 con el walk-forward si hay problemas

---

## Cuándo avisar por WhatsApp

| Momento | Qué escribir |
|---|---|
| config.py actualizado | "P4: config listo" |
| Datos descargados | "P4: ✓ DATOS LISTOS — X tickers, inicio detectado YYYY-MM-DD" |
| Engine listo | "P4: motor listo" |
| Backtest simple completado | "P4: ✓ BACKTEST COMPLETADO — outputs en outputs/bl_omega/" |
| Error en descarga | Pega el error inmediatamente, no esperes |

---

## Dependencias

```
pip install yfinance pandas-datareader numpy pandas openpyxl
```
