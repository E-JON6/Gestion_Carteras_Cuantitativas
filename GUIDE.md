# Guía del Proyecto — Trading Strategy Simulator

## Qué es esto

Un framework para diseñar, simular y comparar estrategias de trading sobre ETFs.
Descargamos datos históricos, ejecutamos estrategias sobre esos datos, y medimos
cómo habrían funcionado (backtest).

---

## Parte 1 — Estructura del proyecto

```
src/
├── domain/           ← Los bloques básicos (Portfolio, Signal, Broker...)
├── models/           ← Matemáticas financieras (Merton, Black-Litterman...)
├── strategy/         ← Estrategias de trading
├── backtesting/      ← Motor de simulación y resultados
├── metrics/          ← Métricas de rendimiento (Sharpe, drawdown...)
└── io/               ← Carga de datos (Yahoo Finance, CSV, YAML)

inputs/universes/     ← Archivos YAML con los activos
notebooks/            ← Jupyter notebooks para análisis
tests/                ← Tests
```

---

## Parte 2 — Conceptos clave

### El flujo de cada día de simulación

```
Strategy genera SEÑALES (qué peso quiero en cada activo)
         ↓
Broker convierte señales en ÓRDENES
         ↓
Broker VALIDA (¿hay cash? ¿excede leverage? ¿short permitido?)
         ↓
Broker EJECUTA (vende primero, compra después)
         ↓
Portfolio se ACTUALIZA (nuevas posiciones y cash)
```

### Signal

Una señal dice: "quiero tener el X% de mi cartera en el ticker Y".

```python
Signal(date=..., ticker="XLK", target_weight=0.25, reason="merton")
```

### Portfolio

Guarda el estado: cash + posiciones (shares por ticker).

```python
portfolio = Portfolio(
    initial_cash=10_000_000,
    max_leverage=1.0,    # 1.0 = sin apalancamiento, 2.0 = puedo invertir 2x
    allow_short=False,   # True = puedo apostar en contra de un activo
)
```

### Universe

Lista de activos con sus costes de transacción y sector.

```yaml
# inputs/universes/jaime.yaml
XLK:
  sector: technology
  cost: 0.0027
XEON.DE:
  sector: money_market
  cost: 0.0
```

---

## Parte 3 — Cómo crear una estrategia

Toda estrategia hereda de `Strategy` y solo implementa `_warmup` y `_generate_signals`:

```python
from dataclasses import dataclass, field
from src.strategy.base import Strategy
from src.domain.trade import Signal

@dataclass(kw_only=True)
class MiEstrategia(Strategy):

    mi_parametro: float = 0.5

    def __post_init__(self):
        if not self.name:
            self.name = "mi_estrategia"

    def _warmup(self, portfolio):
        # Se llama una vez con los datos de warmup.
        # Aquí inicializas lo que necesites.
        pass

    def _generate_signals(self, prices, portfolio):
        # Se llama cada día de simulación.
        # Devuelve lista de Signal o [] si no quieres operar hoy.
        return [
            Signal(date=prices.date, ticker="XLK", target_weight=0.5, reason="mi_razon"),
            Signal(date=prices.date, ticker="XEON.DE", target_weight=0.5, reason="defensive"),
        ]
```

### Herramientas disponibles dentro de `_generate_signals`

| Qué necesitas | Cómo acceder |
|---|---|
| Historial de precios | `self._history` (PriceHistory) |
| Precios de hoy | `prices` (PriceSnapshot) |
| Pesos actuales | `portfolio.positions_weights(prices)` |
| Cash actual | `portfolio.cash` |
| Valor total | `portfolio.total_value(prices)` |
| Log returns | `self._history.log_returns()` |
| Sub-historial | `self._history.select(["XLK", "GLD"])` |

---

## Parte 4 — Piezas reutilizables

### Signalers (generan señales)

Están en `src/strategy/signaler/`. No son estrategias — son componentes que una estrategia puede usar.

```python
from src.strategy.signaler import MertonSignaler

signaler = MertonSignaler(
    risky_tickers=["XLK", "GLD", "TLT"],
    defensive_ticker="XEON.DE",
    gamma=0.5,
    mu_estimator=JamesSteinMean(),      # estimador de retornos
    cov_estimator=LedoitWolfCovariance(), # estimador de covarianza
)

signals = signaler.generate(history, prices, portfolio)
```

**Signalers disponibles:**

| Signaler | Qué hace |
|---|---|
| `MertonSignaler` | Pesos óptimos de Merton |
| `MomentumSignaler` | Escala pesos por momentum 12-1 |
| `VolTargetingSignaler` | Escala pesos para target de volatilidad |
| `DualMomentumSignaler` | Market on/off + filtro por momentum relativo |

### Filters (modifican señales)

Están en `src/strategy/filter/`. Reciben señales y devuelven señales filtradas.

```python
from src.strategy.filter import NoTradeBandFilter, CostAwareFilter, TopNByWeightFilter

# Solo operar si el peso se sale de la banda Davis-Norman
signals = NoTradeBandFilter(...).filter(signals, portfolio, prices)

# Solo las top-5 señales por peso
signals = TopNByWeightFilter(n=5).filter(signals)

# Escalar compras para que quepan en el cash disponible
signals = CostAwareFilter(transaction_costs=...).filter(signals, portfolio, prices)
```

**Filters disponibles:**

| Filter | Qué hace |
|---|---|
| `TopNByWeightFilter` | Queda con las N señales de mayor peso |
| `TopNFilter` | Queda con las N señales por Omega ratio |
| `NoTradeBandFilter` | Solo opera si el peso sale de la banda DN |
| `CostAwareFilter` | Escala compras al cash disponible |
| `FrozenAssetManager` | Gestiona activos congelados |

### Transformers (utilidades)

Están en `src/strategy/transformers/`.

```python
from src.strategy.transformers import normalise_weights, normalise_signals, fill_defensive

# Normalizar pesos respetando max_leverage y allow_short
weights = normalise_weights(weights_dict, portfolio, max_fraction=0.8)

# Añadir señal del ticker defensivo (absorbe el resto)
signals = fill_defensive(signals, risky_tickers, "XEON.DE", prices, portfolio)
```

### Modelos (matemáticas)

Están en `src/models/`.

| Modelo | Qué hace |
|---|---|
| `MertonDavisNormanModel` | Pesos óptimos + bandas de no-transacción |
| `BlackLittermanModel` | Combina equilibrio de mercado con views |
| `OmegaRanker` | Rankea activos por ratio Omega |
| `VolatilityRegimeDetector` | Detecta régimen: normal / cautela / crisis |
| `LeverageGuard` | Detecta si el leverage se excedió |

### Estimadores

Están en `src/models/estimators/`. Todos reciben `PriceHistory` y devuelven dicts.

```python
from src.models.estimators import JamesSteinMean, LedoitWolfCovariance

mu = JamesSteinMean().estimate(history)          # {"XLK": 0.12, "GLD": 0.05, ...}
cov = LedoitWolfCovariance().estimate(history)   # DataFrame (tickers × tickers)
```

**Estimadores de retorno (mu):**
`HistoricalMean`, `EwmaMean`, `JamesSteinMean`, `RollingMu`, `EwmaMu`, `BLOmegaMu`

**Estimadores de volatilidad (sigma):**
`HistoricalVolatility`, `EwmaVolatility`, `RollingSigma`, `EwmaSigma`

**Estimadores de covarianza (cov):**
`SampleCovariance`, `LedoitWolfCovariance`, `ConstantCorrelationCovariance`, `RollingCovariance`

**Estimadores de risk-free rate:**
`DefensiveRfrEstimator`, `FixedRiskFreeRate`

---

## Parte 5 — Cómo montar un pipeline típico

El patrón de una estrategia completa es:

```
Signaler → Filters → Normalise → fill_defensive → CostAwareFilter
```

Ejemplo del pipeline de `MertonCustomStrategy`:

```python
def _generate_signals(self, prices, portfolio):
    # 1. Merton calcula pesos para todos los risky
    signals = self._signaler.generate(self._history, prices, portfolio)

    # 2. Quedarse con top-5 por peso, liquidar el resto
    signals = self._top_n_filter.filter(signals)

    # 3. Momentum penaliza los que van mal
    signals = self._momentum.scale(signals, self._history)

    # 4. Normalizar a max_risky_fraction
    weights = normalise_weights({s.ticker: s.target_weight for s in signals}, portfolio, 0.8)
    signals = [Signal(date=s.date, ticker=s.ticker, target_weight=weights[s.ticker], ...) for s in signals]

    # 5. Davis-Norman: solo operar si el peso se sale de la banda
    signals = NoTradeBandFilter(...).filter(signals, portfolio, prices)

    # 6. XEON.DE absorbe lo que queda
    signals = fill_defensive(signals, risky_tickers, "XEON.DE", prices, portfolio)

    # 7. Asegurar que las compras caben en el cash
    signals = CostAwareFilter(...).filter(signals, portfolio, prices)

    return signals
```

---

## Parte 6 — Cómo ejecutar un backtest

### Backtest simple

```python
from src.backtesting import Backtest

result = Backtest(
    portfolio=Portfolio(initial_cash=10_000_000),
    universe=universe,
    strategies=[mi_estrategia, buy_and_hold],
    history=history,
    warmup_size=252,   # 1 año de warmup
).run()
```

### Ver resultados

```python
# Tabla comparativa
result.summary_df

# Plots comparativos
result.plot_value()
result.plot_returns()
result.plot_drawdown()
result.plot_annual_returns()
result.plot_rolling_sharpe()

# Una estrategia en particular
strat = result.result("mi_estrategia")
strat.plot_weights()
strat.plot_monthly_heatmap()
strat.plot_return_distribution()

# Datos brutos
strat.combined.snapshots_df     # DataFrame con todo el estado del portfolio
strat.combined.trades_df        # Todas las operaciones
strat.combined.signals_df       # Todas las señales
```

### Tipos de backtest

```python
from src.backtesting import Backtest, WalkForwardBacktest, RollingBacktest

# Standard: warmup fijo, simulación continua
Backtest(warmup_size=252, ...)

# Walk-forward: ventana fija que se desliza (trim del historial)
WalkForwardBacktest(warmup_size=252, window_size=252, ...)

# Rolling: re-inicializa la estrategia en ventanas
RollingBacktest(train_size=504, test_size=63, ...)
```

---

## Parte 7 — Optimización de parámetros

### Grid search

```python
from src.backtesting import GridSearchOptimizer, ParameterGrid

optimizer = GridSearchOptimizer(
    strategy_cls=MiEstrategia,
    param_grid=ParameterGrid({
        "mi_parametro": [0.3, 0.5, 0.8],
        "otro_param": [10, 21],
    }),
    metric="sharpe_ratio",
    warmup_size=252,
    base_strategy_kwargs={"defensive_ticker": "XEON.DE"},
)

result = optimizer.optimize(universe, history, lambda: Portfolio(initial_cash=10_000_000))
print(result.best_params)
print(result.best_metric)
```

### Random search (para muchos parámetros)

```python
from src.backtesting import RandomParameterGrid

optimizer = GridSearchOptimizer(
    strategy_cls=MiEstrategia,
    param_grid=RandomParameterGrid({...}, n_samples=30, seed=42),
    ...
)
```

### Walk-forward optimization (para detectar overfitting)

```python
from src.backtesting import WalkForwardOptimizer

wf = WalkForwardOptimizer(
    strategy_cls=MiEstrategia,
    param_grid=ParameterGrid({...}),
    train_size=504,    # 2 años train
    test_size=126,     # 6 meses test
    ...
)
wf_result = wf.run(universe, history, portfolio_factory)
print(wf_result.oos_is_ratio)  # >= 0.6 = robusto, < 0.4 = overfitting
```

---

## Parte 8 — Benchmarks

Toda estrategia debería compararse contra estos:

```python
from src.strategy.buy_and_hold import BuyAndHoldStrategy

# 1. Buy & hold equal-weight
bnh = BuyAndHoldStrategy(universe=universe)

# 2. 60/40 (renta variable global / money market)
bnh_60_40 = BuyAndHoldStrategy(universe=universe, weights={"IWDA.L": 0.6, "XEON.DE": 0.4}, name="60_40")

# 3. Equal-weight rebalanceado mensual
ew_rebal = BuyAndHoldStrategy(universe=universe, rebalance_every=21)

# 4. 100% defensive
defensive = BuyAndHoldStrategy(universe=universe, weights={"XEON.DE": 1.0}, name="100_defensive")
```

Si tu estrategia no supera al equal-weight rebalanceado después de costes, la complejidad no aporta valor.

---

## Parte 9 — Checklist para crear una estrategia nueva

1. **Decide el pipeline**: ¿qué signaler? ¿qué filters? ¿defensive?
2. **Crea el archivo** en `src/strategy/mi_estrategia.py`
3. **Hereda de Strategy**, usa `@dataclass(kw_only=True)`
4. **Implementa `_warmup`**: inicializa signalers, filters, guards
5. **Implementa `_generate_signals`**: el pipeline paso a paso
6. **Siempre acaba con** `fill_defensive` + `CostAwareFilter`
7. **Prueba en un notebook**: compara contra benchmarks
8. **Optimiza parámetros**: grid search o walk-forward
9. **Verifica**: ¿hay órdenes rechazadas? ¿los costes son razonables?

---

## Parte 10 — Estrategias ya implementadas

| Estrategia | Archivo | Qué hace |
|---|---|---|
| `BuyAndHoldStrategy` | `buy_and_hold.py` | Compra y no toca (o rebalancea cada N días) |
| `MertonStrategy` | `merton.py` | Merton + DN bands opcionales + defensive |
| `MertonCustomStrategy` | `merton_custom.py` | Merton top-N + momentum + DN bands |
| `MertonFullStrategy` | `merton_full.py` | Merton + top-N Omega + frozen + vol regime + DN |
| `MertonVolStrategy` | `merton_vol.py` | Merton + vol targeting + DN bands |
| `MertonMomentumStrategy` | `merton_mom.py` | Merton + momentum simple + DN bands |
| `MertonDualMomentumStrategy` | `merton_dual_mom.py` | Merton + dual momentum (market on/off) + DN bands |
