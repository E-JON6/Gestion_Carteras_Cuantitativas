# OVERVIEW DEL PROYECTO — Lee esto primero
### Gestión de Carteras Cuantitativas — Semana del jueves
### VERSIÓN DEFINITIVA — Para compartir con todo el equipo

---

## Qué estamos construyendo

Un sistema que gestiona una cartera de ETFs de forma automática usando matemáticas.
Cada día el sistema mira el mercado, decide qué ETFs comprar/vender y genera las órdenes.

**Regla de oro del proyecto: la cartera SIEMPRE está 100% invertida.**
No puede haber cash. En todo momento el dinero está repartido entre ETFs de riesgo
y XEON.DE (el ETF monetario que replica la tasa BCE). Si no hay ETFs de riesgo buenos,
todo está en XEON.DE. Si los ETFs de riesgo merecen el 100%, XEON.DE tiene 0%.
Pero la suma siempre es exactamente 1.

El flujo completo cada día es:

```
Retornos recientes de 20+ ETFs (incluido XEON.DE)
              ↓
         OMEGA RATIO                          ← P1
   (¿qué ETFs de RIESGO van bien ahora?)
   IMPORTANTE: XEON.DE no genera views aquí
              ↓
       BLACK-LITTERMAN                        ← P1
   (retornos esperados robustos)
   XEON.DE sí participa en Sigma y prior
              ↓
    MERTON MULTIVARIANTE                      ← P2
   (¿cuánto poner en cada ETF de riesgo?)
   Resultado: pesos que suman ≤ 1
   Régimen de volatilidad controla el cap
              ↓
   peso_XEON = 1 - suma(pesos_riesgo)         ← P2/P3
   (XEON.DE absorbe el complemento)
              ↓
        DAVIS-NORMAN                          ← P3
   (¿hay que operar HOY o esperamos?)
   Aplica a ETFs de riesgo Y a XEON.DE
              ↓
  ÓRDENES (siempre en pares)                  ← P3
  Fila 1: ETF(s) de riesgo
  Fila 2: XEON.DE como contrapartida
```

---

## El papel de XEON.DE — leerlo bien

XEON.DE es el ticker de un ETF monetario que cotiza en Xetra (Bolsa de Frankfurt).
Replica la tasa €STR del BCE — es decir, sube aproximadamente lo que sube el tipo
de interés del Banco Central Europeo. Su precio varía cada día igual que cualquier ETF.

**Por qué es importante entender esto:**
- No es "cash". Es un activo cotizado con precio, bid/ask, y coste de transacción real.
- El sistema siempre tiene participaciones reales de XEON.DE en cartera (puede ser 0 si Merton quiere 100% en riesgo, pero generalmente habrá algo).
- Cada vez que compramos un ETF de riesgo, simultáneamente vendemos XEON.DE para financiarlo.
- Cada vez que vendemos un ETF de riesgo, simultáneamente compramos XEON.DE con lo que ingresamos.
- Por eso cada rebalanceo genera **siempre 2 filas** en el Excel de operativa.

**Qué hace XEON.DE en cada módulo:**

| Módulo | Qué hace con XEON.DE |
|---|---|
| P1 — Omega | NO calcula Omega de XEON.DE. Se excluye del ranking. |
| P1 — BL Prior | SÍ participa. Entra en Sigma y en el prior de equilibrio. |
| P1 — BL Views | NO genera views. Solo los ETFs de riesgo generan views. |
| P2 — Merton | NO entra en el cálculo de pesos. Su peso = 1 - suma(riesgo). |
| P3 — DN | SÍ tiene banda DN. Su peso también puede salirse de banda. |
| P3 — Simulación | SÍ genera orden propia. Siempre hay una fila de XEON.DE. |
| P4 — Datos | SÍ se descarga por separado (igual que el resto de ETFs). |
| P4 — Portfolio | SÍ guarda participaciones reales de XEON.DE en el estado. |

---

## Quién hace qué

| Persona | Archivos que crea | Depende de | Entrega para |
|---|---|---|---|
| **P1** | `models/black_litterman.py` | Nadie (trabaja con datos de P4) | P2, P3 |
| **P2** | `models/merton_multivariate.py` | P1 | P3 |
| **P3** | `strategies/bl_omega_strategy.py` + `simulacion_diaria_bl.py` | P1 + P2 + DN existente | P4 |
| **P4** | `data/universe_loader.py` + `engine_multiasset.py` + `main_black_litterman.py` + config | Nadie | Todos |
| **P5** | `metrics_bl.py` + `plots_bl.py` + `run_analysis.py` | P4 | Presentación |

---

## El calendario de la semana

```
LUNES
  P4: ⚠ URGENTE — descargar datos del universo incluyendo XEON.DE
      La fecha de inicio del backtest se detecta automáticamente
      según qué ETFs tienen historia suficiente (~2015-2016)
      → avisar al grupo: "P4: ✓ DATOS LISTOS — inicio detectado YYYY-MM-DD"
  P1: empezar black_litterman.py (con XEON.DE excluido de views)
  P2: empezar merton_multivariate.py (sin XEON.DE, pesos suman ≤1)
  P3: leer su documento, entender bien el rol de XEON.DE
  P5: preparar scripts con datos sintéticos, incluido walk_forward_bl.py

MARTES
  P1: ✓ black_litterman.py con tests pasando
  P2: ✓ merton_multivariate.py con tests pasando
  P3: ✓ bl_omega_strategy.py con tests pasando
      ✓ añadir parámetro omega_window_fast al __init__ (para walk-forward)
  P4: ✓ engine_multiasset.py y main_black_litterman.py listos
  P5: ✓ todos los scripts de análisis y walk-forward listos

MIÉRCOLES
  P4: primera hora → ejecutar python main_black_litterman.py (backtest simple)
      → avisar al grupo: "P4: ✓ BACKTEST COMPLETADO"
  P5: en cuanto P4 avise →
        1. python run_analysis.py  (15 min)
        2. python walk_forward_bl.py  (1-3 horas, lanzarlo y esperar)
  P3: probar simulacion_diaria_bl.py en modo prueba

JUEVES (entrega)
  Todo integrado y funcionando
  P5 reporta: Sharpe IS medio, OOS medio, ratio OOS/IS
```

---

## Las interfaces entre módulos — LO MÁS IMPORTANTE

Estas firmas de funciones son un contrato. Si alguien cambia la forma de lo que
devuelve su función, rompe el trabajo de todos los demás. No las toques sin avisar.

### P1 → P2 y P3: `run_black_litterman()`

```python
# P1 implementa esta función en models/black_litterman.py
# IMPORTANTE: returns_df incluye XEON.DE como columna.
# La función se encarga internamente de excluirlo de las views.
resultado = run_black_litterman(returns_df, risk_free_rate, xeon_ticker='XEON.DE')

# resultado es un dict con EXACTAMENTE estas keys:
# {
#   'mu_BL':        np.array shape (N,)   ← retornos esperados, uno por ETF
#                                            INCLUYE XEON.DE (necesario para Sigma)
#   'Sigma':        np.array shape (N,N)  ← covarianza anualizada, INCLUYE XEON.DE
#   'tickers':      list de N strings     ← todos los tickers, INCLUYE XEON.DE
#   'top_tickers':  list de 3 strings     ← top-3 por Omega, NUNCA XEON.DE
#   'omega_scores': pd.Series             ← Omega de ETFs de riesgo (sin XEON.DE)
#   'Q':            np.array shape (3,)   ← retornos de las views
#   'confidence':   np.array shape (3,)   ← confianza en cada view
# }
```

### P2 → P3 y P4: `compute_optimal_weights()`

```python
# P2 implementa esta función en models/merton_multivariate.py
# IMPORTANTE: calcula pesos SOLO para ETFs de riesgo. XEON.DE NO entra.
# Los pesos pueden sumar menos de 1. El complemento va a XEON.DE.
resultado = compute_optimal_weights(bl_result, risk_free_rate, xeon_ticker='XEON.DE',
                                    sigma_mercado=0.15)

# resultado es un dict con EXACTAMENTE estas keys:
# {
#   'weights_array':      np.array shape (N,) ← pesos ETFs de riesgo + 0.0 en XEON.DE
#                                               suma ≤ 1 (el resto va a XEON.DE)
#   'weights_dict':       dict {ticker: peso} ← solo ETFs de riesgo con peso > 0
#   'selected_tickers':   list               ← ETFs de riesgo en cartera (≤5)
#   'weight_xeon':        float              ← peso asignado a XEON.DE (= 1 - suma)
#   'w_raw':              np.array shape (N,) ← pesos brutos antes de restricciones
#   'regime':             str                ← 'normal', 'caution' o 'crisis'
# }
```

### P3 → P4: `BLOmegaStrategy.decide()`

```python
# P3 implementa esta clase en strategies/bl_omega_strategy.py
estrategia = BLOmegaStrategy(tickers=tickers, xeon_ticker='XEON.DE')
decision = estrategia.decide(
    date=pd.Timestamp,
    current_weights=np.array,    # shape (N,), incluye XEON.DE
    returns_df=pd.DataFrame,     # incluye XEON.DE como columna
    risk_free_rate=float,
    lambda_per_ticker=dict,      # incluye lambda de XEON.DE
    day_index=int
)

# decision es None (no operar) o:
# {
#   'rebalance':       True,
#   'target_weights':  np.array shape (N,), suma exactamente 1.0,
#                      el peso de XEON.DE es el complemento de los de riesgo
#   'reason':          str
# }
```

---

## Reglas anti-errores críticos

Estas son las cosas que más frecuentemente rompen la integración:

**1. Look-ahead bias (el más peligroso)**
El motor (P4) SIEMPRE pasa a la estrategia solo los datos hasta la fecha actual:
```python
returns_hasta_hoy = returns_df.loc[:date]   # ← CORRECTO
returns_completo  = returns_df              # ← INCORRECTO, usa datos futuros
```
P1, P2 y P3 no tienen que preocuparse de esto — P4 lo garantiza.
Pero si alguien hace pruebas locales, recordarlo.

**2. Shapes de arrays**
- `mu_BL` siempre shape `(N,)`, nunca `(N,1)`. Si falla: añadir `.flatten()`.
- `weights_array` siempre shape `(N,)` donde N = número total de tickers incluyendo XEON.DE.
- El índice de XEON.DE en el array corresponde a su posición en la lista `tickers`.

**3. La suma de pesos**
- P2 devuelve pesos de riesgo que suman ≤ 1.
- P3 añade XEON.DE para completar hasta exactamente 1.0.
- El array `target_weights` que recibe P4 debe sumar 1.0 ± 0.001.

**4. Congelados**
Un ETF congelado es uno que salió del top-5 pero ya estaba en cartera.
Se mantiene con su peso actual hasta que su peso caiga por debajo del 2%.
Cuando cae del 2% → se liquida en el siguiente rebalanceo.
P3 gestiona este estado internamente.

---

## Estructura de carpetas del proyecto

```
gestion_cuantitativa/
│
├── config.py                         ← P4 actualiza (añade universo + XEON.DE)
│
├── data/
│   ├── universe_loader.py            ← P4 CREA
│   │   (detecta automáticamente la fecha de inicio según historia disponible)
│   ├── transaction_costs.py          ← Ya existe, no tocar
│   └── cache/
│       └── universe_prices.csv       ← Generado por P4
│
├── models/
│   ├── black_litterman.py            ← P1 CREA
│   ├── merton_multivariate.py        ← P2 CREA
│   └── davis_norman.py               ← Ya existe, no tocar
│
├── strategies/
│   └── bl_omega_strategy.py          ← P3 CREA
│   (añadir parámetro omega_window_fast al __init__ para walk-forward)
│
├── engine_multiasset.py              ← P4 CREA
├── main_black_litterman.py           ← P4 CREA (expone run_backtest() para P5)
├── simulacion_diaria_bl.py           ← P3 CREA
├── run_analysis.py                   ← P5 CREA
├── walk_forward_bl.py                ← P5 CREA
├── metrics_bl.py                     ← P5 CREA
└── plots_bl.py                       ← P5 CREA

outputs/
├── bl_omega/                         ← Backtest simple (P4 + P5)
│   ├── wealth_history.csv
│   ├── trades.csv
│   ├── metricas_comparativas.csv
│   ├── metricas_comparativas.png
│   ├── grafico_wealth.png
│   ├── grafico_drawdown.png
│   ├── grafico_pesos.png
│   └── grafico_xeon.png
│
└── walk_forward/                     ← Validación walk-forward (P5)
    ├── resultados_wf.csv
    ├── mejores_params_por_ventana.csv
    ├── wealth_oos_concatenado.csv
    ├── grafico_wf_sharpe.png
    └── grafico_wf_wealth.png
```

---

## Reglas del grupo

1. **Si no puedes con algo más de 30 minutos, avisa al grupo.** No te quedes bloqueado solo.
2. **No toques los archivos de otro** sin avisar primero por WhatsApp.
3. **Commitea en GitHub al final de cada sesión**, aunque no esté terminado.
4. **P4 es la prioridad absoluta del lunes.** Si P4 tiene problemas con la descarga de datos, todos ayudan.
5. **Cada uno tiene su documento** con el código ya escrito. Copiar, leer los comentarios, testear.

---

## Glosario

| Término | Qué significa en la práctica |
|---|---|
| **Retorno** | Cuánto subió o bajó un ETF ese día. 0.012 = subió 1.2% |
| **Volatilidad (σ)** | Cuánto varía el precio. Alta = más riesgo |
| **mu (μ)** | Retorno esperado anualizado. 0.08 = esperamos ganar 8% al año |
| **Omega Ratio** | Ganancias recientes dividido pérdidas recientes. >1 es buena señal |
| **View BL** | Nuestra opinión sobre el retorno futuro de un activo |
| **Black-Litterman** | Mezcla nuestras views con lo que dice el mercado para estimar mu |
| **Merton** | Calcula el peso óptimo por activo dado mu y la matriz de correlaciones |
| **Davis-Norman** | Si el peso actual está dentro de la banda, no operar. Si se sale, rebalancear |
| **Banda DN** | El rango de peso tolerable alrededor del objetivo antes de operar |
| **Lambda (λ)** | Coste de transacción. 0.002 = 0.2% por operación |
| **XEON.DE** | ETF monetario €STR. El activo "seguro" donde va lo que no está en riesgo |
| **Congelado** | ETF que salió del top-5 pero se mantiene en cartera hasta que su peso < 2% |
| **Look-ahead bias** | Error de usar datos futuros para decisiones pasadas. Lo evita P4 |
| **Walk-forward** | Validación que divide el histórico en ventanas IS/OOS para demostrar robustez |
| **IS (in-sample)** | Periodo donde buscamos los mejores parámetros |
| **OOS (out-of-sample)** | Periodo donde medimos si esos parámetros funcionan en datos no vistos |
| **Ratio OOS/IS** | Si es ≥ 0.6, la estrategia es robusta. Si es < 0.4, hay sobreajuste |
| **Universo dinámico** | El número de ETFs disponibles crece a medida que van apareciendo históricamente |
| **Opción C** | Estrategia de fecha de inicio: empezar cuando hay ≥15 ETFs con historia completa |
| **Régimen de volatilidad** | Si la volatilidad del mercado es muy alta, se reduce el peso en riesgo |
| **Sharpe Ratio** | Rentabilidad ajustada por riesgo. Cuanto mayor mejor |
| **Drawdown** | Cuánto ha caído la cartera desde su máximo histórico |
| **CAGR** | Rentabilidad anual compuesta |
| **Backtest** | Simular la estrategia con datos del pasado para ver cómo habría ido |
