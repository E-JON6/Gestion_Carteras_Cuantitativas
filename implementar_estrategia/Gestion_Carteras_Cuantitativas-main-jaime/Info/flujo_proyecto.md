# Flujo del proyecto


El flujo principal actual es:

`data -> omega -> etf_selector -> black_litterman -> merton -> davis_norman_fake -> registrador -> backtest -> walkforward -> plots`

## Flujo principal archivo por archivo

### `Data/universe.py`

Funcion:
- Define el universo base de ETFs.
- Incluye los ETFs de riesgo y `XEON.DE` como activo monetario.

Recibe:
- No recibe datos externos en esta version 0.

Devuelve:
- La lista de ETFs.
- Los tickers que usara `data_loader.py`.

### `Data/data_loader.py`

Funcion:
- Descarga precios reales con `yfinance`.
- Calcula retornos.
- Calcula un diccionario simple de costes de transaccion.
- Busca el precio de ejecucion del dia siguiente si existe; si no, usa el ultimo disponible.

Recibe:
- `start_date`
- `end_date`
- `tickers`

Devuelve:
- `prices`
- `returns`
- `metadata`: son los etfs 
- `transaction_costs`

### `metricas/omega.py`

Funcion:
- Asigna un score simple a cada ETF.
- En esta version 0 no calcula el Omega real, solo genera una señal minima para que el flujo funcione.

Recibe:
- `returns_df`

Devuelve:
- Un `Series` con un score por ETF.

### `metricas/etf_selector.py`

Funcion:
- Ordena los ETFs por score.
- Se queda con los primeros 5.

Recibe:
- `scores`

Devuelve:
- `selected_etfs`

### `Models/black_litterman.py`

Funcion:
- Representa el bloque Black-Litterman.
- En esta version 0 no estima nada real, solo devuelve una `mu` y una `Sigma` inventadas.

Recibe:
- `returns_df`
- `selected_etfs`
- `omega_scores`

Devuelve:
- `mu_BL`: mu de Black Litterman
- `Sigma`: Correlaciones
- `tickers`
- `top_tickers`: Etfs seleccionados
- `Q`; retorno esperado por Black-Litterman (Sio pone BL es Black_LItterman)
- `confidence`: qué confianza hay de esa Q

### `Models/merton.py`

Funcion:
- Representa el bloque de asignacion optima.
- En esta version 0 reparte los pesos por igual entre los ETFs seleccionados.

Recibe:
- `bl_result`

Devuelve:
- `weights`
- `selected_etfs`
- `weight_sum`
- `weight_xeon`: peso ERF libre de riesgo

### `Models/davis_norman_fake.py`

Funcion:
- Representa la politica de bandas de inaccion.
- En esta version deja una parte fija en riesgo y otra en `XEON.DE`.
- Si la diferencia entre pesos actuales y objetivo supera el umbral, rebalancea.

Recibe:
- `current_weights`
- `target_weights`
- `band`: banda de Davis Norman

Devuelve:
- `rebalance`: Booleano que dice si se rebalancea o no 
- `target_weights`: pesos objetivo
- `final_weights`: pesos rebalanceados
- `weight_xeon`: peso del ETF libre de riesgo
- `reason`: si el target_weight es mayor que el final_weight o no y se supera el umbral.

### `portfolio/registrador.py`

Funcion:
- Convierte los pesos finales en operaciones de rebalanceo.
- Usa cantidades enteras para ETFs de riesgo.
- El sobrante va a `XEON.DE`, que puede tener decimales.
- Genera el Excel de operativa.

Recibe:
- `dn_result`; resultado de Davis Norman (El return de dicho archivo)
- `market_data`
- `current_positions`
- `total_value`

Devuelve:
- `orders`
- `updated_positions`
- `output_path`
- `trade_date`

El Excel generado tiene esta estructura:
- `ID`
- `Cantidad`
- `Precio`
- `CT`
- `Precio Ejecutado`

## Orquestacion (Juntar los archivos para ejecutar)

### `main.py`

Funcion:
- Ejecuta el pipeline completo de una sola corrida.

Orden interno:
1. Descarga datos.
2. Calcula score de Omega.
3. Selecciona ETFs.
4. Ejecuta Black-Litterman.
5. Ejecuta Merton.
6. Ejecuta Davis-Norman.
7. Genera el Excel de operativa con el registrador.

Devuelve:
- Un diccionario con todos los resultados intermedios y finales.

## Backtest

### `backtest/engine.py`

Funcion:
- Ejecuta el flujo principal en varias fechas de revision.
- Guarda un resumen del backtest.
- Guarda una serie simple de riqueza para comparar la estrategia con `SPY`.
- Guarda tambien `Metrics.xlsx`.

Recibe:
- `start_date`
- `end_date`
- `metric_name`
- `freq`

Devuelve:
- `backtest`
- `wealth_history`: hisorico de lo que ganas
- `metrics`
- rutas de salida en `results`

### `main_backtest.py`

Funcion:
- Lanza el backtest version 0.

Devuelve:
- El resumen del backtest ya guardado en `results`.

## Walkforward

### `walkforward/walkforward.py`

Funcion:
- Divide el historico en ventanas de entrenamiento y prueba.
- Reutiliza `backtest/engine.py` para no duplicar logica.

Recibe:
- `start_date`
- `end_date`
- `train_months`
- `test_months`

Devuelve:
- Un resumen por ventana en CSV y Excel.

### `main_walkforward.py`

Funcion:
- Lanza el walkforward version 0.

Devuelve:
- El resumen del walkforward ya guardado en `results`.

## Visualizacion

### `Visualization/plots.py`

Funcion:
- Genera graficos simples a partir de los archivos de `results`.

Devuelve:
- Grafico de estrategia frente a `SP500`
- Grafico del backtest
- Grafico del walkforward

### `Visualization/plot_backtest.py`

Funcion:
- Ejecuta los graficos del backtest.

Devuelve:
- Archivos `.png` guardados en `results`.

### `Visualization/plot_walkforward.py`

Funcion:
- Ejecuta el grafico del walkforward.

Devuelve:
- Archivo `.png` guardado en `results`.

## Archivos que se guardan en `results`

Actualmente la version 0 puede dejar:
- `operaciones_*.xlsx`
- `operaciones_rebalanceo.xlsx`
- `operaciones_rebalanceo_v0.xlsx`
- `backtest_resumen.csv`
- `backtest_resumen.xlsx`
- `walkforward_resumen.csv`
- `walkforward_resumen.xlsx`
- `wealth_history.csv`
- `Metrics.xlsx`
- graficos `.png`



### Sobre una futura version con clases

Una evolucion razonable seria:
- `StrategyPipeline` para coordinar metrica, BL, Merton y Davis-Norman.
- `PortfolioState` para guardar cantidades, pesos y valor.
- `BacktestEngine` para ejecutar el historico.
- `WalkForwardRunner` para lanzar ventanas de entrenamiento y prueba.

