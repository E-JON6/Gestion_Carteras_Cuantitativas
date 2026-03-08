================================================================================
  SIMULACION DIARIA — GUIA PARA EL EQUIPO
  Master en Finanzas Cuantitativas AFI | Gestion Carteras Cuantitativas
================================================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PARTE 1 — ARRANQUE RAPIDO (leer esto y ya puedes ejecutarlo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHIVO PRINCIPAL: simulacion_diaria.py
Abre ese archivo y busca el bloque de configuracion al principio (lineas ~30-45).
Son las unicas lineas que hay que tocar:

    ETF_TICKER      = "IUSE.L"            <- ticker del ETF elegido
    ESTRATEGIA      = "E4"                <- E1, E2, E3, E4 o E5
    GRUPO           = "X"                 <- vuestro numero de grupo
    CAPITAL_INICIAL = 10_000_000.0        <- capital inicial en EUR

    EMAIL_REMITENTE = "vuestro@outlook.com"
    EMAIL_PASSWORD  = "xxxx xxxx xxxx"    <- ver instruccion abajo
    EMAIL_CC        = "vuestro@outlook.com"

IMPORTANTE — contrasena de email:
  Outlook no permite usar la contrasena normal por SMTP.
  Hay que crear una "contrasena de aplicacion":
    1. Ir a https://account.microsoft.com/security
    2. Activar verificacion en dos pasos (si no esta activa)
    3. Buscar "Contrasenas de aplicacion" -> crear nueva
    4. Copiar el codigo generado en EMAIL_PASSWORD (sin espacios)

ETFs disponibles y sus costes reales (Bloomberg enero 2026):
    "MSE.PA"   -> Euro Stoxx 50          CT = 0.1738%
    "IUSE.L"   -> S&P 500 EUR Hedged     CT = 0.0210%
    "IEMA.L"   -> Emerging Markets       CT = 0.0491%
    "IUSN.DE"  -> World Small Cap        CT = 0.0605%

--------------------------------------------------------------------------------

EJECUCION MANUAL (para probar):
    python simulacion_diaria.py

AUTOMATIZACION (ejecutar UNA sola vez, como Administrador):
    python configurar_tarea_windows.py
  Esto programa la tarea en Windows para que corra sola cada dia habil a las 21h.
  El ordenador debe estar encendido a esa hora.

--------------------------------------------------------------------------------

QUE PASA CADA DIA (automaticamente):

  1. El script descarga el precio de cierre del ETF
  2. Calcula si hay que operar o no
  3. Actualiza el estado de la cartera
  4. Genera los archivos de salida (ver abajo)
  5. Envia el email a aguilabert@afi.es

  - Si hay operacion (COMPRAR/VENDER):
      email con adjunto Operativa_GrupoX_YYYYMMDD.xlsx
  - Si no hay operacion (MANTENER):
      email sin adjunto, informando del estado de la cartera

--------------------------------------------------------------------------------

ARCHIVOS QUE GENERA (carpeta Simulacion_real/):

    Operativa_GrupoX_20260310.xlsx   <- Excel diario, el que se envia a AFI
    Operativa_GrupoX_20260311.xlsx
    ...
    Historial_GrupoX.xlsx            <- acumulativo, crece una fila cada dia
    simulacion_estado.json           <- estado interno (no tocar)
    simulacion_log.csv               <- log tecnico completo dia a dia


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PARTE 2 — COMO FUNCIONA LA ESTRATEGIA (para entenderlo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EL PROBLEMA QUE RESUELVE
-------------------------
Dado un inversor con aversion al riesgo y un ETF con rentabilidad esperada mu
y volatilidad sigma, ¿cuanta fraccion de la cartera (alpha) deberia tener en
el ETF en cada momento?

La respuesta teorica viene del modelo de Merton (1969): hay un alpha optimo
que maximiza la utilidad esperada. El problema es que si rebalanceamos cada dia
a ese optimo, los costes de transaccion se comen toda la rentabilidad.

La solucion es Davis-Norman (1990): no rebalancear continuamente, sino dejar
que alpha se mueva dentro de unas bandas y solo actuar cuando se sale de ellas.


LOS PARAMETROS CLAVE
---------------------
GAMMA = -2 → parámetro de aversión al riesgo del inversor en la función de utilidad.
    El valor que importa en la práctica es RRA = 1 - GAMMA = 3, que es la "Aversión
    Relativa al Riesgo". Un RRA de 1 sería un inversor agresivo, de 10 sería extremadamente
    conservador. Con RRA=3 el inversor acepta riesgo pero lo penaliza: ante dos carteras con
    la misma rentabilidad esperada, siempre prefiere la de menor volatilidad. Es un perfil
    moderado-alto habitual en la literatura académica.

EWMA_LAMBDA = 0.94 → controla cómo se estima la volatilidad diaria. EWMA significa que
    no tratamos todos los días históricos igual: los más recientes pesan más que los
    lejanos. Lambda=0.94 significa que cada día pesa un 6% menos que el anterior. Con
    0.94 la "memoria efectiva" del modelo es de unos 3 meses. Es el valor estándar publicado
    por RiskMetrics (JP Morgan, 1994) y el más usado en la industria.

MU_SHRINKAGE = 0.15 → la rentabilidad esperada (mu) es imposible de estimar con precisión.
    Si usamos solo el histórico del ETF, en años buenos nos da una mu altísima y en años
    malos una bajísima, lo que hace que el modelo tome posiciones extremas. Para evitarlo,
    "empujamos" la estimación un 15% hacia un valor razonable de largo plazo (7% anual).
    El 85% restante sigue siendo la mu histórica del ETF. Es una técnica bayesiana estándar
    llamada shrinkage.
            mu_final = 0.85 × mu_histórica + 0.15 × 7%

SIGMA_TARGET = 0.15 → volatilidad anual objetivo del 15%. Solo lo usan las estrategias E3,
    E4 y E5. La idea es que en lugar de apuntar al alpha óptimo de Merton, apuntamos a tener
    siempre una volatilidad de cartera del 15%. Si el mercado está tranquilo (sigma baja)
    subimos exposición, si está agitado (sigma alta) la reducimos. Es una técnica muy común
    en fondos de gestión sistemática.

MAX_LEVERAGE = 1.0 → alpha nunca puede superar 1.0, es decir, nunca invertimos más del 100%
    del capital en el ETF. No hay apalancamiento. Si el modelo quisiera poner un alpha de 1.4,
    lo recortamos a 1.0.


EL MODELO PASO A PASO (cada dia)
----------------------------------
  1. Se descargan todos los precios historicos desde 2008 hasta hoy.

  2. Se estima sigma (volatilidad) con EWMA: da mas peso a lo reciente.
     Es la volatilidad anualizada de hoy mismo.

  3. Se estima mu (rentabilidad esperada) con shrinkage bayesiano:
     mu = 0.85 * mu_historica + 0.15 * 7%
     Esto evita que mu dispare en periodos muy buenos o muy malos.

  4. Se calcula el alpha* optimo de Merton:
     alpha* = (mu - rf) / (RRA * sigma^2)
     Cuanto mayor el exceso de rentabilidad esperado sobre el activo libre
     de riesgo, mayor posicion. Cuanto mayor la volatilidad o la aversion
     al riesgo, menor posicion.

  5. Se calculan las bandas Davis-Norman alrededor de alpha*:
     ancho = sqrt(3 * CT * alpha* / (RRA * sigma^2))
     banda_inf = alpha* - ancho
     banda_sup = alpha* + ancho
     El ancho aumenta cuando los costes son altos o la volatilidad es baja
     (sale mas caro rebalancear en relacion al beneficio).

  6. Si alpha_actual esta dentro de [banda_inf, banda_sup] -> MANTENER
     Si alpha_actual < banda_inf -> COMPRAR (rebalancear a alpha*)
     Si alpha_actual > banda_sup -> VENDER  (rebalancear a alpha*)

  El dinero no invertido en el ETF se considera en activo libre de riesgo
  (tipo BCE).


LAS 5 ESTRATEGIAS
------------------
  E1 — DN Adaptativo
    La implementacion pura de Davis-Norman descrita arriba.
    Sin mas logica adicional.

  E2 — DN + Momentum
    Antes de calcular las bandas, ajusta alpha* segun el momentum del ETF:
    momentum = retorno 12 meses - retorno ultimo mes
    Si momentum positivo, sube alpha*. Si negativo, lo baja.
    Idea: aprovecha tendencias de medio plazo.

  E3 — DN + Vol Targeting
    En lugar de usar directamente alpha* de Merton, apunta a una volatilidad
    fija del 15% en cartera:
    alpha_vol = sigma_target / sigma_actual
    Si el mercado esta muy agitado, reduce exposicion automaticamente.

  E4 — DN + Regimen Defensivo  [CANDIDATA PRINCIPAL]
    Parte de E3 (vol targeting) pero ademas detecta el regimen de mercado
    usando la media movil de 200 dias (SMA200):
    - Precio > SMA200 (tendencia alcista) -> usa alpha_vol normal
    - Precio < SMA200 (tendencia bajista) -> reduce alpha a la mitad
    Idea: ser mas defensivo cuando el mercado esta en tendencia bajista.

  E5 — DN + Drawdown Shield
    Igual que E4 pero añade un segundo nivel de proteccion por drawdown:
    - Si la cartera cae mas de un 8% desde su maximo historico -> modo defensivo
      (alpha se reduce a la mitad adicional)
    - Sale del modo defensivo cuando recupera hasta -3% desde el maximo
    La histéresis (-8% para entrar, -3% para salir) evita entrar y salir
    repetidamente en mercados laterales.


RESULTADOS DEL BACKTEST (resumen)
-----------------------------------
  Periodo In-Sample: 2010-2024
  Periodo Out-of-Sample: 2025

  Sobre S&P 500 (IUSE.L), que es el ETF con mejores resultados:

    Estrategia          Sharpe IS   MaxDD IS    Sharpe OOS
    Buy & Hold           0.787      -34.75%      0.795
    E4 Reg. Defensivo    0.831      -23.77%      0.569  <- mejor Sharpe IS
    E5 Drawdown Shield   0.836      -23.22%      -

  E4 y E5 mejoran el Sharpe y reducen el drawdown maximo respecto a B&H
  en el periodo in-sample. En OOS 2025 (año muy alcista) B&H gano en todos
  los ETFs, lo cual es esperable: las estrategias activas brillan menos
  cuando hay tendencias limpias al alza.

  El coste de transaccion real (λ = 0.021% para IUSE.L, calculado con datos
  BID/ASK de Bloomberg enero 2026) es bajo, lo que favorece la operativa
  activa sobre este ETF frente a los otros tres.


CALENDARIO DEL PROYECTO
------------------------
  12 mar 2026  Presentacion estrategia + backtest + inicio simulacion
  18 mar 2026  Sesion formativa Performance Attribution
  14 abr 2026  Presentacion comportamiento real + VL + ratios + PA
   9 abr 2026  Sesion ML/IA aplicado a gestion Quant
   6 may 2026  Presentacion mejoras ML/IA implementadas
  14 may 2026  Simulacro presentacion final + fin VL
   5 jun 2026  Entrega documentacion (ppt + memoria + material)
  10 jun 2026  Sesion final presentacion resultados


================================================================================
  Dudas: remitir en clase o a las direcciones indicadas por AFI.
  Este documento no sustituye la lectura del codigo — es solo orientacion.
================================================================================