"""
Configuración global del proyecto.
Parámetros del modelo, datos y backtesting.
"""
import os
# Directorio con los CSVs de BID/ASK de Bloomberg
BIDASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bidask')

# ============================================================
# PARÁMETROS DEL MODELO
# ============================================================

GAMMA = -2                 # Utilidad potencial U(w) = w^γ / γ
                             # Coef. aversión relativa al riesgo = 1 - γ
                             # (Estándar en la literatura: RRA ∈ [1, 5])
                             # Con RRA=2: α* = (μ-r)/(2σ²), fracciones razonables (~0.5-1.0)
MAX_LEVERAGE = 1.0           # Apalancamiento efectivo (normas permiten 200%, se limita por prudencia)
MIN_POSITION = 0.01          # Posición mínima (1% del portfolio)
INITIAL_WEALTH = 10_000_000  # Patrimonio inicial: 10M EUR

# ============================================================
# PARÁMETROS DE ESTIMACIÓN
# ============================================================

EWMA_LAMBDA = 0.94           # Factor de decaimiento EWMA (RiskMetrics)
ROLLING_WINDOW_MU = 252      # Ventana para estimar μ (1 año)
ROLLING_WINDOW_SIGMA = 126   # Ventana para estimar σ (6 meses)
EWMA_WINDOW_SIGMA = 21       # Ventana EWMA para σ (1 mes)
RECALIBRATION_FREQ = 5       # Recalibrar fronteras cada N días hábiles

# Shrinkage de μ (Jorion 1986, Black-Litterman)
# μ_shrunk = MU_SHRINKAGE · μ_rolling + (1 - MU_SHRINKAGE) · μ_prior
# Justificación: μ es el parámetro más difícil de estimar, aplicar shrinkage
# hacia una prima de riesgo histórica reduce el error de estimación
MU_SHRINKAGE = 0.15          # Factor de shrinkage (0=solo prior, 1=solo rolling)
                             # 0.15 = muy fuerte shrinkage, μ es el parámetro más difícil
                             # de estimar (Merton 1980: se necesitan siglos de datos)
                             # Se da ~85% peso al prior (r + ERP)
MU_PRIOR_ERP = 0.05          # Prima de riesgo equity prior: 5% anual
                             # (Consistente con Dimson-Marsh-Staunton 2002)

# ============================================================
# PARÁMETROS DE ESTRATEGIAS
# ============================================================

# Benchmark: Merton puro
REBALANCE_FREQ_BENCHMARK = 5     # Rebalanceo cada 5 días (semanal)

# E2: Momentum
MOMENTUM_LOOKBACK = 252          # 12 meses
MOMENTUM_SKIP = 21               # Excluir último mes (12-1)
ALPHA_MIN_MOMENTUM = 0.10        # Posición mínima cuando momentum < 0

# E3: Volatility Targeting
SIGMA_TARGET = 0.15              # Volatilidad objetivo: 15% anualizado

# E4: Régimen Defensivo
E4_VOL_CAUTION_MULT = 1.3        # Umbral cauto: σ ≥ 1.3 × σ_target (19.5%)
E4_VOL_CRISIS_MULT = 2.0         # Umbral crisis: σ ≥ 2.0 × σ_target (30%)
E4_TREND_WINDOW = 200            # SMA(200) para señal de tendencia (días)
E4_BAND_COST_MULT = 2.0          # Multiplicador de λ en bandas DN → bandas más anchas

# E5: Drawdown Shield
E5_DD_ENTER = -0.08              # Drawdown para entrar en modo defensivo: -8%
E5_DD_EXIT = -0.03               # Drawdown para salir de defensivo: -3% (histéresis)
E5_PEAK_WINDOW = 252             # Ventana para calcular el peak (1 año, 252 días hábiles)
E5_BAND_COST_MULT_NORMAL = 4.0   # Multiplicador λ en modo normal → bandas MUY anchas
E5_BAND_COST_MULT_DEFENSIVE = 1.0 # Multiplicador λ en modo defensivo → bandas estándar
E5_VOL_CRISIS_MULT = 2.0         # Válvula seguridad: σ ≥ 2×σ_target → defensivo siempre

# ============================================================
# COSTES DE TRANSACCIÓN
# ============================================================

# Costes proporcionales por defecto (λ_L = λ_M)
DEFAULT_LAMBDA_L = 0.002         # Coste de compra: 0.2%
DEFAULT_LAMBDA_M = 0.002         # Coste de venta: 0.2%

# ============================================================
# DATOS
# ============================================================

# ETFs de acumulación (total return) — no requieren ajuste por dividendos
# Ambos denominados en EUR; se ejecuta backtest por separado sobre cada uno
from data.transaction_costs import load_lambda_from_bidask

ETF_CONFIGS = {
    "MSE.PA": {
        "name": "Euro Stoxx 50 ETF",
        "description": "Amundi EURO STOXX 50 II UCITS ETF Acc - Euronext Paris",
        "fallback_tickers": ["C50.PA", "EUN2.DE"],
        "lambda_L": load_lambda_from_bidask(BIDASK_DIR, "MSE.PA"),
        "lambda_M": load_lambda_from_bidask(BIDASK_DIR, "MSE.PA"),
    },
    "IUSE.L": {
        "name": "S&P 500 EUR Hedged ETF",
        "description": "iShares S&P 500 EUR Hedged UCITS ETF Acc - LSE",
        "fallback_tickers": ["IUSE.DE"],
        "lambda_L": load_lambda_from_bidask(BIDASK_DIR, "IUSE.L"),
        "lambda_M": load_lambda_from_bidask(BIDASK_DIR, "IUSE.L"),
    },
    "IEMA.L": {
        "name": "Emerging Markets ETF",
        "description": "iShares MSCI EM UCITS ETF Acc - LSE",
        "fallback_tickers": ["IEMA.DE", "EIMI.L"],
        "lambda_L": load_lambda_from_bidask(BIDASK_DIR, "IEMA.L"),
        "lambda_M": load_lambda_from_bidask(BIDASK_DIR, "IEMA.L"),
    },
    "IUSN.DE": {
        "name": "World Small Cap ETF",
        "description": "iShares MSCI World Small Cap UCITS ETF Acc - XETRA",
        "fallback_tickers": ["IUSN.L", "WSML.L"],
        "lambda_L": load_lambda_from_bidask(BIDASK_DIR, "IUSN.DE"),
        "lambda_M": load_lambda_from_bidask(BIDASK_DIR, "IUSN.DE"),
    },
}
DEFAULT_ETF_TICKER = "MSE.PA"

RISK_FREE_FRED = "ECBDFR"           # ECB Deposit Facility Rate (FRED) — denominado en EUR
VIX_TICKER = "^VIX"                 # VIX para E3
# Periodos
BACKTEST_START = "2010-01-01"
BACKTEST_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
SIMULATION_START = "2026-02-01"
SIMULATION_END = "2026-03-31"

# Para estimación necesitamos datos previos al backtest
DATA_START = "2008-01-01"          # 2 años antes para warm-up de estimadores

# ============================================================
# TRADING DAYS
# ============================================================

TRADING_DAYS_PER_YEAR = 252
