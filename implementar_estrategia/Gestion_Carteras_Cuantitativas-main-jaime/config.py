"""Configuracion centralizada — filosofia contrarian + momentum."""

import pandas as pd

# ============================================================
# UNIVERSO DE ETFs  (42 activos de riesgo + XEON.DE defensivo)
# ============================================================

XEON_TICKER = "XEON.DE"

ETF_UNIVERSE = {
    # ---- Equity: Desarrollada ----
    "IWDA.L":  "global_equity",
    "VGK":     "europe_equity",
    "EWJ":     "asia_developed",
    "IUSN.DE": "global_equity",

    # ---- Equity: Emergente ----
    "IEMA.L":  "emerging_equity",
    "MCHI":    "asia_emerging",
    "EWZ":     "latam",
    "INDA":    "asia_emerging",
    "EWT":     "asia_emerging",
    "EWY":     "asia_emerging",

    # ---- US Sectorial ----
    "XLY":     "us_discretionary",
    "XLC":     "us_communications",
    "XLF":     "us_financials",
    "KBE":     "us_financials",
    "XLE":     "us_energy",
    "XOP":     "us_energy",
    "XLI":     "us_industrials",
    "IYT":     "us_industrials",
    "XLB":     "us_materials",
    "ITB":     "us_housing",
    "XLV":     "us_healthcare",
    "XBI":     "us_healthcare",
    "XLP":     "us_staples",
    "XLU":     "us_utilities",

    # ---- Tech & Innovation ----
    "XLK":     "tech",
    "SOXX":    "tech",
    "IGV":     "tech",
    "CIBR":    "tech",
    "BOTZ":    "tech",

    # ---- Real Estate ----
    "VNQ":     "real_estate",

    # ---- Tematico ----
    "LIT":     "thematic",

    # ---- Commodities (UN solo grupo -> max 35-40%) ----
    "GLD":     "commodities",
    "SLV":     "commodities",
    "DBA":     "commodities",
    "COPX":    "commodities",

    # ---- Renta Fija (UN solo grupo -> max 35-40%) ----
    "TLT":     "fixed_income",
    "IEF":     "fixed_income",
    "TIP":     "fixed_income",
    "HYG":     "fixed_income",
    "LQD":     "fixed_income",
    "EMB":     "fixed_income",

    # ---- Crypto ----
    "BITO":    "crypto",
}

# ============================================================
# TASA LIBRE DE RIESGO DINAMICA (BCE)
# ============================================================

ECB_RATE_HISTORY = [
    ("2014-06-01", -0.001),
    ("2019-09-01", -0.005),
    ("2022-07-01",  0.000),
    ("2022-09-01",  0.0075),
    ("2022-11-01",  0.015),
    ("2022-12-01",  0.020),
    ("2023-02-01",  0.025),
    ("2023-04-01",  0.0325),
    ("2023-06-01",  0.035),
    ("2023-08-01",  0.0375),
    ("2023-10-01",  0.040),
    ("2024-06-01",  0.0375),
    ("2024-09-01",  0.035),
    ("2024-10-01",  0.0325),
    ("2024-12-01",  0.030),
    ("2025-01-01",  0.0275),
    ("2025-03-01",  0.025),
]

BL_RF = 0.025


def get_risk_free_rate(date) -> float:
    ts = pd.Timestamp(date)
    rate = -0.005
    for d, r in ECB_RATE_HISTORY:
        if ts >= pd.Timestamp(d):
            rate = r
    return rate


# ============================================================
# SENAL COMPUESTA  (momentum + trend dominante)
# ============================================================
# drawdown_buy DESACTIVADO: compraba falling knives (bonos 2022, China).
# El factor existe en composite_signal.py pero su peso es 0.

COMPOSITE_MOMENTUM_WINDOW = 126
COMPOSITE_MOMENTUM_SKIP   = 21
COMPOSITE_REVERSAL_WINDOW = 21
COMPOSITE_TREND_WINDOW    = 200
COMPOSITE_VOL_WINDOW      = 63
COMPOSITE_DRAWDOWN_WINDOW = 252

COMPOSITE_WEIGHTS = {
    "momentum":      0.40,
    "reversal":      0.10,
    "trend":         0.35,
    "vol_penalty":   0.15,
    "drawdown_buy":  0.00,
}

CATEGORY_SIGNAL_WEIGHTS = {
    "global_equity":     {"momentum": 0.42, "reversal": 0.15, "trend": 0.33, "vol_penalty": 0.10},
    "europe_equity":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "asia_developed":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "emerging_equity":   {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "asia_emerging":     {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "latam":             {"momentum": 0.40, "reversal": 0.25, "trend": 0.25, "vol_penalty": 0.10},
    "us_discretionary":  {"momentum": 0.44, "reversal": 0.15, "trend": 0.31, "vol_penalty": 0.10},
    "us_communications": {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "us_financials":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_energy":         {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_industrials":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_materials":      {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_housing":        {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_healthcare":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_staples":        {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "us_utilities":      {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "tech":              {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "real_estate":       {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "thematic":          {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "commodities":       {"momentum": 0.35, "reversal": 0.25, "trend": 0.30, "vol_penalty": 0.10},
    "fixed_income":      {"momentum": 0.30, "reversal": 0.25, "trend": 0.35, "vol_penalty": 0.10},
    "crypto":            {"momentum": 0.54, "reversal": 0.05, "trend": 0.36, "vol_penalty": 0.05},
}

# ============================================================
# COVARIANZA ROBUSTA
# ============================================================

COV_SHORT_WINDOW = 63
COV_LONG_WINDOW  = 252
COV_BLEND_ALPHA  = 0.6
EWMA_LAMBDA      = 0.94
COV_SHRINKAGE    = 0.10

# ============================================================
# BLACK-LITTERMAN
# ============================================================

BL_TAU    = 0.05
BL_DELTA  = 2.5
VIEW_SCALE = 0.28

# Prior BL: "equal" (1/N) o "inv_vol" (1/sigma_i / sum) como proxy de mercado
BL_PRIOR_WEIGHTS_MODE = "inv_vol"

# ============================================================
# REGIMEN — FILOSOFIA CONTRARIAN
# ============================================================
# Umbral legacy: vol media cross-sectional anualizada
VOL_CAUTION_THR    = 0.28
VOL_CRISIS_THR     = 0.40
# Multi-factor (ademas de vol_media): cartera EW ultimos REGIME_LOOKBACK_DD dias
REGIME_LOOKBACK_DD = 252
REGIME_EW_MDD_CRISIS = 0.28
REGIME_EW_MDD_CAUTION = 0.16
REGIME_CORR_LOOKBACK = 63
REGIME_AVG_CORR_CRISIS = 0.52
REGIME_AVG_CORR_CAUTION = 0.38
CRISIS_VIEW_BOOST  = 1.30
CAUTION_VIEW_BOOST = 1.15
DN_CRISIS_MULT     = 2.0
DN_CAUTION_MULT    = 1.5

# ============================================================
# MERTON
# ============================================================

MERTON_GAMMA      = -0.8
MERTON_N_TOP      = 20
MERTON_MAX_WEIGHT = 0.40
MERTON_MIN_WEIGHT = 0.01
MERTON_MAX_SECTOR = 0.35

MAX_SECTOR_OVERRIDE = {}  # Sin overrides, MAX_SECTOR aplica uniformemente

# ============================================================
# DAVIS-NORMAN
# ============================================================

DN_BAND     = 0.05
DN_MIN_BAND = 0.02

# ============================================================
# COSTES DE TRANSACCION (por lado, fraccion del nominal)
# ============================================================
# Rango tipico bróker barato para ETFs: ~0,05%–0,12% por compra o venta.
# La version antigua (0.5*spread$/precio) disparaba el coste en ETFs baratos.

TX_COST_PER_SIDE = 0.0008
TX_COST_PER_SIDE_MIN = 0.0005
TX_COST_PER_SIDE_MAX = 0.0012

# ============================================================
# BACKTEST
# ============================================================

# Solo afecta al MOTOR DE BACKTEST (engine): en que fechas se *evalua* el pipeline.
# No limita a Davis-Norman (las bandas son independientes del calendario).
# En vivo (main.py) no hay "una vez al mes" salvo que ejecutes main solo ese dia.
REBALANCE_FREQ   = "ME"
INITIAL_CAPITAL  = 100_000
BENCHMARK_TICKER = "SPY"
# ETF liquido que replica MSCI World (USD); alternativa europea: SWDA.L
BENCHMARK_MSCI_WORLD_TICKER = "URTH"

# ============================================================
# REGISTRADOR (operativa al ejecutar main)
# ============================================================
# Misma logica que backtest/engine y walkforward (run_backtest -> engine):
#   - Pesos: dn['final_weights_full']
#   - Operar si: dn['rebalance'] OR cartera sin posiciones (portfolio.rebalance_policy)
# REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE: si True, no generar ordenes Excel cuando
# el engine tampoco operaria (hay posiciones y DN no rebalancea).
REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE = True
REGISTRADOR_OUTPUT_TEMPLATE = "results/operaciones_rebalanceo_{date}.xlsx"

# Cartera real (titulos): Excel fuente de verdad para main.run_single cuando
# current_positions no se pasa explicitamente. Columnas: Ticker, Cantidad.
POSITIONS_EXCEL_PATH = "results/posiciones_cartera.xlsx"
CREATE_POSITIONS_TEMPLATE_IF_MISSING = True
# Tras cada ejecucion: posiciones teoricas post-rebalanceo (titulos) — archivo
# historizado por fecha; no sustituye posiciones_cartera.xlsx hasta confirmar en bróker.
SAVE_SUGGESTED_POSITIONS = True
POSICIONES_POST_REBALANCEO_TEMPLATE = "results/posiciones_post_rebalanceo_{date}.xlsx"
# Alias retrocompatible:
SUGGESTED_POSITIONS_TEMPLATE = POSICIONES_POST_REBALANCEO_TEMPLATE
# Copia siempre la ultima ejecucion (mismo contenido que la fecha de hoy).
POSICIONES_ULTIMO_SNAPSHOT_PATH = "results/posiciones_post_rebalanceo_ultimo.xlsx"

# Append CSV de cada ejecucion de main.run_single (auditoria).
APPEND_EJECUCION_LOG = True
EJECUCION_LOG_CSV = "results/historial_ejecuciones.csv"

# Informe cartera viva (main_portfolio_backtest.py): reconstruccion desde operaciones_*.xlsx
PORTFOLIO_LIVE_RESULTS_DIR = "results"
PORTFOLIO_LIVE_INITIAL_POSITIONS = None  # Excel opcional antes del primer archivo de ordenes
PORTFOLIO_LIVE_START_DATE = None  # "YYYY-MM-DD" o None = primera fecha de operaciones
PORTFOLIO_LIVE_INFORME_SUBDIR = "informe_cartera_vivo"

# Correo opcional (operaciones_rebalanceo): definir en entorno o dejar vacio.
# SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD,
# MAIL_FROM, MAIL_TO (varios separados por coma), SMTP_USE_TLS (default 1).
EMAIL_OPERACIONES_AFTER_RUN = False

# ============================================================
# WALKFORWARD
# ============================================================

WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS  = 3
