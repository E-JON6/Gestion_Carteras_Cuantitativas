"""
Configuración de los parámetros del modelo
"""


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