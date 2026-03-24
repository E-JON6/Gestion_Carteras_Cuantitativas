"""
Main version 0.
Recibe: fechas y una metrica a usar.
Devuelve: datos descargados, scores, ETFs seleccionados, Black-Litterman, Merton, Davis-Norman y registro.
"""

from Data.data_loader import download_market_data
from Models.black_litterman import run_black_litterman_v0
from Models.davis_norman_fake import run_davis_norman_fake_v0
from Models.merton import run_merton_v0
from metricas.etf_selector import select_top_etfs_v0
from metricas.omega import compute_omega_scores_v0
from portfolio.registrador import run_registrador_v0


def _get_risk_returns(market_data):
    metadata_by_ticker = {item["ticker"]: item for item in market_data["metadata"]}
    risk_tickers = [
        ticker
        for ticker in market_data["tickers"]
        if metadata_by_ticker.get(ticker, {}).get("role") != "defensive"
    ]

    if not risk_tickers:
        raise ValueError("No hay ETFs de riesgo disponibles para ejecutar el pipeline v0.")

    return market_data["returns"][risk_tickers]



def run_v0(
    start_date="2024-01-01",
    end_date="2024-03-01",
    metric_name="omega",
    output_path="results/operaciones_rebalanceo.xlsx",
):
    market_data = download_market_data(start_date=start_date, end_date=end_date)
    returns_df = _get_risk_returns(market_data)

    if metric_name == "omega":
        scores = compute_omega_scores_v0(returns_df)
    else:
        raise ValueError("La version 0 solo soporta la metrica omega.")

    selected_etfs = select_top_etfs_v0(scores, top_n=5)
    bl_result = run_black_litterman_v0(returns_df, selected_etfs, scores)
    merton_result = run_merton_v0(bl_result)
    current_weights = {ticker: 0.0 for ticker in merton_result["selected_etfs"]}
    dn_result = run_davis_norman_fake_v0(current_weights, merton_result["weights"])
    current_positions = {ticker: 0.0 for ticker in merton_result["selected_etfs"]}
    registrador_result = run_registrador_v0(
        dn_result,
        market_data,
        current_positions=current_positions,
        output_path=output_path,
    )

    return {
        "metric_name": metric_name,
        "market_data": market_data,
        "scores": scores,
        "selected_etfs": selected_etfs,
        "bl_result": bl_result,
        "merton_result": merton_result,
        "dn_result": dn_result,
        "registrador_result": registrador_result,
    }


if __name__ == "__main__":
    result = run_v0(metric_name="omega")
    print("Metrica usada:", result["metric_name"])
    print("ETFs seleccionados:", result["selected_etfs"])
    print("mu_BL:", result["bl_result"]["mu_BL"])
    print("Sigma shape:", result["bl_result"]["Sigma"].shape)
    print("Pesos Merton:", result["merton_result"]["weights"])
    print("Davis-Norman:", result["dn_result"])
    print("Excel guardado en:", result["registrador_result"]["output_path"])
