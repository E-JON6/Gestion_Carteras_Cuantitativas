"""
Genera un grafico historico simple y profesional con todos los precios del universo.
Usa Data/data_loader.py y Data/universe.py como fuente de datos del proyecto.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Seaborn no esta instalado. Actualiza dependencias e instala requirements.txt antes de usar historic_plot.py."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Data.data_loader import download_market_data
from Data.universe import get_defensive_ticker, get_universe, get_universe_tickers


def _build_palette(tickers: list[str], defensive_ticker: str) -> dict[str, tuple[float, float, float]]:
    risk_tickers = [ticker for ticker in tickers if ticker != defensive_ticker]
    risk_palette = sns.color_palette("tab20", n_colors=max(len(risk_tickers), 1))
    palette = {ticker: risk_palette[index] for index, ticker in enumerate(risk_tickers)}
    if defensive_ticker in tickers:
        palette[defensive_ticker] = sns.color_palette(["#111827"])[0]
    return palette



def _normalize_prices_base_one(prices: pd.DataFrame) -> pd.DataFrame:
    normalized_prices = prices.copy()

    for ticker in normalized_prices.columns:
        valid_prices = normalized_prices[ticker].dropna()
        if valid_prices.empty:
            continue
        normalized_prices[ticker] = normalized_prices[ticker] / valid_prices.iloc[0]

    return normalized_prices



def _build_plot_frame(prices: pd.DataFrame, ordered_tickers: list[str]) -> pd.DataFrame:
    plot_df = prices.copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    plot_df = plot_df.reset_index().rename(columns={plot_df.index.name or "index": "Date"})
    plot_df = plot_df.melt(id_vars="Date", var_name="Ticker", value_name="Price")
    plot_df["Ticker"] = pd.Categorical(plot_df["Ticker"], categories=ordered_tickers, ordered=True)
    return plot_df.sort_values(["Ticker", "Date"])



def plot_historic_prices_v0(
    start_date: str = "2014-01-01",
    end_date: str | None = None,
    output_path: str = "results/historic_prices_2014_today.png",
) -> str:
    """Descarga precios historicos del universo y genera un plot profesional basico."""
    requested_tickers = get_universe_tickers(include_defensive=True)
    defensive_ticker = get_defensive_ticker()
    market_data = download_market_data(
        start_date=start_date,
        end_date=end_date,
        tickers=requested_tickers,
    )

    prices = _normalize_prices_base_one(market_data["prices"].copy())
    available_tickers = market_data["tickers"]
    universe_by_ticker = {item["ticker"]: item for item in get_universe()}
    plot_df = _build_plot_frame(prices, available_tickers)
    palette = _build_palette(available_tickers, defensive_ticker)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")

    for ticker in available_tickers:
        ticker_df = plot_df[plot_df["Ticker"] == ticker]
        if ticker_df.empty:
            continue

        role = universe_by_ticker.get(ticker, {}).get("role", "risk")
        sns.lineplot(
            data=ticker_df,
            x="Date",
            y="Price",
            ax=ax,
            label=ticker,
            color=palette[ticker],
            linewidth=2.7 if role == "defensive" else 1.7,
            alpha=0.95 if role == "defensive" else 0.88,
            linestyle="--" if role == "defensive" else "-",
        )

    ax.set_title(
        "Precios históricos normalizados del universo de ETFs",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Base 1", fontsize=12)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#374151")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    legend = ax.legend(
        title="Tickers",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        ncol=1,
        fontsize=9,
        title_fontsize=10,
    )
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(2.2)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_file)


if __name__ == "__main__":
    output = plot_historic_prices_v0()
    print("Grafico historico guardado en:", output)
