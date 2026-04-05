"""Fill remaining allocation to a defensive ticker."""

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal


def fill_defensive(
    signals: list[Signal],
    risky_tickers: list[str],
    defensive_ticker: str,
    prices: PriceSnapshot,
    portfolio: Portfolio,
) -> list[Signal]:
    """Append a defensive ticker signal so the portfolio is fully invested.

    Computes the risky total from signal weights (for tickers with
    signals) and current weights (for tickers without signals).
    The defensive ticker absorbs ``max_leverage - risky_total``.
    """
    updated = {s.ticker: s.target_weight for s in signals}
    current_weights = portfolio.positions_weights(prices)

    risky_total = sum(
        updated.get(t, current_weights.get(t, 0.0))
        for t in risky_tickers
    )

    target = portfolio.max_leverage - risky_total
    if not portfolio.allow_short:
        target = max(target, 0.0)

    signals.append(Signal(
        date=prices.date,
        ticker=defensive_ticker,
        target_weight=target,
        reason="defensive",
    ))
    return signals
