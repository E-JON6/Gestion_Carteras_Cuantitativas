"""Weight and signal normalisation."""

from src.domain.portfolio import Portfolio
from src.domain.trade import Signal


def normalise_weights(
    weights: dict[str, float],
    portfolio: Portfolio,
    max_fraction: float = 1.0,
) -> dict[str, float]:
    """Normalise weights preserving relative proportions.

    1. If ``allow_short`` is False, negative weights are zeroed.
    2. If the total exceeds ``max_fraction * max_leverage``, all
       weights are scaled down proportionally.

    Args:
        weights: ticker -> raw weight.
        portfolio: For ``allow_short`` and ``max_leverage``.
        max_fraction: Fraction of max_leverage to use (0, 1].
    """
    if not 0 < max_fraction <= 1:
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")

    if not portfolio.allow_short:
        weights = {t: max(w, 0.0) for t, w in weights.items()}

    cap = max_fraction * portfolio.max_leverage
    total = sum(abs(w) for w in weights.values())
    if total > cap and total > 0:
        weights = {t: w * cap / total for t, w in weights.items()}

    return weights


def normalise_signals(
    signals: list[Signal],
    portfolio: Portfolio,
    max_fraction: float = 1.0,
) -> list[Signal]:
    """Normalise signal weights preserving relative proportions."""
    weights = normalise_weights(
        {s.ticker: s.target_weight for s in signals},
        portfolio,
        max_fraction,
    )
    return [
        Signal(date=s.date, ticker=s.ticker, target_weight=weights[s.ticker], reason=s.reason, info=s.info)
        for s in signals
    ]
