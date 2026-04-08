"""Caps the total weight of each sector."""

from collections import defaultdict
from dataclasses import dataclass, field

from src.domain.trade import Signal


@dataclass
class SectorCapFilter:
    """Caps the sum of weights per sector to ``max_sector_weight``.

    Sector membership is provided as ``sectors``: a ``ticker -> sector``
    dict. Tickers without an entry fall into the empty bucket. Per-sector
    overrides can be passed via ``sector_overrides``.
    """

    sectors: dict[str, str]
    max_sector_weight: float = 0.50
    sector_overrides: dict[str, float] = field(default_factory=dict)
    max_iterations: int = 10

    def filter(self, signals: list[Signal]) -> list[Signal]:
        if not signals:
            return signals

        weights = {s.ticker: s.target_weight for s in signals}

        for _ in range(self.max_iterations):
            buckets: dict[str, list[str]] = defaultdict(list)
            for t in weights:
                buckets[self.sectors.get(t, "")].append(t)

            changed = False
            for sec, tickers in buckets.items():
                cap = self.sector_overrides.get(sec, self.max_sector_weight)
                total = sum(weights[t] for t in tickers)
                if total > cap + 1e-9:
                    scale = cap / total
                    for t in tickers:
                        weights[t] *= scale
                    changed = True
            if not changed:
                break

        return [
            Signal(
                date=s.date,
                ticker=s.ticker,
                target_weight=weights[s.ticker],
                reason=s.reason,
                info=s.info,
            )
            for s in signals
        ]
