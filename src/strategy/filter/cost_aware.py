from dataclasses import dataclass

from src.domain.asset import PriceSnapshot
from src.domain.portfolio import Portfolio
from src.domain.trade import Signal


@dataclass
class CostAwareFilter:
    """Scales down buy signals so they fit within available cash.

    Simulates the broker's cash check: sell proceeds (after tx costs)
    plus current cash must cover all buys (including tx costs).  If
    buys exceed the budget, they are scaled down proportionally so
    that every signal that reaches the broker will be accepted.

    Args:
        transaction_costs: ticker -> proportional cost (same as broker).
    """

    transaction_costs: dict[str, float]
    safety_margin: float = 0.02

    def filter(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        prices: PriceSnapshot,
    ) -> list[Signal]:
        if not signals:
            return signals

        current_weights = portfolio.positions_weights(prices)
        total_value = portfolio.total_value(prices)

        sells = []
        buys = []
        for s in signals:
            current_w = current_weights.get(s.ticker, 0.0)
            delta_w = s.target_weight - current_w
            if delta_w < -1e-9:
                sells.append(s)
            elif delta_w > 1e-9:
                buys.append(s)
            # unchanged signals pass through

        if not buys:
            return signals

        # Cash from sells (after tx costs)
        sell_cash = 0.0
        for s in sells:
            current_w = current_weights.get(s.ticker, 0.0)
            sell_value = abs(s.target_weight - current_w) * total_value
            tx = self._tx_rate(s.ticker)
            sell_cash += sell_value * (1 - tx)

        available = portfolio.cash + sell_cash
        if portfolio.max_leverage > 1.0:
            available += total_value * (portfolio.max_leverage - 1.0)
        available *= (1 - self.safety_margin)

        # Total cost of buys (including tx costs)
        buy_costs = {}
        total_buy_cost = 0.0
        for s in buys:
            current_w = current_weights.get(s.ticker, 0.0)
            buy_value = (s.target_weight - current_w) * total_value
            tx = self._tx_rate(s.ticker)
            cost = buy_value * (1 + tx)
            buy_costs[s.ticker] = cost
            total_buy_cost += cost

        if total_buy_cost <= available:
            return signals

        # Scale down buys proportionally
        scale = available / total_buy_cost if total_buy_cost > 0 else 0.0

        scaled = []
        for s in signals:
            if s.ticker in buy_costs:
                current_w = current_weights.get(s.ticker, 0.0)
                delta_w = s.target_weight - current_w
                new_target = current_w + delta_w * scale
                scaled.append(Signal(
                    date=s.date,
                    ticker=s.ticker,
                    target_weight=new_target,
                    reason=s.reason,
                    info=s.info,
                ))
            else:
                scaled.append(s)

        return scaled

    def _tx_rate(self, ticker: str) -> float:
        return self.transaction_costs.get(ticker, 0.0)
