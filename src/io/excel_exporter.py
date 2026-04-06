from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.asset import Universe
from src.domain.trade import Trade


@dataclass
class OperativaExporter:
    """Generates the standardised daily operations Excel file.

    Format matches Operativa_GrupoX.xlsx:
        ID | Cantidad | Precio | CT | Precio Ejecutado

    Where:
        - Compras: Precio Ejecutado = Precio * (1 + CT)
        - Ventas:  Precio Ejecutado = Precio * (1 - CT)
    """

    universe: Universe
    output_dir: str = "outputs/operativa"
    group_name: str = "Grupo4"

    def export(
        self,
        trades: list[Trade],
        date: pd.Timestamp,
        extra_costs: dict[str, float] | None = None,
    ) -> Path | None:
        """Export trades to Excel. Returns path or None if no trades.

        Args:
            extra_costs: CT overrides for tickers not in the universe
                         (e.g. migration liquidations).
        """
        if not trades:
            return None

        rows = []
        for trade in trades:
            if extra_costs and trade.ticker in extra_costs:
                ct = extra_costs[trade.ticker]
            else:
                ct = self.universe.transaction_cost(trade.ticker)
            precio = trade.price
            if trade.shares > 0:
                precio_ejecutado = precio * (1 + ct)
            else:
                precio_ejecutado = precio * (1 - ct)

            rows.append({
                "ID": trade.ticker,
                "Cantidad": trade.shares,
                "Precio": precio,
                "CT": ct,
                "Precio Ejecutado": precio_ejecutado,
            })

        df = pd.DataFrame(rows, columns=["ID", "Cantidad", "Precio", "CT", "Precio Ejecutado"])

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        date_str = date.strftime("%Y-%m-%d")
        filename = f"Operativa_{self.group_name}_{date_str}.xlsx"
        path = out_dir / filename

        df.to_excel(path, sheet_name="Operativa", index=False, engine="openpyxl")
        return path
