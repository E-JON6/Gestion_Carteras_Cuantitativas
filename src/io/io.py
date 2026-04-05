from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

import yaml

from src.domain.asset import Asset, Universe


@dataclass(frozen=True)
class IO:

    INPUTS: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent / "inputs"
    UNIVERSES: ClassVar[Path] = INPUTS / "universes"

    def load_universe(self, name: str) -> Universe:
        """Load a named universe from inputs/universes/{name}.yaml."""
        path = self.UNIVERSES / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Universe '{name}' not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        assets = tuple(
            Asset(
                ticker=ticker,
                sector=info.get("sector", ""),
                transaction_cost=info.get("cost"),
            )
            for ticker, info in data.items()
        )

        return Universe(assets=assets)

    def list_universes(self) -> list[str]:
        """List all universe names (filenames without .yaml)."""
        return sorted(
            p.stem for p in self.UNIVERSES.glob("*.yaml")
        )
