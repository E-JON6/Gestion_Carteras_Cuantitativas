"""Parameter space generators for optimization."""

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator
import random


@dataclass
class ParameterGrid:
    """Generates all combinations from a parameter space.

    Usage:
        grid = ParameterGrid({"gamma": [0.3, 0.5], "window": [63, 126]})
        for params in grid:
            # params = {"gamma": 0.3, "window": 63}, etc.
    """

    param_space: dict[str, list]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def __len__(self) -> int:
        total = 1
        for values in self.param_space.values():
            total *= len(values)
        return total


@dataclass
class RandomParameterGrid:
    """Samples N random combinations from a parameter space.

    Usage:
        grid = RandomParameterGrid(
            {"gamma": [0.3, 0.5, 0.7], "window": [42, 84, 126]},
            n_samples=10,
            seed=42,
        )
    """

    param_space: dict[str, list]
    n_samples: int = 100
    seed: int | None = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        full_size = 1
        for v in self.param_space.values():
            full_size *= len(v)
        self._actual_samples = min(self.n_samples, full_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        full_size = 1
        for v in values:
            full_size *= len(v)

        if self._actual_samples >= full_size:
            for combo in product(*values):
                yield dict(zip(keys, combo))
            return

        seen = set()
        while len(seen) < self._actual_samples:
            combo = tuple(self._rng.choice(v) for v in values)
            if combo not in seen:
                seen.add(combo)
                yield dict(zip(keys, combo))

    def __len__(self) -> int:
        return self._actual_samples
