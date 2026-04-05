"""Registro append-only de ejecuciones diarias (trazabilidad a 6+ meses)."""

from __future__ import annotations

import csv
from pathlib import Path


EJECUCION_FIELDS = [
    "fecha_datos",
    "nav_previo_eur",
    "path_operaciones",
    "path_posiciones_post_rebalanceo",
    "path_snapshot_ultimo",
    "dn_rebalance",
    "ordenes_filas",
    "skipped_due_to_dn",
]


def append_ejecucion_row(path: str | Path, row: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EJECUCION_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        out = {k: row.get(k, "") for k in EJECUCION_FIELDS}
        w.writerow(out)
    return path
