"""Wrapper de compatibilidad para el nuevo walk-forward BL-Omega."""

from __future__ import annotations

from pathlib import Path

from walk_forward_bl import run_walk_forward


def run_walkforward_v0(*args, **kwargs):
    _ = args
    result = run_walk_forward(**{k: v for k, v in kwargs.items() if k in {"save_results"}})
    summary = result["summary"]
    return {
        "walkforward": summary,
        "summary": summary,
        "output_path": str(Path(result["output_dir"]) / "resultados_wf.csv"),
        "excel_output_path": None,
        "ratio_oos_is": result["ratio_oos_is"],
    }


if __name__ == "__main__":
    run_walk_forward()
