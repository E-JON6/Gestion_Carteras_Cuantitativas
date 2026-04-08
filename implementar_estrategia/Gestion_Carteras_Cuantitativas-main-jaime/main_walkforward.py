"""
Entry point para ejecutar el walkforward completo.

Uso:
    python main_walkforward.py
"""

from walkforward.walkforward import run_walkforward


if __name__ == "__main__":
    result = run_walkforward(
        start_date="2020-01-01",
        end_date="2025-12-31",
    )
    print("\nWalkforward completado.")
    print(f"Resultados en: {result['output_dir']}")
