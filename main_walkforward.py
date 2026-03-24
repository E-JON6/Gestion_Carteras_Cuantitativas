"""
Main del walkforward version 0.
Recibe: fechas y metrica.
Devuelve: un resumen guardado en results.
"""

from walkforward.walkforward import run_walkforward_v0


if __name__ == "__main__":
    result = run_walkforward_v0(
        start_date="2024-01-01",
        end_date="2024-12-31",
        metric_name="omega",
    )
    print("Walkforward guardado en:", result["output_path"])
    print(result["walkforward"])
