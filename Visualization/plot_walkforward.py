"""
Plot walkforward version 0.
Recibe: el CSV del walkforward.
Devuelve: un grafico guardado en results.
"""

from Visualization.plots import plot_walkforward_v0


if __name__ == "__main__":
    walkforward_plot = plot_walkforward_v0()
    print("Grafico walkforward:", walkforward_plot)
