"""
configurar_tarea_windows.py
============================
Ejecuta este script UNA SOLA VEZ para programar la tarea automatica en Windows.
Despues de ejecutarlo, simulacion_diaria.py se ejecutara automaticamente
cada dia habil a las 21:00h.

Uso:
    python configurar_tarea_windows.py
"""

import subprocess
import sys
from pathlib import Path

# Ruta al script principal — ajustar si es necesario
SCRIPT_PATH  = Path(__file__).parent / "simulacion_diaria.py"
PYTHON_PATH  = sys.executable
TAREA_NOMBRE = "SimulacionDiariaAFI"
HORA_EJECUCION = "21:00"

def configurar_tarea():
    print("\n" + "="*60)
    print("  CONFIGURADOR DE TAREA AUTOMATICA — WINDOWS")
    print("="*60)

    if not SCRIPT_PATH.exists():
        print(f"\n  ERROR: No se encuentra {SCRIPT_PATH}")
        print("  Asegurate de que simulacion_diaria.py esta en la misma carpeta.")
        return

    print(f"\n  Script:  {SCRIPT_PATH}")
    print(f"  Python:  {PYTHON_PATH}")
    print(f"  Hora:    {HORA_EJECUCION} cada dia habil")
    print(f"  Nombre:  {TAREA_NOMBRE}")

    # Comando schtasks para crear la tarea
    # /SC WEEKLY /D MON,TUE,WED,THU,FRI = solo dias laborables
    cmd = [
        "schtasks", "/Create",
        "/TN", TAREA_NOMBRE,
        "/TR", f'"{PYTHON_PATH}" "{SCRIPT_PATH}"',
        "/SC", "WEEKLY",
        "/D", "MON,TUE,WED,THU,FRI",
        "/ST", HORA_EJECUCION,
        "/RL", "HIGHEST",
        "/F"  # Forzar (sobreescribir si ya existe)
    ]

    print(f"\n  Creando tarea en el Programador de Windows...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  OK Tarea creada correctamente.")
            print(f"\n  La simulacion se ejecutara automaticamente a las {HORA_EJECUCION}h")
            print("  de lunes a viernes.")
        else:
            print(f"  ERROR: {result.stderr}")
            print("\n  Intenta ejecutar este script como Administrador:")
            print("  Click derecho en CMD -> 'Ejecutar como administrador'")
            print("  Luego: python configurar_tarea_windows.py")
    except FileNotFoundError:
        print("  ERROR: schtasks no encontrado. Asegurate de estar en Windows.")

    # Verificar que se creo correctamente
    print("\n  Verificando tarea...")
    verify = subprocess.run(
        ["schtasks", "/Query", "/TN", TAREA_NOMBRE],
        capture_output=True, text=True
    )
    if TAREA_NOMBRE in verify.stdout:
        print(f"  OK Tarea '{TAREA_NOMBRE}' verificada en el Programador.")
    else:
        print("  No se pudo verificar la tarea.")

    print("\n" + "="*60)
    print("  INSTRUCCIONES FINALES")
    print("="*60)
    print("""
  1. Abre simulacion_diaria.py y configura:
       ETF_TICKER      = "IUSE.L"       (o el ETF elegido)
       ESTRATEGIA      = "E4"           (o la estrategia elegida)
       GRUPO           = "1"            (vuestro numero de grupo)
       EMAIL_REMITENTE = "tu@outlook.com"
       EMAIL_PASSWORD  = "tu_contrasena_de_aplicacion"
       EMAIL_CC        = "tu@outlook.com"  (copia a vosotros)

  2. Para obtener la contrasena de aplicacion de Outlook:
       a) Ve a https://account.microsoft.com/security
       b) Activa verificacion en dos pasos
       c) Busca "Contrasenas de aplicacion" -> Crear nueva
       d) Copia la contrasena (sin espacios) en EMAIL_PASSWORD

  3. Haz una prueba manual ejecutando:
       python simulacion_diaria.py

  4. Si todo funciona, la tarea correra automaticamente cada dia
     a las 21:00h de lunes a viernes.

  5. Puedes ver/modificar la tarea en:
       Inicio -> Programador de tareas -> SimulacionDiariaAFI

  6. Para eliminar la tarea si fuera necesario:
       schtasks /Delete /TN SimulacionDiariaAFI /F
""")

if __name__ == "__main__":
    configurar_tarea()