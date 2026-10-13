"""
ÑAWPAQ PACHA — Ejecuta todo el sistema de principio a fin.

    python main.py --datos-reales  # descarga/actualiza datos reales y ejecuta todo
    python main.py                 # usa los datos ya descargados en data/raw/
    python main.py --ablacion      # además, compara v0.2 vs v0.3 (outputs/ablacion.json)

Pasos: datos → variables → modelo de IA → pronóstico + alertas → panel web.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no muestra ⚠ ni tildes

import config as C  # noqa: E402
from entrenar_modelo import entrenar  # noqa: E402
from predecir import predecir  # noqa: E402
from preparar_datos import preparar  # noqa: E402

if __name__ == "__main__":
    print("=" * 62)
    print(" ÑAWPAQ PACHA · Riesgo de incendios de vegetación · Puno")
    print("=" * 62)
    if "--datos-reales" in sys.argv or not C.PANEL_MENSUAL.exists():
        from descargar_datos_reales import descargar
        descargar()
        print()
    preparar()
    print()
    entrenar()
    print()
    predecir()
    if "--ablacion" in sys.argv:
        print()
        from ablacion import main as ablacion
        ablacion()
