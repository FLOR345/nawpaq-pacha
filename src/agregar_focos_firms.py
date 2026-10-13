"""
ÑAWPAQ PACHA — Convierte focos de calor REALES de NASA FIRMS a conteos
mensuales por distrito (columna focos_calor del panel).

Cómo obtener los datos (gratis):
 1. Entra a https://firms.modaps.eosdis.nasa.gov/download/
 2. Área: dibuja un rectángulo sobre Puno (aprox. lon -71.2 a -68.8, lat -17.3 a -13.0)
 3. Fuente: MODIS (desde 2000) o VIIRS S-NPP (desde 2012) · formato CSV
 4. Guarda el archivo en data/raw/ (ej. data/raw/firms_puno.csv)

Uso:
    python src/agregar_focos_firms.py data/raw/firms_puno.csv

Salida: data/raw/focos_firms_mensual.csv  (ubigeo, anio, mes, focos_calor)
Luego une esa tabla con tus datos de clima/vegetación para armar panel_mensual.csv.
"""
import json
import sys

import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

import config as C


def agregar(ruta_csv, confianza_min=None):
    focos = pd.read_csv(ruta_csv)
    focos["acq_date"] = pd.to_datetime(focos["acq_date"])
    # Filtra detecciones de baja confianza (MODIS: 0–100; VIIRS: l/n/h)
    if "confidence" in focos:
        if pd.api.types.is_numeric_dtype(focos.confidence):
            focos = focos[focos.confidence >= (confianza_min or 30)]
        else:
            focos = focos[focos.confidence.astype(str).str.lower() != "l"]

    gj = json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    poligonos = [shape(f["geometry"]) for f in gj["features"]]
    ubigeos = [f["properties"]["ubigeo"] for f in gj["features"]]
    arbol = STRtree(poligonos)

    asignado = []
    for lon, lat in zip(focos.longitude, focos.latitude):
        pt = Point(lon, lat)
        idx = next((i for i in arbol.query(pt) if poligonos[i].contains(pt)), None)
        asignado.append(ubigeos[idx] if idx is not None else None)
    focos["ubigeo"] = asignado
    focos = focos.dropna(subset=["ubigeo"])

    mensual = (focos.assign(anio=focos.acq_date.dt.year, mes=focos.acq_date.dt.month)
               .groupby(["ubigeo", "anio", "mes"]).size().rename("focos_calor").reset_index())

    # Completa con ceros los meses sin focos
    anios = range(mensual.anio.min(), mensual.anio.max() + 1)
    completo = pd.MultiIndex.from_product([ubigeos, anios, range(1, 13)], names=["ubigeo", "anio", "mes"])
    mensual = mensual.set_index(["ubigeo", "anio", "mes"]).reindex(completo, fill_value=0).reset_index()

    salida = C.DATA_RAW / "focos_firms_mensual.csv"
    mensual.to_csv(salida, index=False)
    print(f"[FIRMS] {len(focos):,} focos dentro de Puno → {salida}")
    return mensual


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    agregar(sys.argv[1])
