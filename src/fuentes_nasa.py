"""
ÑAWPAQ PACHA — Fuentes NASA adicionales (opcionales, v0.4).

Agrega al panel, para cada distrito y mes:

  AppEEARS (promedio sobre TODO el polígono del distrito, no un solo punto)
    ndvi_area        NDVI MODIS MOD13A2 1 km (16 días), solo píxeles confiables
    lst_dia_c        Temperatura de la superficie de día, MODIS MOD11A2 1 km (8 días), °C
    quemado_pct      % del distrito con área quemada, MODIS MCD64A1 500 m (mensual)
    humedad_suelo    Humedad del suelo superficial, SMAP SPL3SMP_E 9 km (diaria), m³/m³
  y, por distrito y AÑO:
    pct_pastizal_anual, pct_agricola_anual   MODIS MCD12Q1 500 m (IGBP), año por año

  FIRMS
    focos_viirs      focos VIIRS S-NPP 375 m (confianza nominal o alta) por mes

Requisitos:
  pip install -r requirements-nasa.txt          (rasterio, requests)
  Cuenta gratuita de NASA Earthdata: https://urs.earthdata.nasa.gov
  En el archivo .env:   EARTHDATA_USER=tu_usuario
                        EARTHDATA_PASS=tu_contraseña
  (el usuario, NO el correo; .env no se sube a GitHub)

Uso:
    python main.py --nasa            # descarga estas fuentes y luego ejecuta todo
    python src/fuentes_nasa.py       # solo descarga / actualiza estas fuentes

Las tareas de AppEEARS se procesan en los servidores de la NASA y pueden tardar
horas la primera vez. Si se corta, vuelve a ejecutar el mismo comando: las
tareas ya enviadas se retoman (data/raw/appeears/tareas.json) y lo ya
descargado no se vuelve a pedir.

Salidas:
    data/raw/nasa_mensual.csv   ubigeo, anio, mes, ndvi_area, lst_dia_c, quemado_pct,
                                humedad_suelo, focos_viirs
    data/raw/nasa_anual.csv     ubigeo, anio, pct_pastizal_anual, pct_agricola_anual
"""
import datetime as dt
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

API = "https://appeears.earthdatacloud.nasa.gov/api/"
CARPETA = C.DATA_RAW / "appeears"
TAREAS = CARPETA / "tareas.json"
ANIO_INICIO = 2015
ESPERA_SEG = 60                 # cada cuánto se consulta el estado de las tareas
ESPERA_MAX_MIN = 240            # luego se detiene y pide volver a ejecutar más tarde

# Producto (sin versión: se usa la más reciente disponible en AppEEARS) → capas.
# Para cada capa se da una lista de nombres posibles, por si AppEEARS los cambia.
PRODUCTOS = {
    "ndvi": {"producto": "MOD13A2", "capas": {
        "valor": ["_1_km_16_days_NDVI", "_1km_16_days_NDVI"],
        "fiabilidad": ["_1_km_16_days_pixel_reliability", "_1km_16_days_pixel_reliability"]}},
    "lst": {"producto": "MOD11A2", "capas": {"valor": ["LST_Day_1km"]}},
    "quemado": {"producto": "MCD64A1", "capas": {"valor": ["Burn_Date"]}},
    "cobertura": {"producto": "MCD12Q1", "capas": {"valor": ["LC_Type1"]}},
    "humedad_suelo": {"producto": "SPL3SMP_E", "capas": {
        "valor": ["Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                  "Soil_Moisture_Retrieval_Data_AM_soil_moisture_dca"]}},
}
PASTIZAL_IGBP = [7, 9, 10]      # matorral abierto, sabana, pastizal (igual que v0.3)
AGRICOLA_IGBP = 12              # cultivos (+ mitad del mosaico cultivo/vegetación, 14)
MOSAICO_IGBP = 14
# p. ej. MOD11A2.061_LST_Day_1km_doy2015209000000_aid0001.tif (la hora HHMMSS es opcional)
NOMBRE_TIF = re.compile(r"^(?P<prod>[A-Za-z0-9_]+)\.(?P<ver>\d{3})_(?P<capa>.+)_doy(?P<anio>\d{4})(?P<doy>\d{3})(\d{6})?_aid\d+\.tif$")


# --- Credenciales -----------------------------------------------------------------
def leer_env(nombre):
    valor = os.environ.get(nombre)
    env = C.RAIZ / ".env"
    if not valor and env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith(nombre + "="):
                valor = linea.split("=", 1)[1].strip().strip('"\'')
    return valor or None


# --- Cliente mínimo de la API de AppEEARS -------------------------------------
class AppEEARS:
    def __init__(self, usuario, clave):
        import requests
        self.rq = requests
        r = requests.post(API + "login", auth=(usuario, clave), timeout=120)
        if r.status_code in (400, 401, 403):
            raise SystemExit("AppEEARS rechazó el usuario o la contraseña de Earthdata. "
                             "Revisa EARTHDATA_USER y EARTHDATA_PASS en .env (usuario, no correo).")
        r.raise_for_status()
        self.cabecera = {"Authorization": "Bearer " + r.json()["token"]}

    def get(self, ruta, **kw):
        for i in range(4):
            try:
                r = self.rq.get(API + ruta, headers=self.cabecera, timeout=300, **kw)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r
            except self.rq.ConnectionError:
                if i == 3:
                    raise
            time.sleep(10 * (i + 1))
        r.raise_for_status()

    def productos(self):
        return self.get("product").json()

    def capas(self, producto_version):
        return self.get(f"product/{producto_version}").json()

    def enviar(self, tarea):
        r = self.rq.post(API + "task", json=tarea, headers=self.cabecera, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"AppEEARS no aceptó la tarea {tarea['task_name']}: {r.text[:300]}")
        return r.json()["task_id"]

    def estado(self, task_id):
        return self.get(f"task/{task_id}").json().get("status")

    def archivos(self, task_id):
        return self.get(f"bundle/{task_id}").json()["files"]

    def descargar(self, task_id, file_id, destino, intentos=4):
        """Descarga un archivo del paquete. Devuelve False si falla tras varios intentos
        (p. ej. un 400 pasajero del servidor de descargas); no detiene el programa."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(destino.suffix + ".parcial")
        for i in range(intentos):
            try:
                # cada intento pide un enlace firmado nuevo a AppEEARS
                r = self.rq.get(API + f"bundle/{task_id}/{file_id}", headers=self.cabecera,
                                stream=True, allow_redirects=True, timeout=300)
                if r.status_code == 200:
                    with open(tmp, "wb") as f:
                        for bloque in r.iter_content(1 << 20):
                            f.write(bloque)
                    tmp.replace(destino)
                    return True
            except (self.rq.ConnectionError, self.rq.Timeout, self.rq.exceptions.ChunkedEncodingError):
                pass
            time.sleep(5 * (i + 1))
        tmp.unlink(missing_ok=True)
        return False


def ultima_version(lista_productos, nombre):
    """'MOD13A2' → 'MOD13A2.061' (la versión más alta disponible)."""
    cands = sorted({p["ProductAndVersion"] for p in lista_productos
                    if p["ProductAndVersion"].split(".")[0] == nombre})
    if not cands:
        raise RuntimeError(f"AppEEARS no ofrece el producto {nombre}")
    return cands[-1]


def elegir_capa(disponibles, opciones):
    for o in opciones:
        if o in disponibles:
            return o
    raise RuntimeError(f"Ninguna de las capas {opciones} existe. Disponibles: {sorted(disponibles)[:20]}…")


# --- Área de estudio ------------------------------------------------------------
def geometrias():
    from shapely.geometry import shape
    gj = json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    return {f["properties"]["ubigeo"]: shape(f["geometry"]) for f in gj["features"]}


def area_estudio(geoms):
    """Contorno de toda la región (simplificado), como FeatureCollection para AppEEARS."""
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    region = unary_union(list(geoms.values())).buffer(0.02).simplify(0.01)
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(region)}]}


# --- Tareas: una por fuente y año (los años cerrados se piden una sola vez) -----
def _cargar_tareas():
    return json.loads(TAREAS.read_text(encoding="utf-8")) if TAREAS.exists() else {}


def _guardar_tareas(t):
    CARPETA.mkdir(parents=True, exist_ok=True)
    TAREAS.write_text(json.dumps(t, indent=2), encoding="utf-8")


def enviar_tareas(api, hoy, geo):
    tareas = _cargar_tareas()
    lista = api.productos()
    for clave, info in PRODUCTOS.items():
        try:
            pv = ultima_version(lista, info["producto"])
            meta = api.capas(pv)
            capas = {rol: elegir_capa(meta, ops) for rol, ops in info["capas"].items()}
        except RuntimeError as e:
            print(f"   ⚠ {clave}: {e} · se omite")
            continue
        for anio in range(ANIO_INICIO, hoy.year + 1):
            nombre = f"nawpaq_{clave}_{anio}"
            previa = tareas.get(nombre)
            en_curso = anio == hoy.year
            if previa and (not en_curso or previa.get("pedido") == hoy.strftime("%Y-%m")):
                continue                                   # ya pedida (y vigente)
            fin = min(dt.date(anio, 12, 31), hoy - dt.timedelta(days=1))
            tarea = {"task_type": "area", "task_name": nombre, "params": {
                "dates": [{"startDate": f"01-01-{anio}", "endDate": fin.strftime("%m-%d-%Y")}],
                "layers": [{"product": pv, "layer": c} for c in capas.values()],
                "geo": geo,
                "output": {"format": {"type": "geotiff"}, "projection": "geographic"}}}
            try:
                task_id = api.enviar(tarea)
            except RuntimeError as e:                     # p. ej. años fuera del producto
                print(f"   ⚠ {e}")
                continue
            tareas[nombre] = {"task_id": task_id, "fuente": clave, "anio": anio, "producto": pv,
                              "capas": capas, "pedido": hoy.strftime("%Y-%m"), "descargado": False,
                              "meta": {c: {k: meta[c].get(k) for k in ("ScaleFactor", "AddOffset", "FillValue",
                                                                         "ValidMin", "ValidMax", "Units")}
                                       for c in capas.values()}}
            print(f"   tarea enviada: {nombre} ({pv})")
            _guardar_tareas(tareas)
    return tareas


def esperar_y_descargar(api, tareas):
    pendientes = [n for n, t in tareas.items() if not t["descargado"]]
    inicio = time.time()
    while pendientes:
        for n in list(pendientes):
            t = tareas[n]
            est = api.estado(t["task_id"])
            if est == "done":
                carpeta = CARPETA / n
                fallidos = 0
                capas = set(t["capas"].values())
                for f in api.archivos(t["task_id"]):
                    nombre = Path(f["file_name"]).name
                    if not nombre.endswith(".tif") or fecha_de(nombre)[0] not in capas:
                        continue                  # solo las capas pedidas (no las de calidad extra)
                    destino = carpeta / nombre
                    if not destino.exists() and not api.descargar(t["task_id"], f["file_id"], destino):
                        fallidos += 1
                pendientes.remove(n)
                if fallidos:
                    print(f"   ⚠ {n}: {fallidos} archivos no se pudieron descargar; "
                          "se reintentarán en la próxima ejecución")
                    continue
                t["descargado"] = True
                _guardar_tareas(tareas)
                print(f"   ✓ {n} descargada")
            elif est == "error":
                print(f"   ⚠ {n}: AppEEARS informó un error; se volverá a pedir en la próxima ejecución")
                del tareas[n]
                _guardar_tareas(tareas)
                pendientes.remove(n)
        if not pendientes:
            break
        if (time.time() - inicio) / 60 > ESPERA_MAX_MIN:
            print(f"   {len(pendientes)} tareas siguen en proceso en AppEEARS. Vuelve a ejecutar más tarde "
                  "el mismo comando: se retoman sin pedirlas otra vez.")
            return False
        print(f"   esperando a AppEEARS · {len(pendientes)} tareas pendientes…")
        time.sleep(ESPERA_SEG)
    return True


# --- Promedios por distrito (estadística zonal) -------------------------------
_MASCARAS = {}


def mascaras(geoms, transform, forma):
    """Máscara de cada distrito en una grilla. Si un distrito es más chico que un píxel
    (p. ej. SMAP 9 km), se usan todos los píxeles que toca."""
    from rasterio.features import geometry_mask
    clave = (tuple(transform)[:6], forma)
    if clave not in _MASCARAS:
        m = {}
        for u, g in geoms.items():
            dentro = ~geometry_mask([g], out_shape=forma, transform=transform, all_touched=False)
            if not dentro.any():
                dentro = ~geometry_mask([g], out_shape=forma, transform=transform, all_touched=True)
            m[u] = dentro
        _MASCARAS[clave] = m
    return _MASCARAS[clave]


def leer_tif(ruta, meta=None):
    """Matriz float SIN escalar, con NaN en los valores de relleno; y su grilla."""
    import rasterio
    with rasterio.open(ruta) as src:
        v = src.read(1).astype(float)
        nodata, transform, forma = src.nodata, src.transform, (src.height, src.width)
    meta = meta or {}
    for relleno in (nodata, meta.get("FillValue")):
        if relleno not in (None, ""):
            v[v == float(relleno)] = np.nan
    return v, transform, forma


def escalar(v, meta):
    """Aplica factor de escala y desplazamiento y luego el rango válido, que AppEEARS
    da en unidades ya escaladas (p. ej. LST: 150–1310 K, no el valor crudo 7500–65535)."""
    f = meta.get("ScaleFactor")
    o = meta.get("AddOffset")
    f = float(f) if f not in (None, "", 0) else 1.0
    o = float(o) if o not in (None, "") else 0.0
    v = v * f + o
    lo, hi = meta.get("ValidMin"), meta.get("ValidMax")
    if lo not in (None, ""):
        v[v < float(lo)] = np.nan
    if hi not in (None, ""):
        v[v > float(hi)] = np.nan
    return v


def fecha_de(nombre):
    m = NOMBRE_TIF.match(nombre)
    if not m:
        return None, None
    d = dt.date(int(m["anio"]), 1, 1) + dt.timedelta(days=int(m["doy"]) - 1)
    return m["capa"], d


def media_por_distrito(v, geoms, transform, forma, minimo=1):
    ms = mascaras(geoms, transform, forma)
    res = {}
    for u, m in ms.items():
        x = v[m]
        x = x[~np.isnan(x)]
        res[u] = float(x.mean()) if len(x) >= minimo else np.nan
    return res


def zonal_fuente(tarea, geoms):
    """Recorre los GeoTIFF de una tarea y devuelve filas (ubigeo, fecha, variable, valor)."""
    carpeta = CARPETA / f"nawpaq_{tarea['fuente']}_{tarea['anio']}"
    capa_valor = tarea["capas"]["valor"]
    meta = tarea["meta"][capa_valor]
    filas = []
    tifs = sorted(carpeta.glob("*.tif"))
    por_fecha = {}
    for t in tifs:
        capa, fecha = fecha_de(t.name)
        if fecha:
            por_fecha.setdefault(fecha, {})[capa] = t
    for fecha, capas in sorted(por_fecha.items()):
        if capa_valor not in capas:
            continue
        v, tr, forma = leer_tif(capas[capa_valor], meta)
        fuente = tarea["fuente"]
        if fuente == "ndvi":
            rel_capa = tarea["capas"].get("fiabilidad")
            if rel_capa in capas:
                rel, _, _ = leer_tif(capas[rel_capa])
                v[~np.isin(rel, [0, 1])] = np.nan          # 0 bueno, 1 marginal
            v = escalar(v, meta)
            for u, x in media_por_distrito(v, geoms, tr, forma).items():
                filas.append((u, fecha, "ndvi_area", x))
        elif fuente == "lst":
            v = escalar(v, meta) - 273.15
            for u, x in media_por_distrito(v, geoms, tr, forma).items():
                filas.append((u, fecha, "lst_dia_c", x))
        elif fuente == "humedad_suelo":
            v = escalar(v, meta)
            for u, x in media_por_distrito(v, geoms, tr, forma).items():
                filas.append((u, fecha, "humedad_suelo", x))
        elif fuente == "quemado":
            v = escalar(v, {k: meta.get(k) for k in ("ValidMax",)})      # 0 = no quemado, 1–366 = día
            valido = np.where(np.isnan(v), np.nan, (v >= 0).astype(float))   # −1 sin mapear, −2 agua
            quem = np.where(np.isnan(v) | (v < 0), np.nan, (v > 0).astype(float))
            for u, m in mascaras(geoms, tr, forma).items():
                n = np.nansum(valido[m])
                filas.append((u, fecha, "quemado_pct", float(100 * np.nansum(quem[m]) / n) if n else np.nan))
        elif fuente == "cobertura":
            v = escalar(v, {k: meta.get(k) for k in ("ValidMin", "ValidMax")})
            for u, m in mascaras(geoms, tr, forma).items():
                x = v[m]
                x = x[(~np.isnan(x)) & (x > 0) & (x < 255)]
                if len(x):
                    filas.append((u, fecha, "pct_pastizal_anual", float(100 * np.isin(x, PASTIZAL_IGBP).mean())))
                    filas.append((u, fecha, "pct_agricola_anual",
                                  float(100 * ((x == AGRICOLA_IGBP).mean() + 0.5 * (x == MOSAICO_IGBP).mean()))))
    return filas


def tablas_desde_filas(filas):
    """Filas (ubigeo, fecha, variable, valor) → tabla mensual y tabla anual."""
    if not filas:
        return pd.DataFrame(columns=["ubigeo", "anio", "mes"]), pd.DataFrame(columns=["ubigeo", "anio"])
    df = pd.DataFrame(filas, columns=["ubigeo", "fecha", "variable", "valor"])
    df["fecha"] = pd.to_datetime(df.fecha)
    df["anio"], df["mes"] = df.fecha.dt.year, df.fecha.dt.month
    anual_vars = ["pct_pastizal_anual", "pct_agricola_anual"]
    es_anual = df.variable.isin(anual_vars)
    mensual = (df[~es_anual].groupby(["ubigeo", "anio", "mes", "variable"]).valor.mean()
               .unstack("variable").reset_index())
    anual = (df[es_anual].groupby(["ubigeo", "anio", "variable"]).valor.mean()
             .unstack("variable").reset_index())
    mensual.columns.name = anual.columns.name = None
    return mensual, anual


# --- FIRMS VIIRS S-NPP 375 m (histórico) --------------------------------------
def focos_viirs(hoy):
    """Focos VIIRS S-NPP por distrito y mes, desde 2015.
    Archivo anual de Perú; si no existe, API de FIRMS (requiere FIRMS_MAP_KEY)."""
    import descargar_datos_reales as DR
    dists = DR.distritos()
    clave = DR.leer_clave_firms()
    partes = []
    for anio in range(ANIO_INICIO, hoy.year + 1):
        archivo = DR.CACHE / f"firms_viirs_{anio}_puno.csv"
        if archivo.exists() and anio < hoy.year:
            partes.append(pd.read_csv(archivo))
            continue
        df = None
        if anio < hoy.year:
            crudo = DR.obtener(f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{anio}/"
                               f"viirs-snpp_{anio}_Peru.csv")
            if crudo:
                df = pd.read_csv(io.BytesIO(crudo))
        if df is None and clave:
            df = _viirs_api(DR, clave, anio, hoy)
        if df is None:
            print(f"   ⚠ VIIRS {anio}: no disponible (sin archivo anual ni FIRMS_MAP_KEY)")
            continue
        df = df[df.longitude.between(DR.BBOX[0], DR.BBOX[2]) & df.latitude.between(DR.BBOX[1], DR.BBOX[3])]
        DR.CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(archivo, index=False)
        partes.append(df)
        print(f"   VIIRS {anio}: {len(df):,} focos en el recuadro de Puno")
    if not partes:
        return pd.DataFrame(columns=["ubigeo", "anio", "mes", "focos_viirs"])
    focos = DR._asignar_distrito(DR._filtrar_confianza(pd.concat(partes, ignore_index=True)), dists)
    focos["acq_date"] = pd.to_datetime(focos.acq_date)
    m = (focos.assign(anio=focos.acq_date.dt.year, mes=focos.acq_date.dt.month)
         .groupby(["ubigeo", "anio", "mes"]).size().rename("focos_viirs"))
    # meses sin focos = 0 (solo dentro del periodo cubierto por los archivos descargados)
    fin = DR.ultimo_mes_completo(hoy)
    anios = sorted({int(re.search(r"(\d{4})", p.name).group(1))
                    for p in DR.CACHE.glob("firms_viirs_*_puno.csv")})
    meses = [(a, b) for a in anios for b in range(1, 13) if (a, b) <= fin]
    idx = pd.MultiIndex.from_tuples([(d["ubigeo"], a, b) for d in dists for a, b in meses],
                                    names=["ubigeo", "anio", "mes"])
    return m.reindex(idx, fill_value=0).reset_index()


def _viirs_api(DR, clave, anio, hoy):
    area = ",".join(map(str, DR.BBOX))
    desde, fin = dt.date(anio, 1, 1), min(dt.date(anio, 12, 31), hoy)
    partes = []
    while desde <= fin:
        for fuente in ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"):
            crudo = DR.obtener(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{clave}/{fuente}/{area}/5/{desde}",
                               intentos=3, timeout=90)
            if crudo and crudo[:8].lower().startswith(b"latitude"):
                partes.append(pd.read_csv(io.BytesIO(crudo)))
        desde += dt.timedelta(days=5)
    if not partes:
        return None
    return pd.concat(partes, ignore_index=True).drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time"])


# --- Todo junto ----------------------------------------------------------------
def descargar_nasa(hoy=None):
    hoy = hoy or dt.date.today()
    geoms = geometrias()
    print("[nasa] FIRMS VIIRS S-NPP 375 m…")
    viirs = focos_viirs(hoy)

    usuario, clave = leer_env("EARTHDATA_USER"), leer_env("EARTHDATA_PASS")
    filas = []
    if not (usuario and clave):
        print("⚠  [nasa] Falta EARTHDATA_USER / EARTHDATA_PASS en .env: se omite AppEEARS "
              "(NDVI por área, temperatura de superficie, área quemada, cobertura anual, SMAP).\n"
              "   Cuenta gratis: https://urs.earthdata.nasa.gov")
    else:
        try:
            import rasterio  # noqa: F401
        except ImportError:
            raise SystemExit("Falta rasterio. Instala:  pip install -r requirements-nasa.txt")
        print("[nasa] AppEEARS: enviando tareas (una por fuente y año)…")
        api = AppEEARS(usuario, clave)
        tareas = enviar_tareas(api, hoy, area_estudio(geoms))
        esperar_y_descargar(api, tareas)
        print("[nasa] Promedios por distrito…")
        for nombre, t in sorted(tareas.items()):
            if t["descargado"]:
                filas += zonal_fuente(t, geoms)

    mensual, anual = tablas_desde_filas(filas)
    if len(viirs):
        mensual = mensual.merge(viirs, on=["ubigeo", "anio", "mes"], how="outer") if len(mensual) else viirs
    mensual.to_csv(C.NASA_MENSUAL, index=False)
    anual.to_csv(C.NASA_ANUAL, index=False)
    print(f"[nasa] {C.NASA_MENSUAL.name}: {len(mensual):,} filas · columnas "
          f"{[c for c in mensual.columns if c not in ('ubigeo', 'anio', 'mes')]}")
    print(f"[nasa] {C.NASA_ANUAL.name}: {len(anual):,} filas")
    return mensual, anual


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    descargar_nasa()
