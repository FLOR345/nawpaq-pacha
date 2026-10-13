"""
ÑAWPAQ PACHA — Descarga DATOS REALES y arma las tablas de entrada del modelo.

Fuentes (todas públicas):
  Clima diario (T. máx, lluvia, humedad, viento) → NASA POWER (MERRA-2/GEOS)      sin clave
  Vegetación NDVI 16 días, 250 m                 → MODIS MOD13Q1 (ORNL DAAC)       sin clave
  Cobertura del suelo (pastizal, agrícola)       → MODIS MCD12Q1 (ORNL DAAC)       sin clave
  Altitud y pendiente                            → Open-Meteo Elevation (Copernicus DEM 90 m)
  Focos de calor 2015–año anterior               → NASA FIRMS, archivo anual de Perú (MODIS)
  Focos de calor del año en curso                → NASA FIRMS API  ⚠ requiere FIRMS_MAP_KEY (gratis)
  Focos activos últimos 7 días                   → NASA FIRMS (MODIS + VIIRS), sin clave
  Comunidades y áreas protegidas                 → OpenStreetMap (Overpass)

Clave FIRMS (gratis, llega al correo en 1 minuto):
  https://firms.modaps.eosdis.nasa.gov/api/map_key/
  Guárdala en un archivo  .env  en la raíz del proyecto:   FIRMS_MAP_KEY=tu_clave
  Sin clave, el panel solo llega hasta diciembre del año anterior.

Uso:
    python src/descargar_datos_reales.py
Las descargas se guardan en data/raw/cache/ para no repetirlas.
"""
import calendar
import datetime as dt
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

import config as C

CACHE = C.DATA_RAW / "cache"
ANIO_INICIO = 2015
BBOX = (-71.3, -17.4, -68.7, -12.9)          # oeste, sur, este, norte (región Puno)
AGENTE = "nawpaq-pacha/0.2 (prototipo de investigación, UNA Puno)"
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


# --- Utilidades ----------------------------------------------------------------
def obtener(url, datos=None, intentos=4, espera=5, timeout=180):
    """Descarga una URL con reintentos. Devuelve bytes o None si es 404."""
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, data=datos, headers={"User-Agent": AGENTE, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == intentos - 1:
                raise
        except Exception:
            if i == intentos - 1:
                raise
        time.sleep(espera * (i + 1))


def leer_clave_firms():
    clave = os.environ.get("FIRMS_MAP_KEY")
    env = C.RAIZ / ".env"
    if not clave and env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith("FIRMS_MAP_KEY="):
                clave = linea.split("=", 1)[1].strip().strip('"\'')
    return clave or None


def distritos():
    gj = json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    res = []
    for f in gj["features"]:
        g = shape(f["geometry"])
        pt = g.representative_point()   # siempre cae dentro del distrito
        res.append({"ubigeo": f["properties"]["ubigeo"], "geom": g,
                    "lon": round(pt.x, 4), "lat": round(pt.y, 4)})
    return res


def ultimo_mes_completo(hoy):
    primero = hoy.replace(day=1) - dt.timedelta(days=1)
    return primero.year, primero.month


# --- 1. Clima: NASA POWER ------------------------------------------------------
def clima_distrito(d, hoy):
    archivo = CACHE / f"power_{d['ubigeo']}.json"
    if archivo.exists():
        cache = json.loads(archivo.read_text())
        if cache.get("descargado") == hoy.isoformat():
            return d["ubigeo"], cache["parametros"]
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
           "parameters=T2M_MAX,PRECTOTCORR,RH2M,WS2M&community=AG&format=JSON"
           f"&longitude={d['lon']}&latitude={d['lat']}"
           f"&start={ANIO_INICIO}0101&end={hoy:%Y%m%d}")
    par = json.loads(obtener(url))["properties"]["parameter"]
    archivo.write_text(json.dumps({"descargado": hoy.isoformat(), "parametros": par}))
    return d["ubigeo"], par


def clima_mensual(dists, hoy):
    print(f"[clima] NASA POWER · {len(dists)} distritos…")
    with ThreadPoolExecutor(4) as ex:
        resultados = list(ex.map(lambda d: clima_distrito(d, hoy), dists))
    filas = []
    for ub, par in resultados:
        df = pd.DataFrame(par).replace(-999.0, np.nan)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        g = df.groupby([df.index.year, df.index.month])
        m = pd.DataFrame({
            "temp_max": g.T2M_MAX.mean(),
            "precip_mm": g.PRECTOTCORR.sum(min_count=1),
            "humedad_rel": g.RH2M.mean(),
            "viento_kmh": g.WS2M.mean() * 3.6,       # m/s → km/h
            "dias_validos": g.T2M_MAX.count(),
        })
        m.index.names = ["anio", "mes"]
        m = m.reset_index()
        m["dias_mes"] = [calendar.monthrange(a, b)[1] for a, b in zip(m.anio, m.mes)]
        m = m[m.dias_validos >= 0.9 * m.dias_mes]     # solo meses (casi) completos
        m["ubigeo"] = ub
        filas.append(m.drop(columns=["dias_validos", "dias_mes"]))
    return pd.concat(filas, ignore_index=True)


# --- 2. NDVI: MODIS MOD13Q1 vía ORNL DAAC -------------------------------------
def _ndvi_ventana(d, anio, doys):
    url = ("https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset?band=250m_16_days_NDVI"
           f"&latitude={d['lat']}&longitude={d['lon']}&kmAboveBelow=2&kmLeftRight=2"
           f"&startDate=A{anio}{doys[0]:03d}&endDate=A{anio}{doys[-1]:03d}")
    try:
        js = json.loads(obtener(url, timeout=120))
    except Exception:
        return None                                   # falla: no se guarda en caché
    out = []
    for s in js.get("subset", []):
        v = np.array(s["data"], dtype=float)
        v = v[v > -2000] * 0.0001                     # quita valores de relleno
        if len(v):
            out.append({"fecha": s["calendar_date"], "ndvi": round(float(v.mean()), 4)})
    return out


def ndvi_distrito(d, hoy):
    datos = []
    for anio in range(ANIO_INICIO, hoy.year + 1):
        archivo = CACHE / f"ndvi_{d['ubigeo']}_{anio}.json"
        if archivo.exists() and anio < hoy.year:
            datos += json.loads(archivo.read_text())
            continue
        doys = [x for x in range(1, 366, 16)
                if dt.date(anio, 1, 1) + dt.timedelta(days=x + 15) <= hoy]
        anio_d, completo = [], True
        for i in range(0, len(doys), 9):               # ORNL permite ≤10 fechas por consulta
            v = _ndvi_ventana(d, anio, doys[i:i + 9])
            completo &= v is not None
            anio_d += v or []
        if completo:
            archivo.write_text(json.dumps(anio_d))
        datos += anio_d
    return d["ubigeo"], datos


def ndvi_mensual(dists, hoy):
    print(f"[ndvi] MODIS MOD13Q1 · {len(dists)} distritos (la 1.ª vez tarda ~30 min)…")
    hechos = 0
    filas = []
    with ThreadPoolExecutor(12) as ex:
        for ub, datos in ex.map(lambda d: ndvi_distrito(d, hoy), dists):
            hechos += 1
            if hechos % 20 == 0:
                print(f"   {hechos}/{len(dists)}")
            for x in datos:
                f = dt.date.fromisoformat(x["fecha"])
                filas.append({"ubigeo": ub, "anio": f.year, "mes": f.month, "ndvi": x["ndvi"]})
    return pd.DataFrame(filas).groupby(["ubigeo", "anio", "mes"], as_index=False).ndvi.mean()


# --- 3. Territorio: cobertura MODIS + relieve ---------------------------------
# IGBP (MCD12Q1 LC_Type1): 7 matorral abierto, 9 sabana, 10 pastizal (pajonal/ichu),
# 12 cultivos, 14 mosaico cultivo/vegetación natural
def cobertura_distrito(d):
    archivo = CACHE / f"cobertura_{d['ubigeo']}.json"
    if archivo.exists():
        return d["ubigeo"], json.loads(archivo.read_text())
    url = ("https://modis.ornl.gov/rst/api/v1/MCD12Q1/subset?band=LC_Type1"
           f"&latitude={d['lat']}&longitude={d['lon']}&kmAboveBelow=6&kmLeftRight=6"
           "&startDate=A2024001&endDate=A2024001")
    v = np.array(json.loads(obtener(url))["subset"][0]["data"])
    v = v[(v > 0) & (v < 255)]
    res = {"pct_pastizal": round(100 * np.isin(v, [7, 9, 10]).mean(), 1),
           "pct_agricola": round(100 * (np.isin(v, [12]).mean() + 0.5 * np.isin(v, [14]).mean()), 1)}
    archivo.write_text(json.dumps(res))
    return d["ubigeo"], res


def relieve(dists):
    """Altitud en el punto central y pendiente media hacia 4 puntos a ~1 km."""
    off = 0.009                                       # ≈ 1 km
    pts = []
    for d in dists:
        pts += [(d["lat"], d["lon"]), (d["lat"] + off, d["lon"]), (d["lat"] - off, d["lon"]),
                (d["lat"], d["lon"] + off), (d["lat"], d["lon"] - off)]
    elev = []
    for i in range(0, len(pts), 100):
        bloque = pts[i:i + 100]
        url = ("https://api.open-meteo.com/v1/elevation?latitude=" + ",".join(str(p[0]) for p in bloque)
               + "&longitude=" + ",".join(str(p[1]) for p in bloque))
        elev += json.loads(obtener(url))["elevation"]
    res = {}
    for j, d in enumerate(dists):
        z = elev[5 * j:5 * j + 5]
        dx = 1000 * np.cos(np.radians(d["lat"]))
        pend = np.mean([abs(z[1] - z[0]) / 1000, abs(z[2] - z[0]) / 1000,
                        abs(z[3] - z[0]) / dx, abs(z[4] - z[0]) / dx]) * 100
        res[d["ubigeo"]] = {"altitud_m": round(z[0]), "pendiente_pct": round(float(pend), 1)}
    return res


def territorio(dists):
    print("[territorio] Cobertura MODIS MCD12Q1 + relieve…")
    with ThreadPoolExecutor(6) as ex:
        cob = dict(ex.map(cobertura_distrito, dists))
    rel = relieve(dists)
    return pd.DataFrame([{"ubigeo": d["ubigeo"], **rel[d["ubigeo"]], **cob[d["ubigeo"]]} for d in dists])


# --- 4. Focos de calor: NASA FIRMS --------------------------------------------
def _asignar_distrito(df, dists):
    arbol = STRtree([d["geom"] for d in dists])
    ubs = []
    for lon, lat in zip(df.longitude, df.latitude):
        pt = Point(lon, lat)
        idx = next((i for i in arbol.query(pt) if dists[i]["geom"].contains(pt)), None)
        ubs.append(dists[idx]["ubigeo"] if idx is not None else None)
    return df.assign(ubigeo=ubs).dropna(subset=["ubigeo"])


def _filtrar_confianza(df):
    if pd.api.types.is_numeric_dtype(df.confidence):
        return df[df.confidence >= 30]
    return df[df.confidence.astype(str).str.lower().str[0] != "l"]


def focos_historicos(dists, hoy, clave):
    """Devuelve (focos dentro de Puno, último mes con datos de focos completo)."""
    partes = []
    for anio in range(ANIO_INICIO, hoy.year):
        archivo = CACHE / f"firms_modis_{anio}_puno.csv"
        if not archivo.exists():
            print(f"[focos] FIRMS MODIS {anio} (archivo anual de Perú)…")
            crudo = obtener(f"https://firms.modaps.eosdis.nasa.gov/data/country/modis/{anio}/modis_{anio}_Peru.csv")
            if crudo is None:
                break
            df = pd.read_csv(io.BytesIO(crudo))
            df = df[df.longitude.between(BBOX[0], BBOX[2]) & df.latitude.between(BBOX[1], BBOX[3])]
            df.to_csv(archivo, index=False)
        partes.append(pd.read_csv(archivo))
    ultimo_anio = ANIO_INICIO + len(partes) - 1
    fin = (ultimo_anio, 12)

    if clave and ultimo_anio < hoy.year:
        # Año en curso (y cualquier año faltante) con la API: bloques de 5 días
        inicio = dt.date(ultimo_anio + 1, 1, 1)
        archivo = CACHE / f"firms_api_{inicio.year}_puno.csv"
        previo = pd.read_csv(archivo) if archivo.exists() else pd.DataFrame()
        desde = inicio
        if len(previo):
            desde = max(inicio, dt.date.fromisoformat(previo.acq_date.max()) - dt.timedelta(days=6))
        area = ",".join(map(str, BBOX))
        print(f"[focos] FIRMS API {desde} → {hoy} (MODIS)…")
        api = previo
        while desde <= hoy:
            for fuente in ("MODIS_SP", "MODIS_NRT"):
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{clave}/{fuente}/{area}/5/{desde}"
                crudo = obtener(url, intentos=3, timeout=90)
                if crudo and crudo[:8].lower().startswith(b"latitude"):
                    nuevo = pd.read_csv(io.BytesIO(crudo))
                    if len(nuevo):
                        api = pd.concat([api, nuevo], ignore_index=True)
                elif crudo and b"invalid" in crudo.lower():
                    raise SystemExit("FIRMS_MAP_KEY inválida. Revisa tu archivo .env")
            if len(api):   # guarda el avance en cada bloque
                api = api.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
                api.to_csv(archivo, index=False)
            print(f"   {desde} · {len(api)} focos acumulados")
            desde += dt.timedelta(days=5)
        if len(api):
            partes.append(api)
        fin = ultimo_mes_completo(hoy)
    elif ultimo_anio < hoy.year:
        print("⚠  [focos] Sin FIRMS_MAP_KEY: no hay focos del año en curso. El pronóstico será "
              f"para enero {ultimo_anio + 1}.\n   Consigue tu clave gratis en "
              "https://firms.modaps.eosdis.nasa.gov/api/map_key/ y guárdala en .env")

    focos = pd.concat(partes, ignore_index=True)
    focos = _filtrar_confianza(focos)
    focos = _asignar_distrito(focos, dists)
    focos["acq_date"] = pd.to_datetime(focos.acq_date)
    return focos, fin


def focos_mensuales(focos, ubigeos, fin):
    m = (focos.assign(anio=focos.acq_date.dt.year, mes=focos.acq_date.dt.month)
         .groupby(["ubigeo", "anio", "mes"]).size().rename("focos_calor"))
    meses = [(a, b) for a in range(ANIO_INICIO, fin[0] + 1) for b in range(1, 13) if (a, b) <= fin]
    idx = pd.MultiIndex.from_tuples([(u, a, b) for u in ubigeos for a, b in meses], names=["ubigeo", "anio", "mes"])
    return m.reindex(idx, fill_value=0).reset_index()


def focos_activos(dists):
    """Focos detectados en los últimos 7 días (MODIS + VIIRS), dentro de Puno."""
    print("[focos] Focos activos de los últimos 7 días…")
    fuentes = {
        "MODIS": "modis-c6.1/csv/MODIS_C6_1_South_America_7d.csv",
        "VIIRS S-NPP": "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_America_7d.csv",
        "VIIRS NOAA-20": "noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_America_7d.csv",
    }
    partes = []
    for nombre, ruta in fuentes.items():
        try:
            crudo = obtener("https://firms.modaps.eosdis.nasa.gov/data/active_fire/" + ruta)
        except Exception as e:
            print(f"   {nombre}: no disponible ({e})")
            continue
        if not crudo:
            continue
        df = pd.read_csv(io.BytesIO(crudo))
        df = df[df.longitude.between(BBOX[0], BBOX[2]) & df.latitude.between(BBOX[1], BBOX[3])]
        partes.append(_filtrar_confianza(df).assign(sensor=nombre))
    if not partes:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "frp", "sensor", "ubigeo"])
    df = _asignar_distrito(pd.concat(partes, ignore_index=True), dists)
    return df[["latitude", "longitude", "acq_date", "acq_time", "frp", "sensor", "ubigeo"]]


# --- 5. Capas de contexto: OpenStreetMap --------------------------------------
def capas_osm(dists):
    archivo = C.DATA_RAW / "capas_osm.json"
    if archivo.exists():
        return
    print("[osm] Comunidades y áreas protegidas (OpenStreetMap)…")
    s, w, n, e = BBOX[1], BBOX[0], BBOX[3], BBOX[2]
    consultas = {
        "lugares": f'[out:json][timeout:120];node["place"~"^(city|town|village)$"]({s},{w},{n},{e});out;',
        "protegidas": f'[out:json][timeout:180];relation["boundary"~"^(protected_area|national_park)$"]({s},{w},{n},{e});out geom;',
    }
    res = {}
    for clave, q in consultas.items():
        for servidor in OVERPASS:
            try:
                res[clave] = json.loads(obtener(servidor, urllib.parse.urlencode({"data": q}).encode(),
                                                intentos=2, espera=15, timeout=200))["elements"]
                break
            except Exception as ex:
                print(f"   {clave} @ {servidor}: {ex}")
    lugares = pd.DataFrame([{"nombre": x["tags"].get("name", ""), "tipo": x["tags"]["place"],
                             "lat": x["lat"], "lon": x["lon"]} for x in res.get("lugares", [])])
    if len(lugares):
        lugares = _asignar_distrito(lugares.rename(columns={"lat": "latitude", "lon": "longitude"}), dists)
    protegidas = []
    for r in res.get("protegidas", []):
        anillos = [[[round(p["lon"], 4), round(p["lat"], 4)] for p in m["geometry"]]
                   for m in r.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
        if anillos:
            protegidas.append({"nombre": r["tags"].get("name", "Área protegida"), "lineas": anillos})
    archivo.write_text(json.dumps({
        "lugares": lugares.to_dict("records") if len(lugares) else [],
        "protegidas": protegidas,
    }, ensure_ascii=False), encoding="utf-8")


# --- Todo junto ----------------------------------------------------------------
def descargar():
    CACHE.mkdir(parents=True, exist_ok=True)
    hoy = dt.date.today()
    clave = leer_clave_firms()
    dists = distritos()
    ubigeos = [d["ubigeo"] for d in dists]

    focos, fin_focos = focos_historicos(dists, hoy, clave)
    clima = clima_mensual(dists, hoy)
    fin_clima = (clima.anio * 12 + clima.mes - 1).groupby(clima.ubigeo).max().min()
    fin = min(fin_focos, (int(fin_clima // 12), int(fin_clima % 12 + 1)))
    ndvi = ndvi_mensual(dists, hoy)

    panel = focos_mensuales(focos, ubigeos, fin)
    panel = panel.merge(clima, on=["ubigeo", "anio", "mes"], how="left")
    panel = panel.merge(ndvi, on=["ubigeo", "anio", "mes"], how="left")
    panel = panel.sort_values(["ubigeo", "anio", "mes"])
    # Meses sueltos sin dato (nubes, fallas de descarga): interpolación por distrito
    for col in ["temp_max", "precip_mm", "humedad_rel", "viento_kmh", "ndvi"]:
        panel[col] = panel.groupby("ubigeo")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    panel = panel.round({"temp_max": 1, "precip_mm": 1, "humedad_rel": 1, "viento_kmh": 1, "ndvi": 3})
    panel[C.COLUMNAS_PANEL].to_csv(C.PANEL_MENSUAL, index=False)

    terr = territorio(dists)
    terr.to_csv(C.TERRITORIO, index=False)

    activos = focos_activos(dists)
    activos.to_csv(C.DATA_RAW / "focos_activos_7d.csv", index=False)
    # Puntos de los últimos 12 meses para la capa "Incendios históricos" del mapa
    desde = pd.Timestamp(hoy) - pd.DateOffset(months=12)
    focos[focos.acq_date >= desde][["latitude", "longitude", "acq_date", "frp", "ubigeo"]] \
        .to_csv(C.DATA_RAW / "focos_12m.csv", index=False)
    try:
        capas_osm(dists)
    except Exception as ex:
        print(f"[osm] Capas no disponibles por ahora: {ex}")

    meta = {
        "descargado": hoy.isoformat(),
        "ultimo_mes_datos": f"{fin[0]}-{fin[1]:02d}",
        "firms_anio_en_curso": bool(clave),
        "fuentes": [
            "Clima: NASA POWER (MERRA-2), diario agregado a mensual",
            "NDVI: MODIS MOD13Q1 250 m, ORNL DAAC",
            "Cobertura: MODIS MCD12Q1 2024 (IGBP), ORNL DAAC",
            "Altitud: Copernicus DEM 90 m vía Open-Meteo",
            "Focos de calor: NASA FIRMS (MODIS C6.1, confianza ≥ 30)",
            "Comunidades y áreas protegidas: © OpenStreetMap",
        ],
    }
    (C.DATA_RAW / "fuentes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[datos reales] Panel {ANIO_INICIO}-01 → {fin[0]}-{fin[1]:02d} · {len(panel):,} filas · "
          f"{int(panel.focos_calor.sum()):,} focos · {len(activos)} focos activos (7 días)")
    return meta


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    descargar()
