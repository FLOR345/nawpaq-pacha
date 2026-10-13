"""
ÑAWPAQ PACHA — Pronóstico de riesgo por distrito, alertas y exportación al panel web.

Genera:
  outputs/pronostico_AAAA_MM.csv   → riesgo de cada distrito (para informes)
  outputs/alertas_AAAA_MM.csv      → solo distritos con alerta (Alto / Muy alto)
  dashboard/datos_riesgo.js        → datos que lee el panel web (mapa)

La alerta final combina tres cosas:
  1. Riesgo del mes estimado por el modelo de IA.
  2. Focos activos detectados por satélite (NASA FIRMS, últimas 48 h).
  3. Pronóstico del tiempo de 7 días (Open-Meteo) → regla PELIGRO_TIEMPO.
"""
import datetime as dt
import json
import urllib.request

import joblib
import pandas as pd
from shapely.geometry import shape

import alertas as A
import config as C
from entrenar_modelo import a_nivel, probabilidad

MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def leer_csv(nombre, **kw):
    ruta = C.DATA_RAW / nombre
    return pd.read_csv(ruta, dtype={"ubigeo": str}, **kw) if ruta.exists() else None


def pronostico_tiempo(puntos):
    """Pronóstico diario de 7 días para cada distrito (Open-Meteo, sin clave)."""
    salida = {}
    for i in range(0, len(puntos), 50):
        bloque = puntos[i:i + 50]
        url = ("https://api.open-meteo.com/v1/forecast?timezone=America%2FLima&forecast_days=7"
               "&daily=temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum"
               "&latitude=" + ",".join(str(p[1]) for p in bloque)
               + "&longitude=" + ",".join(str(p[2]) for p in bloque))
        with urllib.request.urlopen(url, timeout=60) as r:
            js = json.loads(r.read())
        for (ub, _, _), d in zip(bloque, js if isinstance(js, list) else [js]):
            dd = d["daily"]
            salida[ub] = [{"fecha": f, "tmax": t, "hum_min": h, "viento_max": v, "lluvia": ll,
                           "peligro": A.peligro_dia(h, v, ll, C.PELIGRO_TIEMPO)}
                          for f, t, h, v, ll in zip(dd["time"], dd["temperature_2m_max"],
                                                    dd["relative_humidity_2m_min"],
                                                    dd["wind_speed_10m_max"], dd["precipitation_sum"])]
    return salida


def predecir():
    paquete = joblib.load(C.MODELO_ARCHIVO)
    normales = paquete["normales"]
    df = pd.read_csv(C.DATASET_MODELO, dtype={"ubigeo": str})
    df["prob"] = probabilidad(paquete, df)
    # En el historial del panel se muestran las predicciones FUERA DE MUESTRA de la
    # evaluación año por año (lo que el modelo habría dicho sin conocer ese año).
    ruta_ev = C.SALIDAS / "predicciones_evaluacion.csv"
    if ruta_ev.exists():
        ev = pd.read_csv(ruta_ev, dtype={"ubigeo": str})[["ubigeo", "anio", "mes", "prob"]]
        df = df.merge(ev.rename(columns={"prob": "prob_ev"}), on=["ubigeo", "anio", "mes"], how="left")
        df["prob_hist"] = df.prob_ev.fillna(df.prob)
    else:
        df["prob_hist"] = df.prob
    df["nivel"] = a_nivel(df.prob)

    fut = df[df.es_pronostico].copy()
    anio, mes = int(fut.anio.iloc[0]), int(fut.mes.iloc[0])
    norm = normales.set_index(["ubigeo", "mes"])

    gj = json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    info = {f["properties"]["ubigeo"]: f["properties"] for f in gj["features"]}
    centro = {}
    for f in gj["features"]:
        pt = shape(f["geometry"]).representative_point()
        centro[f["properties"]["ubigeo"]] = (round(pt.y, 4), round(pt.x, 4))

    # --- Datos de corto plazo: focos activos y pronóstico del tiempo ---
    activos = leer_csv("focos_activos_7d.csv")
    focos_48h = {}
    if activos is not None and len(activos):
        hora = pd.to_datetime(activos.acq_date + " " + activos.acq_time.astype(str).str.zfill(4), format="%Y-%m-%d %H%M")
        recientes = activos[hora >= pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=C.HORAS_FOCO_ACTIVO)]
        focos_48h = recientes.ubigeo.value_counts().to_dict()
    tiempo = {}
    try:
        tiempo = pronostico_tiempo([(u, *centro[u]) for u in fut.ubigeo if u in centro])
    except Exception as ex:
        print(f"[tiempo] Pronóstico no disponible ahora ({ex}); se usará solo el modelo.")
    terr = pd.read_csv(C.TERRITORIO, dtype={"ubigeo": str}).set_index("ubigeo")

    registros = []
    for _, f in fut.iterrows():
        normal = norm.loc[(f.ubigeo, mes)] if (f.ubigeo, mes) in norm.index else None
        p = info.get(f.ubigeo, {})
        dias = tiempo.get(f.ubigeo, [])
        final, motivos = A.nivel_final(f.nivel, [d["peligro"] for d in dias[:3]],
                                       int(focos_48h.get(f.ubigeo, 0)), C.NIVELES)
        t = terr.loc[f.ubigeo] if f.ubigeo in terr.index else None
        registros.append({
            "ubigeo": f.ubigeo, "distrito": p.get("distrito", f.ubigeo), "provincia": p.get("provincia", ""),
            "probabilidad": round(float(f.prob), 3), "nivel": f.nivel,
            "nivel_final": final, "motivos_alerta": motivos,
            "focos_48h": int(focos_48h.get(f.ubigeo, 0)),
            "factores": A.explicar(f, normal),
            "condiciones": {"temp_max": round(f.temp_max_lag1, 1), "lluvia_3m": round(f.precip_3m),
                            "humedad": round(f.humedad_lag1), "viento": round(f.viento_lag1),
                            "ndvi": round(f.ndvi_lag1, 2)},
            "territorio": {k: (round(float(t[k]), 1) if t is not None else None)
                           for k in ["altitud_m", "pendiente_pct", "pct_pastizal", "pct_agricola"]},
            "centro": centro.get(f.ubigeo),
            "tiempo": dias,
        })
    orden = {n: i for i, n in enumerate(C.NIVELES)}
    pron = pd.DataFrame(registros)
    pron = pron.assign(_o=pron.nivel_final.map(orden)).sort_values(["_o", "probabilidad"], ascending=False).drop(columns="_o")

    # --- Archivos CSV ---
    C.SALIDAS.mkdir(exist_ok=True)
    tabla = (pron[["ubigeo", "distrito", "provincia", "probabilidad", "nivel", "nivel_final", "focos_48h", "factores", "motivos_alerta"]]
             .rename(columns={"nivel": "nivel_modelo"})
             .assign(factores=lambda d: d.factores.str.join(" | "),
                     motivos_alerta=lambda d: d.motivos_alerta.str.join(" | ")))
    tabla.to_csv(C.SALIDAS / f"pronostico_{anio}_{mes:02d}.csv", index=False, encoding="utf-8-sig")
    umbral = C.NIVELES.index(C.NIVEL_ALERTA)
    con_alerta = tabla[tabla.nivel_final.map(C.NIVELES.index) >= umbral]
    con_alerta.to_csv(C.SALIDAS / f"alertas_{anio}_{mes:02d}.csv", index=False, encoding="utf-8-sig")

    # --- Historial reciente (IA vs. lo que pasó) para el panel ---
    hist = df[(~df.es_pronostico) & (df.anio >= anio - 2)]
    serie = {u: {"prob": g.prob_hist.round(3).tolist(), "incendio": g.hubo_incendio.astype(int).tolist()}
             for u, g in hist.sort_values(["anio", "mes"]).groupby("ubigeo")}
    etiquetas = [f"{MESES[m][:3]} {a % 100:02d}" for a, m in
                 hist[["anio", "mes"]].drop_duplicates().sort_values(["anio", "mes"]).itertuples(index=False)]

    # --- Capas del mapa ---
    def puntos(tabla_focos, extra):
        if tabla_focos is None:
            return []
        return [[round(r.latitude, 4), round(r.longitude, 4), *[getattr(r, c) for c in extra]]
                for r in tabla_focos.itertuples(index=False)]
    focos_12m = leer_csv("focos_12m.csv")
    osm = json.loads((C.DATA_RAW / "capas_osm.json").read_text(encoding="utf-8")) \
        if (C.DATA_RAW / "capas_osm.json").exists() else {"lugares": [], "protegidas": []}
    fuentes = json.loads((C.DATA_RAW / "fuentes.json").read_text(encoding="utf-8")) \
        if (C.DATA_RAW / "fuentes.json").exists() else {}

    evaluacion = json.loads((C.SALIDAS / "evaluacion_modelo.json").read_text(encoding="utf-8"))
    datos = {
        "generado": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mes_pronostico": f"{MESES[mes]} {anio}",
        "fuentes": fuentes,
        "quechua_validado": A.QUECHUA_VALIDADO,
        "niveles": C.NIVELES, "colores": C.COLORES_NIVEL, "umbrales": C.UMBRALES_NIVEL,
        "nivel_alerta": C.NIVEL_ALERTA, "peligro_tiempo": C.PELIGRO_TIEMPO,
        "nivel_qu": A.NIVEL_QU, "recomendaciones": A.RECOMENDACIONES,
        "titulo": A.TITULO, "aviso": A.AVISO,
        "distritos": pron.to_dict("records"),
        "historial": {"etiquetas": etiquetas, "series": serie},
        "evaluacion": evaluacion,
        "capas": {
            "focos_activos": puntos(activos, ["acq_date", "acq_time", "sensor", "ubigeo"]),
            "focos_12m": puntos(focos_12m, ["acq_date"]),
            "lugares": [[x["latitude"], x["longitude"], x["nombre"], x["tipo"], x["ubigeo"]] for x in osm["lugares"]],
            "protegidas": osm["protegidas"],
        },
        "geojson": gj,
    }
    C.DASHBOARD.mkdir(exist_ok=True)
    (C.DASHBOARD / "datos_riesgo.js").write_text(
        "// Generado automáticamente por src/predecir.py — no editar a mano\n"
        "window.NAWPAQ = " + json.dumps(datos, ensure_ascii=False, default=str) + ";\n", encoding="utf-8")

    conteo = pron.nivel_final.value_counts().reindex(C.NIVELES[::-1], fill_value=0)
    print(f"[pronóstico] {MESES[mes]} {anio} (alerta final): " + " · ".join(f"{k}: {v}" for k, v in conteo.items()))
    print(f"[pronóstico] Distritos con alerta: {len(con_alerta)} → outputs/alertas_{anio}_{mes:02d}.csv")
    print(f"[pronóstico] Focos activos (48 h): {sum(focos_48h.values())} · pronóstico del tiempo: "
          f"{'sí' if tiempo else 'no'}")
    print("[panel] Abre dashboard/index.html en tu navegador")
    return pron


if __name__ == "__main__":
    predecir()
