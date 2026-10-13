"""
ÑAWPAQ PACHA — Preparación de variables para el modelo.

Idea clave (anticipación): para estimar el riesgo del mes t, el modelo SOLO
usa información disponible hasta el mes t-1. Así el sistema realmente se
adelanta y no "hace trampa" mirando el mismo mes que quiere predecir.

Objetivo a predecir: hubo_incendio = 1 si en el mes t se registró al menos
un foco de calor/incendio en el distrito.
"""
import json

import numpy as np
import pandas as pd
from shapely.geometry import shape

import config as C
import variables_diarias as VD


def cargar_datos():
    panel = pd.read_csv(C.PANEL_MENSUAL, dtype={"ubigeo": str})
    terr = pd.read_csv(C.TERRITORIO, dtype={"ubigeo": str})
    faltan = set(C.COLUMNAS_PANEL) - set(panel.columns)
    if faltan:
        raise ValueError(f"Al panel mensual le faltan columnas: {faltan}")
    faltan = set(C.COLUMNAS_TERRITORIO) - set(terr.columns)
    if faltan:
        raise ValueError(f"A la tabla de territorio le faltan columnas: {faltan}")
    for df in (panel, terr):
        df["ubigeo"] = df["ubigeo"].str.zfill(6)
    return panel, terr


def agregar_mes_siguiente(panel):
    """Agrega una fila vacía (sin datos aún) del mes siguiente al último
    registrado, para poder pronosticar ese mes."""
    ult = panel.sort_values(["anio", "mes"]).iloc[-1]
    anio, mes = int(ult.anio), int(ult.mes) + 1
    if mes == 13:
        anio, mes = anio + 1, 1
    futuro = pd.DataFrame({"ubigeo": panel.ubigeo.unique(), "anio": anio, "mes": mes})
    return pd.concat([panel, futuro], ignore_index=True), (anio, mes)


def vecinos():
    """Distritos que comparten límite con cada distrito (según el GeoJSON)."""
    gj = json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    geo = {f["properties"]["ubigeo"]: shape(f["geometry"]).buffer(0.001) for f in gj["features"]}
    return {u: [v for v in geo if v != u and geo[u].intersects(geo[v])] for u in geo}


def variables_vecinos(df):
    """Focos en los distritos vecinos el mes anterior y en los 12 meses previos."""
    tabla = df.pivot_table(index=["anio", "mes"], columns="ubigeo", values="focos_calor",
                           aggfunc="sum", dropna=False).sort_index()
    vec = vecinos()
    suma = pd.DataFrame({u: tabla[[v for v in vec.get(u, []) if v in tabla]].sum(axis=1, min_count=1)
                         for u in tabla.columns})
    salida = []
    for nombre, t in (("vec_focos_lag1", suma.shift(1)),
                      ("vec_focos_12m", suma.shift(1).rolling(12, min_periods=6).sum())):
        larga = t.stack(future_stack=True).rename(nombre)
        larga.index.names = ["anio", "mes", "ubigeo"]
        salida.append(larga)
    return pd.concat(salida, axis=1).reset_index()


def construir_variables(panel, terr):
    df = panel.sort_values(["ubigeo", "anio", "mes"]).reset_index(drop=True)
    diarias = VD.calcular(df.ubigeo.unique())
    df = df.merge(diarias, on=["ubigeo", "anio", "mes"], how="left")
    df = df.merge(variables_vecinos(df), on=["ubigeo", "anio", "mes"], how="left")
    df = df.sort_values(["ubigeo", "anio", "mes"]).reset_index(drop=True)
    g = df.groupby("ubigeo", group_keys=False)

    # --- Indicadores de días peligrosos del mes anterior (clima diario) ---
    for col in VD.COLUMNAS:
        df[f"{col}_lag1"] = g[col].shift(1)

    # --- Rezagos (mes anterior) ---
    df["temp_max_lag1"] = g.temp_max.shift(1)
    df["precip_lag1"] = g.precip_mm.shift(1)
    df["humedad_lag1"] = g.humedad_rel.shift(1)
    df["viento_lag1"] = g.viento_kmh.shift(1)
    df["ndvi_lag1"] = g.ndvi.shift(1)
    df["focos_lag1"] = g.focos_calor.shift(1)
    df["focos_lag2"] = g.focos_calor.shift(2)
    df["focos_mismo_mes_ant"] = g.focos_calor.shift(12)     # mismo mes del año anterior

    # --- Acumulados ---
    df["precip_3m"] = g.precip_mm.transform(lambda s: s.shift(1).rolling(3).sum())
    df["focos_12m"] = g.focos_calor.transform(lambda s: s.shift(1).rolling(12, min_periods=6).sum())

    # --- Anomalías: comparadas con años ANTERIORES del mismo distrito y mes ---
    df["mes_ref"] = (df.mes - 2) % 12 + 1          # mes calendario del rezago
    for col, nueva in (("precip_3m", "anom_precip_3m"), ("ndvi_lag1", "anom_ndvi_lag1")):
        normal = (df.groupby(["ubigeo", "mes_ref"])[col]
                  .transform(lambda s: s.shift(1).expanding(min_periods=2).mean()))
        df[nueva] = df[col] - normal

    # --- Historial: frecuencia de incendios en este mes en años previos ---
    df["hubo_incendio"] = (df.focos_calor > 0).astype(float)
    df.loc[df.focos_calor.isna(), "hubo_incendio"] = np.nan
    df["frec_hist_mes"] = (df.groupby(["ubigeo", "mes"]).hubo_incendio
                           .transform(lambda s: s.shift(1).expanding(min_periods=2).mean()))

    # --- Estacionalidad ---
    df["mes_sin"] = np.sin(2 * np.pi * df.mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df.mes / 12)

    df = df.merge(terr, on="ubigeo", how="left")
    return df.drop(columns=["mes_ref", *VD.COLUMNAS])


def preparar():
    panel, terr = cargar_datos()
    panel, mes_pronostico = agregar_mes_siguiente(panel)
    df = construir_variables(panel, terr)
    df["es_pronostico"] = (df.anio == mes_pronostico[0]) & (df.mes == mes_pronostico[1])
    # Quitamos filas iniciales sin historial suficiente
    df = df.dropna(subset=C.VARIABLES_MODELO).reset_index(drop=True)
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATASET_MODELO, index=False)
    print(f"[preparación] {len(df):,} filas con variables · mes a pronosticar: "
          f"{mes_pronostico[1]:02d}/{mes_pronostico[0]}")
    return df, mes_pronostico


if __name__ == "__main__":
    preparar()
