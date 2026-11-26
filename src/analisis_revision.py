"""
ÑAWPAQ PACHA — Análisis adicionales pedidos en la revisión del artículo (sin internet).

1. Regla de corto plazo (48 h): ¿cuánto cambia la alerta si a la predicción mensual
   se le suma la regla "focos activos en las últimas 48 h → al menos nivel Alto"?
   Se reconstruye con el archivo histórico de FIRMS (MODIS) para 2020–2025.
   La parte meteorológica de la regla (pronóstico a 3 días) no puede evaluarse:
   no hay archivo de pronósticos pasados para todo el periodo.
2. Focos MODIS de confianza alta (≥ 80) por distrito y mes, para la etiqueta
   alternativa "modis_alta" (análisis de sensibilidad).

Uso:  python src/analisis_revision.py      (después de python main.py)
Salida: outputs/regla_corto_plazo.json
"""
import json

import numpy as np
import pandas as pd

import config as C
import descargar_datos_reales as DR


def focos_modis_archivados():
    """Focos MODIS del archivo local (data/raw/cache), asignados a distritos, confianza ≥ 30."""
    partes = [pd.read_csv(p) for p in sorted(DR.CACHE.glob("firms_modis_*_puno.csv"))]
    partes += [pd.read_csv(p) for p in sorted(DR.CACHE.glob("firms_api_*_puno.csv"))]
    if not partes:
        return pd.DataFrame(columns=["ubigeo", "acq_date", "confidence"])
    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    df = DR._asignar_distrito(DR._filtrar_confianza(df), DR.distritos())
    df["acq_date"] = pd.to_datetime(df.acq_date)
    return df


def focos_modis_alta(focos=None, umbral=80):
    """Conteo mensual por distrito de focos MODIS con confianza ≥ 80 (clase 'alta')."""
    focos = focos_modis_archivados() if focos is None else focos
    conf = pd.to_numeric(focos.confidence, errors="coerce")
    alta = focos[conf >= umbral]
    return (alta.assign(anio=alta.acq_date.dt.year, mes=alta.acq_date.dt.month)
            .groupby(["ubigeo", "anio", "mes"]).size().rename("focos_modis_alta").reset_index())


def _resumen(y, alerta):
    vp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    return {"pct_alertados": round(100 * float(alerta.mean()), 1),
            "deteccion_pct": round(100 * vp / max(int(y.sum()), 1), 1),
            "precision_pct": round(100 * vp / max(vp + fp, 1), 1),
            "tasa_falsas_alarmas_pct": round(100 * fp / max(int((y == 0).sum()), 1), 1)}


def evaluar_corto_plazo(ev=None, focos=None, horas=None):
    """Compara la alerta del modelo (≥ 15 %) con la alerta tras la regla de 48 h.
    Emisión simulada el día 1 de cada mes: se usan los focos de los dos últimos días del
    mes anterior (así la regla no ve ningún día del mes que se pronostica)."""
    ev = pd.read_csv(C.SALIDAS / "predicciones_evaluacion.csv", dtype={"ubigeo": str}) if ev is None else ev
    focos = focos_modis_archivados() if focos is None else focos
    dias = int((horas or C.HORAS_FOCO_ACTIVO) // 24)
    inicio_mes = pd.to_datetime(dict(year=ev.anio, month=ev.mes, day=1))
    f = focos[["ubigeo", "acq_date"]]
    cuenta = np.zeros(len(ev), dtype=int)
    por_dist = {u: g.acq_date.to_numpy() for u, g in f.groupby("ubigeo")}
    for i, (u, t0) in enumerate(zip(ev.ubigeo, inicio_mes)):
        fechas = por_dist.get(u)
        if fechas is not None:
            cuenta[i] = int(((fechas >= np.datetime64(t0 - pd.Timedelta(days=dias))) &
                             (fechas < np.datetime64(t0))).sum())
    y = ev.hubo_incendio.to_numpy().astype(int)
    umbral = C.UMBRALES_NIVEL[C.NIVELES.index(C.NIVEL_ALERTA) - 1]
    modelo = ev.prob.to_numpy() >= umbral
    con_regla = modelo | (cuenta > 0)
    res = {"emision_simulada": "día 1 de cada mes; focos de los 2 días previos",
           "anios": f"{ev.anio.min()}–{ev.anio.max()}",
           "distrito_meses_con_focos_48h": int((cuenta > 0).sum()),
           "solo_modelo": _resumen(y, modelo), "modelo_mas_regla_48h": _resumen(y, con_regla),
           "nota": "La parte meteorológica (Open-Meteo, 3 días) no se evalúa: no hay pronósticos archivados."}
    return res


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r = evaluar_corto_plazo()
    C.SALIDAS.mkdir(exist_ok=True)
    (C.SALIDAS / "regla_corto_plazo.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2))


def bootstrap_por_celda(ev=None, n=1000, semilla=42):
    """Intervalos con bootstrap por celdas de NASA POWER (0,5° × 0,625°).

    Los distritos de una misma celda comparten el clima diario, así que se
    remuestrean juntos: complementa el bootstrap por bloques de distrito.
    """
    import json as _json
    from shapely.geometry import shape
    from sklearn.metrics import brier_score_loss, roc_auc_score

    ev = pd.read_csv(C.SALIDAS / "predicciones_evaluacion.csv", dtype={"ubigeo": str}) if ev is None else ev
    geo = _json.loads(C.GEOJSON_DISTRITOS.read_text(encoding="utf-8"))
    celda = {}
    for f in geo["features"]:
        pt = shape(f["geometry"]).representative_point()
        celda[str(f["properties"]["ubigeo"])] = f"{round((pt.y + 90) / 0.5)}_{round((pt.x + 180) / 0.625)}"
    ev = ev.assign(celda=ev.ubigeo.map(celda))
    celdas = ev.celda.unique()
    pos = {c: np.where(ev.celda.values == c)[0] for c in celdas}
    y, p, b = ev.hubo_incendio.values, ev.prob.values, ev.frec_hist_mes.values
    rng = np.random.default_rng(semilla)
    auc, dif, bss = [], [], []
    for _ in range(n):
        ii = np.concatenate([pos[c] for c in rng.choice(celdas, len(celdas), replace=True)])
        if y[ii].min() == y[ii].max():
            continue
        a1, a0 = roc_auc_score(y[ii], p[ii]), roc_auc_score(y[ii], b[ii])
        auc.append(a1); dif.append(a1 - a0)
        bss.append(1 - brier_score_loss(y[ii], p[ii]) / brier_score_loss(y[ii], b[ii]))
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
    return {"celdas": int(len(celdas)), "distritos": int(ev.ubigeo.nunique()), "repeticiones": len(auc),
            "auc_ic95": q(auc), "diferencia_auc_ic95": q(dif), "brier_skill_ic95": q(bss),
            "prop_diferencia_no_positiva": round(float(np.mean(np.array(dif) <= 0)), 4)}
