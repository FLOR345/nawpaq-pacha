"""
ÑAWPAQ PACHA — Estudio de ablación: ¿qué mejora aporta cada cambio de v0.3?

Compara, con la MISMA evaluación año por año (config.ANIOS_PRUEBA):
  A. Referencia climatológica (solo historial del distrito en ese mes)
  B. v0.2: 17 variables, calibración con un solo año (el anterior al de prueba)
  C. 17 variables + calibración multianual combinada con la referencia
  D. Gradient Boosting, 28 variables + calibración multianual con la referencia
  E. Regresión logística con las mismas 28 variables y la misma calibración
  F. Random Forest con las mismas 28 variables y la misma calibración (modelo actual)
  G. (v0.4, si existe data/raw/nasa_mensual.csv) F + cada fuente NASA por separado
     y F + todas juntas (AppEEARS, SMAP, VIIRS)
Así se ve qué aporta cada cambio y qué modelo base conviene usar.

Además escribe outputs/sensibilidad_etiqueta.json: la referencia y el modelo F
evaluados con cada definición de "hubo incendio" disponible (MODIS, VIIRS,
MODIS o VIIRS, área quemada MCD64A1).

Uso:  python src/ablacion.py      (tarda unos minutos; no se usa en main.py)
Salida: outputs/ablacion.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C
from entrenar_modelo import MESES_SECOS, _gradient_boosting, _logit, _random_forest, ajustar, probabilidad

VARIABLES_V02 = C.VARIABLES_MODELO[:17]
VARIABLES_V03 = [v for v in C.VARIABLES_MODELO if v not in C.VARIABLES_NASA]
GRUPOS_NASA = {
    "NDVI del distrito (MOD13A2)": ["ndvi_area_lag1", "anom_ndvi_area_lag1"],
    "temperatura de superficie (MOD11A2)": ["lst_lag1", "anom_lst_lag1"],
    "humedad del suelo (SMAP)": ["humedad_suelo_lag1", "anom_humedad_suelo_lag1"],
    "área quemada (MCD64A1)": ["quemado_lag1", "quemado_12m"],
    "focos VIIRS": ["focos_viirs_lag1", "focos_viirs_12m"],
    "cobertura anual (MCD12Q1)": ["pct_pastizal_anual", "pct_agricola_anual"],
}


def regresion_logistica():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))




def v02_un_anio(tr, te):
    """Esquema de v0.2: entrena con años < T−1 y calibra (Platt) con el año T−1."""
    ult = tr.anio.max()
    a, b = tr[tr.anio < ult], tr[tr.anio == ult]
    base = _gradient_boosting().fit(a[VARIABLES_V02], a.hubo_incendio)
    platt = LogisticRegression().fit(_logit(base.predict_proba(b[VARIABLES_V02])[:, 1]).reshape(-1, 1), b.hubo_incendio)
    return platt.predict_proba(_logit(base.predict_proba(te[VARIABLES_V02])[:, 1]).reshape(-1, 1))[:, 1]


def main():
    df = pd.read_csv(C.DATASET_MODELO, dtype={"ubigeo": str})
    hist = df[~df.es_pronostico]
    modelos = {
        "A. Referencia climatológica": lambda tr, te: te.frec_hist_mes.to_numpy(),
        "B. v0.2 (17 variables, calibración 1 año)": v02_un_anio,
        "C. 17 variables + calibración multianual": lambda tr, te: probabilidad(ajustar(tr, VARIABLES_V02, _gradient_boosting), te),
        "D. Gradient Boosting, 28 variables + calibración multianual":
            lambda tr, te: probabilidad(ajustar(tr, VARIABLES_V03, _gradient_boosting), te),
        "E. Regresión logística (mismas variables y calibración)":
            lambda tr, te: probabilidad(ajustar(tr, VARIABLES_V03, regresion_logistica), te),
        "F. Random Forest (modelo por defecto; mismas variables y calibración)":
            lambda tr, te: probabilidad(ajustar(tr, VARIABLES_V03, _random_forest), te),
    }
    # v0.4: cada fuente NASA por separado y todas juntas (solo las que existen en los datos)
    grupos = {k: v for k, v in GRUPOS_NASA.items() if all(c in hist for c in v) and hist[v].notna().any().all()}
    for nombre, vs in grupos.items():
        modelos[f"G. F + {nombre}"] = (lambda vs: lambda tr, te:
                                       probabilidad(ajustar(tr, VARIABLES_V03 + vs, _random_forest), te))(vs)
    if len(grupos) > 1:
        todas = [c for vs in grupos.values() for c in vs]
        modelos["G. F + todas las fuentes NASA"] = lambda tr, te: probabilidad(
            ajustar(tr, VARIABLES_V03 + todas, _random_forest), te)
    pred = {k: [] for k in modelos}
    ys, secos = [], []
    for T in C.ANIOS_PRUEBA:
        tr, te = hist[hist.anio < T], hist[hist.anio == T]
        ys.append(te.hubo_incendio.to_numpy())
        secos.append(te.mes.isin(MESES_SECOS).to_numpy())
        for k, f in modelos.items():
            pred[k].append(f(tr, te))
        print(f"[ablación] {T} listo")
    y, seco = np.concatenate(ys), np.concatenate(secos)
    filas = []
    for k, partes in pred.items():
        p = np.concatenate(partes)
        filas.append({"modelo": k, "auc": round(roc_auc_score(y, p), 3),
                      "auc_epoca_seca": round(roc_auc_score(y[seco], p[seco]), 3),
                      "pr_auc": round(average_precision_score(y, p), 3),
                      "brier": round(brier_score_loss(y, p), 4),
                      "auc_por_anio": [round(roc_auc_score(a, b), 3) for a, b in zip(ys, partes)]})
    (C.SALIDAS / "ablacion.json").write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(filas).drop(columns="auc_por_anio").to_string(index=False))
    sensibilidad_etiquetas()


def sensibilidad_etiquetas():
    """¿Se mantienen los resultados si 'hubo incendio' se define con otro sensor/producto?"""
    import preparar_datos as PD
    try:
        panel, terr = PD.cargar_datos()
    except FileNotFoundError:
        return
    nasa = PD.cargar_nasa()
    # focos MODIS de confianza alta (≥ 80) desde el archivo local de FIRMS
    try:
        import analisis_revision as AR
        alta = AR.focos_modis_alta()
        panel = panel.merge(alta, on=["ubigeo", "anio", "mes"], how="left")
        panel["focos_modis_alta"] = panel.focos_modis_alta.fillna(0)
        panel.loc[panel.focos_calor.isna(), "focos_modis_alta"] = np.nan
    except Exception as e:                                     # sin archivo FIRMS local
        print(f"[sensibilidad] sin focos de confianza alta: {e}")
    filas = []
    for etiqueta in C.ETIQUETAS:
        try:
            df = PD.construir_variables(panel, terr, etiqueta=etiqueta, nasa=nasa)
        except ValueError as e:
            print(f"[sensibilidad] {etiqueta}: {e}")
            continue
        df = df.dropna(subset=VARIABLES_V03 + ["hubo_incendio"])
        ys, p_ref, p_rf = [], [], []
        for T in C.ANIOS_PRUEBA:
            tr, te = df[df.anio < T], df[df.anio == T]
            if te.empty or tr.anio.nunique() < 2 or te.hubo_incendio.nunique() < 2:
                continue
            ys.append(te.hubo_incendio.to_numpy())
            p_ref.append(te.frec_hist_mes.to_numpy())
            p_rf.append(probabilidad(ajustar(tr, VARIABLES_V03, _random_forest), te))
        if not ys:
            continue
        y, a, b = np.concatenate(ys), np.concatenate(p_ref), np.concatenate(p_rf)
        filas.append({"etiqueta": etiqueta, "prevalencia": round(float(y.mean()), 3),
                      "anios_prueba": len(ys),
                      "auc_referencia": round(roc_auc_score(y, a), 3), "auc_modelo": round(roc_auc_score(y, b), 3),
                      "brier_referencia": round(brier_score_loss(y, a), 4),
                      "brier_modelo": round(brier_score_loss(y, b), 4)})
        print(f"[sensibilidad] {etiqueta}: AUC modelo {filas[-1]['auc_modelo']} · referencia {filas[-1]['auc_referencia']}")
    (C.SALIDAS / "sensibilidad_etiqueta.json").write_text(json.dumps(filas, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
    # regla de corto plazo (48 h) reconstruida con el archivo de FIRMS
    try:
        import analisis_revision as AR
        r = AR.evaluar_corto_plazo()
        (C.SALIDAS / "regla_corto_plazo.json").write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
        print(f"[regla 48 h] solo modelo {r['solo_modelo']} · con regla {r['modelo_mas_regla_48h']}")
    except Exception as e:
        print(f"[regla 48 h] no evaluada: {e}")
    # bootstrap por celdas climáticas de NASA POWER
    try:
        import analisis_revision as AR
        r = AR.bootstrap_por_celda()
        (C.SALIDAS / "bootstrap_celdas.json").write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
        print(f"[bootstrap por celda] {r}")
    except Exception as e:
        print(f"[bootstrap por celda] no calculado: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
