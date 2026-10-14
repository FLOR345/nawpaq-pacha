"""
ÑAWPAQ PACHA — Entrenamiento y evaluación del modelo de IA (v0.3).

Modelo:
 1. Random Forest (por defecto) o Gradient Boosting, según config.MODELO_BASE,
    con las variables de config.VARIABLES_MODELO.
 2. Calibración y combinación: con predicciones FUERA DE MUESTRA (se deja un
    año fuera cada vez) se ajusta una regresión logística que combina la
    puntuación del modelo base con la referencia climatológica
    (frecuencia histórica de incendio del distrito en ese mes). Así la
    probabilidad final queda calibrada con varios años, secos y húmedos, y no
    con uno solo como en v0.2.

Evaluación (origen móvil, config.ANIOS_PRUEBA):
    para cada año T se entrena con los años < T y se pronostica T completo.
    Se compara con la referencia "solo historial". Si la IA no la supera,
    no aporta valor más allá del calendario.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C

MESES_SECOS = [6, 7, 8, 9, 10, 11]


# --- Modelo ----------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _gradient_boosting():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=C.SEMILLA)


def _random_forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                  n_jobs=-1, random_state=C.SEMILLA)


def _base():
    """Clasificador base según config.MODELO_BASE."""
    return _random_forest() if C.MODELO_BASE == "random_forest" else _gradient_boosting()


def _entradas_calibrador(p_base, df):
    return np.c_[_logit(p_base), _logit(df.frec_hist_mes.to_numpy())]


def ajustar(tr, X=None, crear_base=None):
    """Entrena el modelo completo (base + calibrador) con la tabla tr.
    crear_base: función que devuelve un clasificador nuevo (por defecto, Gradient Boosting)."""
    X = X or C.VARIABLES_MODELO
    crear_base = crear_base or _base
    y = tr.hubo_incendio.to_numpy()
    fuera = np.zeros(len(tr))
    for anio in tr.anio.unique():                       # un año fuera cada vez
        k = (tr.anio == anio).to_numpy()
        fuera[k] = crear_base().fit(tr.loc[~k, X], y[~k]).predict_proba(tr.loc[k, X])[:, 1]
    calibrador = LogisticRegression().fit(_entradas_calibrador(fuera, tr), y)
    base = crear_base().fit(tr[X], y)
    return {"base": base, "calibrador": calibrador, "variables": X}


def probabilidad(paquete, df):
    """Probabilidad calibrada de al menos un incendio en el distrito-mes."""
    p = paquete["base"].predict_proba(df[paquete["variables"]])[:, 1]
    return paquete["calibrador"].predict_proba(_entradas_calibrador(p, df))[:, 1]


def a_nivel(prob):
    return pd.cut(prob, [-np.inf, *C.UMBRALES_NIVEL, np.inf], labels=C.NIVELES, right=False).astype(str)


# --- Métricas ----------------------------------------------------------------
def umbral_alerta():
    return C.UMBRALES_NIVEL[C.NIVELES.index(C.NIVEL_ALERTA) - 1]


def metricas(y, prob, nombre, umbral=None):
    """Métricas en el lenguaje de la propuesta: detección y falsas alarmas."""
    alerta = prob >= (umbral_alerta() if umbral is None else umbral)
    vp = int((alerta & (y == 1)).sum())
    fp = int((alerta & (y == 0)).sum())
    return {
        "modelo": nombre,
        "auc": round(roc_auc_score(y, prob), 3),
        "pr_auc": round(average_precision_score(y, prob), 3),
        "brier": round(brier_score_loss(y, prob), 4),
        "deteccion_pct": round(100 * vp / max(int(y.sum()), 1), 1),   # meses con incendio que tuvieron alerta
        "falsas_alarmas_pct": round(100 * fp / max(vp + fp, 1), 1),   # alertas sin incendio
        "alertas_emitidas": int(alerta.sum()),
        "meses_con_incendio": int(y.sum()),
    }


def intervalos_bootstrap(y, p_ia, p_ref, n=1000):
    rng = np.random.default_rng(C.SEMILLA)
    auc, dif, bss = [], [], []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        a1, a2 = roc_auc_score(y[b], p_ia[b]), roc_auc_score(y[b], p_ref[b])
        auc.append(a1)
        dif.append(a1 - a2)
        bss.append(1 - brier_score_loss(y[b], p_ia[b]) / brier_score_loss(y[b], p_ref[b]))
    ic = lambda v: [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
    return {"auc_ia": ic(auc), "diferencia_auc": ic(dif), "brier_skill_score": ic(bss)}


def importancia_permutacion(paquete, te, repeticiones=5):
    rng = np.random.default_rng(C.SEMILLA)
    y = te.hubo_incendio.to_numpy()
    base = roc_auc_score(y, probabilidad(paquete, te))
    filas = []
    for v in paquete["variables"]:
        caidas = []
        for _ in range(repeticiones):
            t = te.copy()
            t[v] = rng.permutation(t[v].to_numpy())
            caidas.append(base - roc_auc_score(y, probabilidad(paquete, t)))
        filas.append({"variable": v, "importancia": float(np.mean(caidas))})
    return (pd.DataFrame(filas).assign(descripcion=lambda d: d.variable.map(C.NOMBRES_LEGIBLES))
            .sort_values("importancia", ascending=False))


# --- Flujo principal -----------------------------------------------------------
def entrenar():
    df = pd.read_csv(C.DATASET_MODELO, dtype={"ubigeo": str})
    hist = df[~df.es_pronostico].reset_index(drop=True)

    # 1. Evaluación año por año
    partes, por_anio, paquete_ult = [], [], None
    for T in C.ANIOS_PRUEBA:
        tr, te = hist[hist.anio < T], hist[hist.anio == T]
        if te.empty:
            continue
        paquete = ajustar(tr)
        p = probabilidad(paquete, te)
        partes.append(te[["ubigeo", "anio", "mes", "hubo_incendio", "frec_hist_mes"]].assign(prob=p))
        y = te.hubo_incendio.to_numpy()
        por_anio.append({"anio": int(T), "meses_con_incendio": int(y.sum()),
                         "auc_ia": round(roc_auc_score(y, p), 3),
                         "auc_referencia": round(roc_auc_score(y, te.frec_hist_mes), 3)})
        paquete_ult, te_ult = paquete, te
        print(f"[modelo] {T}: AUC IA {por_anio[-1]['auc_ia']:.3f} · referencia {por_anio[-1]['auc_referencia']:.3f}")
    ev = pd.concat(partes, ignore_index=True)
    y, p_ia, p_ref = ev.hubo_incendio.to_numpy(), ev.prob.to_numpy(), ev.frec_hist_mes.to_numpy()
    seco = ev.mes.isin(MESES_SECOS).to_numpy()

    resultados = [metricas(y, p_ia, "ÑAWPAQ PACHA (IA)"), metricas(y, p_ref, "Referencia: solo historial")]
    for r, p in zip(resultados, (p_ia, p_ref)):
        r["auc_epoca_seca"] = round(roc_auc_score(y[seco], p[seco]), 3)
    umbrales = [{"umbral": u, "ia": metricas(y, p_ia, "ia", u), "referencia": metricas(y, p_ref, "ref", u)}
                for u in (0.05, 0.10, 0.15, 0.20, 0.30)]
    calibracion = (pd.DataFrame({"rango": pd.cut(p_ia, [0, .05, .1, .2, .3, .5, 1]).astype(str), "y": y, "p": p_ia})
                   .groupby("rango", sort=False).agg(n=("y", "size"), observado=("y", "mean"), predicho=("p", "mean"))
                   .round(3).reset_index().to_dict("records"))
    C.SALIDAS.mkdir(exist_ok=True)
    ev.to_csv(C.SALIDAS / "predicciones_evaluacion.csv", index=False)

    # 2. Importancia de variables (modelo del último año evaluado, sobre ese año)
    importancia = importancia_permutacion(paquete_ult, te_ult)

    # 3. Modelo final con todos los años, para pronosticar el mes siguiente
    paquete = ajustar(hist)
    paquete["normales"] = (hist.groupby(["ubigeo", "mes"])[
        ["precip_3m", "humedad_lag1", "temp_max_lag1", "ndvi_lag1", "viento_lag1", "dias_secos_lag1"]]
        .mean().reset_index())
    C.MODELOS.mkdir(exist_ok=True)
    joblib.dump(paquete, C.MODELO_ARCHIVO)

    resumen = {
        "version": "0.3.1",
        "periodos": {"entrenamiento": f"{hist.anio.min()} hasta el año anterior a cada prueba",
                     "validacion": "calibración fuera de muestra (un año fuera cada vez)",
                     "prueba": f"{C.ANIOS_PRUEBA[0]}–{C.ANIOS_PRUEBA[-1]}, año por año"},
        "anticipacion": "1 mes (usa solo datos hasta el mes anterior)",
        "umbral_alerta": umbral_alerta(),
        "metricas_prueba": resultados,
        "intervalos_95": intervalos_bootstrap(y, p_ia, p_ref),
        "brier_skill_score": round(1 - brier_score_loss(y, p_ia) / brier_score_loss(y, p_ref), 3),
        "por_anio": por_anio,
        "umbrales": umbrales,
        "calibracion": calibracion,
        "importancia": importancia.head(12).round(4).to_dict("records"),
    }
    (C.SALIDAS / "evaluacion_modelo.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[modelo] Evaluación {C.ANIOS_PRUEBA[0]}–{C.ANIOS_PRUEBA[-1]} (cada año con un modelo que no lo vio):")
    print(pd.DataFrame(resultados).to_string(index=False))
    ic = resumen["intervalos_95"]
    print(f"[modelo] Diferencia de AUC IA − referencia: IC 95 % {ic['diferencia_auc']} · "
          f"Brier skill score {resumen['brier_skill_score']} (IC 95 % {ic['brier_skill_score']})")
    print("\n[modelo] Variables más influyentes:")
    print(importancia.head(8)[["descripcion", "importancia"]].to_string(index=False))
    return resumen


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    entrenar()
