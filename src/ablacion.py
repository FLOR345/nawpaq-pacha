"""
ÑAWPAQ PACHA — Estudio de ablación: ¿qué mejora aporta cada cambio de v0.3?

Compara, con la MISMA evaluación año por año (config.ANIOS_PRUEBA):
  A. Referencia climatológica (solo historial del distrito en ese mes)
  B. v0.2: 17 variables, calibración con un solo año (el anterior al de prueba)
  C. 17 variables + calibración multianual combinada con la referencia
  D. v0.3: 28 variables + calibración multianual combinada con la referencia

Uso:  python src/ablacion.py      (tarda unos minutos; no se usa en main.py)
Salida: outputs/ablacion.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C
from entrenar_modelo import MESES_SECOS, _base, _logit, ajustar, probabilidad

VARIABLES_V02 = C.VARIABLES_MODELO[:17]


def v02_un_anio(tr, te):
    """Esquema de v0.2: entrena con años < T−1 y calibra (Platt) con el año T−1."""
    ult = tr.anio.max()
    a, b = tr[tr.anio < ult], tr[tr.anio == ult]
    base = _base().fit(a[VARIABLES_V02], a.hubo_incendio)
    platt = LogisticRegression().fit(_logit(base.predict_proba(b[VARIABLES_V02])[:, 1]).reshape(-1, 1), b.hubo_incendio)
    return platt.predict_proba(_logit(base.predict_proba(te[VARIABLES_V02])[:, 1]).reshape(-1, 1))[:, 1]


def main():
    df = pd.read_csv(C.DATASET_MODELO, dtype={"ubigeo": str})
    hist = df[~df.es_pronostico]
    modelos = {
        "A. Referencia climatológica": lambda tr, te: te.frec_hist_mes.to_numpy(),
        "B. v0.2 (17 variables, calibración 1 año)": v02_un_anio,
        "C. 17 variables + calibración multianual": lambda tr, te: probabilidad(ajustar(tr, VARIABLES_V02), te),
        "D. v0.3 (28 variables + calibración multianual)": lambda tr, te: probabilidad(ajustar(tr), te),
    }
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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
