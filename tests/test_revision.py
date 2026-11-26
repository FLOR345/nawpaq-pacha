"""Pruebas de los análisis añadidos en la revisión (bootstrap por bloques, regla de 48 h)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import analisis_revision as AR  # noqa: E402
import entrenar_modelo as EM  # noqa: E402


def test_bootstrap_por_bloques_de_distrito():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 20)
    y = (rng.random(600) < 0.2).astype(int)
    p = np.clip(y * 0.3 + rng.random(600) * 0.5, 0, 1)
    r = rng.random(600)
    ic = EM.intervalos_bootstrap(y, p, r, grupos=g, n=200)
    assert ic["metodo"] == "bloques de distrito" and ic["auc_ia"][0] < ic["auc_ia"][1]
    assert EM.intervalos_bootstrap(y, p, r, n=50)["metodo"] == "distrito-meses"


def test_regla_48h_usa_solo_dias_previos_al_mes():
    ev = pd.DataFrame({"ubigeo": ["A", "A", "B"], "anio": [2020, 2020, 2020], "mes": [8, 9, 8],
                       "hubo_incendio": [1, 1, 0], "prob": [0.05, 0.05, 0.05]})
    focos = pd.DataFrame({"ubigeo": ["A", "A"], "confidence": [50, 90],
                          "acq_date": pd.to_datetime(["2020-07-31", "2020-09-01"])})
    r = AR.evaluar_corto_plazo(ev, focos, horas=48)
    # el foco del 31/07 eleva agosto; el del 01/09 cae DENTRO de septiembre y no se usa
    assert r["distrito_meses_con_focos_48h"] == 1
    assert r["solo_modelo"]["deteccion_pct"] == 0 and r["modelo_mas_regla_48h"]["deteccion_pct"] == 50
    assert AR.focos_modis_alta(focos).focos_modis_alta.sum() == 1


def test_textos_alerta_en_tres_idiomas():
    import alertas as A
    for nivel, textos in A.RECOMENDACIONES.items():
        assert set(textos) == {"es", "qu", "en"}, nivel
        assert len(textos["es"]) == len(textos["qu"]) == len(textos["en"]), nivel
    assert set(A.TITULO) == set(A.AVISO) == {"es", "qu", "en"}
    assert set(A.NIVEL_EN) == set(A.NIVEL_QU)


def test_imputacion_no_usa_meses_posteriores(tmp_path):
    import numpy as np
    import pandas as pd
    from descargar_datos_reales import imputar_retrospectiva
    panel = pd.DataFrame({"ubigeo": ["a"] * 5 + ["b"] * 3,
                          "anio": [2020] * 8, "mes": [1, 2, 3, 4, 5, 1, 2, 3],
                          "ndvi": [np.nan, 0.2, np.nan, 0.9, np.nan, 0.5, np.nan, 0.1]})
    out = imputar_retrospectiva(panel, ["ndvi"], tmp_path / "imp.json")
    a = out[out.ubigeo == "a"].ndvi.tolist()
    assert np.isnan(a[0])                 # sin pasado: queda vacío (no se rellena hacia atrás)
    assert a[2] == 0.2                    # último valor anterior, no la media con 0.9
    assert a[4] == 0.9
    assert out[out.ubigeo == "b"].ndvi.tolist()[1] == 0.5   # no cruza distritos
