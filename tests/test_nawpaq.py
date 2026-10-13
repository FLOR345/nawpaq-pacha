"""
Pruebas automáticas de ÑAWPAQ PACHA.  Ejecutar:  python -m pytest -q

No necesitan internet ni los datos descargados: usan datos sintéticos pequeños.
Lo más importante que verifican es que NINGUNA variable del mes t use
información del propio mes t o posterior (anticipación real).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import alertas as A  # noqa: E402
import config as C  # noqa: E402
import entrenar_modelo as EM  # noqa: E402
import preparar_datos as PD  # noqa: E402
import variables_diarias as VD  # noqa: E402

UBIGEOS = ["210101", "210102", "210110"]          # Puno, Acora, Paucarcolla (vecinos)


def panel_sintetico(anios=range(2015, 2022), semilla=0):
    rng = np.random.default_rng(semilla)
    filas = [dict(ubigeo=u, anio=a, mes=m, temp_max=15 + rng.normal(), precip_mm=max(0, 60 + 40 * rng.normal()),
                  humedad_rel=60 + 5 * rng.normal(), viento_kmh=8 + rng.normal(), ndvi=0.3 + 0.05 * rng.normal(),
                  focos_calor=int(rng.poisson(1.0 if m in (7, 8, 9) else 0.1)))
             for u in UBIGEOS for a in anios for m in range(1, 13)]
    terr = pd.DataFrame({"ubigeo": UBIGEOS, "altitud_m": [3850, 3900, 3870], "pendiente_pct": [5, 6, 7],
                         "pct_pastizal": [80, 90, 70], "pct_agricola": [5, 3, 8]})
    return pd.DataFrame(filas), terr


def diarias_falsas(ubigeos):
    """Sustituye la lectura de la caché de NASA POWER por valores sintéticos."""
    filas = [dict(ubigeo=u, anio=a, mes=m, **{c: (a * 13 + m) % 7 for c in VD.COLUMNAS})
             for u in ubigeos for a in range(2015, 2023) for m in range(1, 13)]
    return pd.DataFrame(filas)


@pytest.fixture
def sin_cache(monkeypatch):
    monkeypatch.setattr(VD, "calcular", diarias_falsas)


def test_racha_maxima():
    s = pd.Series([True, True, False, True, True, True, False])
    assert VD._racha_maxima(s) == 3
    assert VD._racha_maxima(pd.Series([False, False])) == 0


def test_percentiles_solo_periodo_referencia():
    """Cambiar el clima posterior a ANIO_REF_FIN no debe cambiar los indicadores pasados."""
    fechas = pd.date_range("2015-01-01", "2022-12-31")
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"T2M_MAX": 15 + rng.normal(size=len(fechas)), "PRECTOTCORR": rng.exponential(2, len(fechas)),
                      "RH2M": 60 + 10 * rng.normal(size=len(fechas)), "WS2M": 3 + rng.random(len(fechas))}, index=fechas)
    a = VD.indicadores_distrito("210101", d)
    d2 = d.copy()
    d2.loc[d2.index.year > C.ANIO_REF_FIN, "T2M_MAX"] += 10       # futuro mucho más caluroso
    b = VD.indicadores_distrito("210101", d2)
    pasado = a.anio <= C.ANIO_REF_FIN
    pd.testing.assert_frame_equal(a[pasado], b[pasado])


def test_sin_fuga_de_informacion_futura(sin_cache):
    """Las variables del mes t no cambian si se altera lo ocurrido en el mes t."""
    panel, terr = panel_sintetico()
    base = PD.construir_variables(panel, terr)
    alterado = panel.copy()
    k = (alterado.anio == 2020) & (alterado.mes == 8)
    alterado.loc[k, ["focos_calor", "precip_mm", "temp_max", "humedad_rel", "viento_kmh", "ndvi"]] = 999
    nuevo = PD.construir_variables(alterado, terr)
    fila = lambda df: df[(df.anio == 2020) & (df.mes == 8)].sort_values("ubigeo")[C.VARIABLES_MODELO]
    pd.testing.assert_frame_equal(fila(base).reset_index(drop=True), fila(nuevo).reset_index(drop=True))
    # ...pero sí cambian las del mes siguiente
    sig = lambda df: df[(df.anio == 2020) & (df.mes == 9)].sort_values("ubigeo")["focos_lag1"].to_numpy()
    assert (sig(nuevo) == 999).all() and not (sig(base) == 999).any()


def test_vecinos_simetricos():
    vec = PD.vecinos()
    assert len(vec) == 108
    for u, vs in vec.items():
        for v in vs:
            assert u in vec[v]


def test_modelo_entrega_probabilidades(sin_cache):
    panel, terr = panel_sintetico(range(2015, 2023))
    df = PD.construir_variables(panel, terr).dropna(subset=C.VARIABLES_MODELO)
    paquete = EM.ajustar(df[df.anio < 2022])
    p = EM.probabilidad(paquete, df[df.anio == 2022])
    assert len(p) == (df.anio == 2022).sum()
    assert np.all((p > 0) & (p < 1))


def test_niveles_de_riesgo():
    u = C.UMBRALES_NIVEL
    niveles = EM.a_nivel(np.array([0.0, u[0], u[1], u[2], 0.99]))
    assert list(niveles) == ["Bajo", "Medio", "Alto", "Muy alto", "Muy alto"]


def test_regla_corto_plazo():
    n = C.NIVELES
    assert A.nivel_final("Bajo", [], 0, n)[0] == "Bajo"
    assert A.nivel_final("Bajo", [], 2, n)[0] == "Alto"                 # foco activo → al menos Alto
    assert A.nivel_final("Medio", ["Alto", "Extremo", "Bajo"], 0, n)[0] == "Alto"   # tiempo peligroso → sube
    assert A.nivel_final("Bajo", ["Alto", "Extremo", "Bajo"], 0, n)[0] == "Bajo"    # solo si ya era Medio+
