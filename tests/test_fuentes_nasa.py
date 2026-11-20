"""
Pruebas de las fuentes NASA opcionales (v0.4). Sin internet: datos sintéticos.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import fuentes_nasa as FN  # noqa: E402
import preparar_datos as PD  # noqa: E402
from test_nawpaq import UBIGEOS, diarias_falsas, panel_sintetico  # noqa: E402
import variables_diarias as VD  # noqa: E402


@pytest.fixture
def sin_cache(monkeypatch):
    monkeypatch.setattr(VD, "calcular", diarias_falsas)


def nasa_sintetica(anios=range(2015, 2022), semilla=1):
    rng = np.random.default_rng(semilla)
    mensual = pd.DataFrame([dict(ubigeo=u, anio=a, mes=m, ndvi_area=0.3 + 0.05 * rng.normal(),
                                 lst_dia_c=20 + rng.normal(), humedad_suelo=0.2 + 0.02 * rng.normal(),
                                 quemado_pct=float(rng.random() < 0.1), focos_viirs=int(rng.poisson(1.5)))
                            for u in UBIGEOS for a in anios for m in range(1, 13)])
    anual = pd.DataFrame([dict(ubigeo=u, anio=a, pct_pastizal_anual=80 + a - 2015, pct_agricola_anual=5.0)
                          for u in UBIGEOS for a in anios])
    return mensual, anual


def test_nombre_de_archivo_appeears():
    capa, fecha = FN.fecha_de("MOD13A2.061__1km_16_days_NDVI_doy2020065_aid0001.tif")
    assert capa == "_1km_16_days_NDVI" and fecha == dt.date(2020, 3, 5)
    capa, fecha = FN.fecha_de("SPL3SMP_E.006_Soil_Moisture_Retrieval_Data_AM_soil_moisture_doy2016366_aid0001.tif")
    assert capa.endswith("AM_soil_moisture") and fecha == dt.date(2016, 12, 31)
    assert FN.fecha_de("MOD13A2-061-QA-lookup.csv") == (None, None)
    # nombre real de AppEEARS con hora en la fecha (doyAAAADDDhhmmss)
    assert FN.fecha_de("MOD11A2.061_LST_Day_1km_doy2015209000000_aid0001.tif") == ("LST_Day_1km", dt.date(2015, 7, 28))


def test_rango_valido_en_unidades_escaladas():
    # AppEEARS da ValidMin/ValidMax en unidades ya escaladas (LST en kelvin), no en el valor crudo
    meta = {"ScaleFactor": 0.02, "AddOffset": 0, "FillValue": 0, "ValidMin": 150.0, "ValidMax": 1310.7}
    v = FN.escalar(np.array([14916, 5000, np.nan], dtype=float), meta)
    assert v[0] == pytest.approx(298.32)
    assert np.isnan(v[1]) and np.isnan(v[2])


def _tif(ruta, datos, x0=-70.0, y0=-15.0, paso=0.01, nodata=-3000):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(ruta, "w", driver="GTiff", height=datos.shape[0], width=datos.shape[1], count=1,
                       dtype=datos.dtype, crs="EPSG:4326", transform=from_origin(x0, y0, paso, paso),
                       nodata=nodata) as dst:
        dst.write(datos, 1)


def test_promedio_sobre_el_poligono(tmp_path):
    pytest.importorskip("rasterio")
    from shapely.geometry import box
    FN._MASCARAS.clear()
    datos = np.zeros((10, 20), dtype="int16")
    datos[:, :10] = 2000          # mitad oeste: NDVI 0.2
    datos[:, 10:] = 6000          # mitad este:  NDVI 0.6
    datos[0, 0] = -3000           # relleno: no cuenta
    _tif(tmp_path / "a.tif", datos)
    geoms = {"oeste": box(-70.0, -15.1, -69.9, -15.0), "este": box(-69.9, -15.1, -69.8, -15.0),
             "chico": box(-69.951, -15.051, -69.949, -15.049)}      # más chico que un píxel
    v, tr, forma = FN.leer_tif(tmp_path / "a.tif", {"FillValue": -3000})
    m = FN.media_por_distrito(FN.escalar(v, {"ScaleFactor": 0.0001}), geoms, tr, forma)
    assert m["oeste"] == pytest.approx(0.2) and m["este"] == pytest.approx(0.6)
    assert not np.isnan(m["chico"])                                 # usa los píxeles que toca


def test_area_quemada_y_cobertura(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    from shapely.geometry import box
    FN._MASCARAS.clear()
    monkeypatch.setattr(FN, "CARPETA", tmp_path)
    geoms = {"210101": box(-70.0, -15.1, -69.9, -15.0)}
    q = np.zeros((10, 10), dtype="int16")
    q[:2, :] = 200                # 20 % quemado (día del año 200)
    q[9, 9] = -2                  # agua: no cuenta en el denominador
    (tmp_path / "nawpaq_quemado_2020").mkdir()
    _tif(tmp_path / "nawpaq_quemado_2020" / "MCD64A1.061_Burn_Date_doy2020183_aid0001.tif", q, nodata=-32768)
    t = {"fuente": "quemado", "anio": 2020, "capas": {"valor": "Burn_Date"}, "meta": {"Burn_Date": {}}}
    filas = FN.zonal_fuente(t, geoms)
    assert filas[0][2] == "quemado_pct" and filas[0][3] == pytest.approx(100 * 20 / 99)

    lc = np.full((10, 10), 10, dtype="uint8")
    lc[:5, :] = 12                # mitad cultivos, mitad pastizal
    (tmp_path / "nawpaq_cobertura_2020").mkdir()
    _tif(tmp_path / "nawpaq_cobertura_2020" / "MCD12Q1.061_LC_Type1_doy2020001_aid0001.tif", lc, nodata=255)
    t = {"fuente": "cobertura", "anio": 2020, "capas": {"valor": "LC_Type1"}, "meta": {"LC_Type1": {}}}
    _, anual = FN.tablas_desde_filas(FN.zonal_fuente(t, geoms))
    assert anual.pct_pastizal_anual.iloc[0] == pytest.approx(50) and anual.pct_agricola_anual.iloc[0] == pytest.approx(50)


def test_composites_se_promedian_por_mes():
    filas = [("210101", dt.date(2020, 3, 5), "ndvi_area", 0.2), ("210101", dt.date(2020, 3, 21), "ndvi_area", 0.4),
             ("210101", dt.date(2020, 4, 6), "ndvi_area", 0.5)]
    mensual, _ = FN.tablas_desde_filas(filas)
    assert mensual.set_index("mes").ndvi_area.to_dict() == pytest.approx({3: 0.3, 4: 0.5})


def test_variables_nasa_sin_fuga(sin_cache):
    """Las variables NASA del mes t no cambian si se altera el mes t; la cobertura usa el año anterior."""
    panel, terr = panel_sintetico()
    mensual, anual = nasa_sintetica()
    base = PD.construir_variables(panel, terr, nasa=(mensual, anual))
    alt = mensual.copy()
    k = (alt.anio == 2020) & (alt.mes == 8)
    alt.loc[k, ["ndvi_area", "lst_dia_c", "humedad_suelo", "quemado_pct", "focos_viirs"]] = 999
    alt_an = anual.copy()
    alt_an.loc[alt_an.anio == 2020, ["pct_pastizal_anual", "pct_agricola_anual"]] = 999
    nuevo = PD.construir_variables(panel, terr, nasa=(alt, alt_an))
    fila = lambda df: (df[(df.anio == 2020) & (df.mes == 8)].sort_values("ubigeo")[C.VARIABLES_NASA]
                       .reset_index(drop=True))
    pd.testing.assert_frame_equal(fila(base), fila(nuevo))
    sig = nuevo[(nuevo.anio == 2020) & (nuevo.mes == 9)]
    assert (sig.ndvi_area_lag1 == 999).all() and (sig.focos_viirs_lag1 == 999).all()
    assert (nuevo[nuevo.anio == 2021].pct_pastizal_anual == 999).all()


def test_etiquetas_alternativas(sin_cache):
    panel, terr = panel_sintetico()
    mensual, anual = nasa_sintetica()
    for etiqueta, col in (("viirs", "focos_viirs"), ("area_quemada", "quemado_pct")):
        df = PD.construir_variables(panel, terr, etiqueta=etiqueta, nasa=(mensual, anual))
        assert ((df[col] > 0).astype(float) == df.hubo_incendio).all()
    df = PD.construir_variables(panel, terr, etiqueta="modis_viirs", nasa=(mensual, anual))
    assert (((df.focos_calor + df.focos_viirs) > 0).astype(float) == df.hubo_incendio).all()
    with pytest.raises(ValueError):
        PD.construir_variables(panel, terr, etiqueta="viirs", nasa=(None, None))


def test_modelo_acepta_variables_nasa_vacias(sin_cache):
    import entrenar_modelo as EM
    panel, terr = panel_sintetico(range(2015, 2023))
    mensual, anual = nasa_sintetica(range(2015, 2023))
    mensual.loc[mensual.index % 5 == 0, "humedad_suelo"] = np.nan        # SMAP con huecos
    df = PD.construir_variables(panel, terr, nasa=(mensual, anual))
    df = df.dropna(subset=[v for v in C.VARIABLES_MODELO if v not in C.VARIABLES_NASA])
    X = [v for v in C.VARIABLES_MODELO if v not in C.VARIABLES_NASA] + C.VARIABLES_NASA
    paquete = EM.ajustar(df[df.anio < 2022], X=X, crear_base=EM._random_forest)
    p = EM.probabilidad(paquete, df[df.anio == 2022])
    assert np.all((p > 0) & (p < 1))
