"""
ÑAWPAQ PACHA — Indicadores mensuales calculados a partir del clima DIARIO.

Los promedios mensuales esconden los días peligrosos: una quema se descontrola
en un día seco, caluroso y con viento, no "en promedio". Por eso, con los datos
diarios de NASA POWER (ya descargados en data/raw/cache/power_*.json), se
cuentan para cada distrito y mes:

  dias_secos      días con lluvia < 1 mm
  racha_seca      racha más larga de días secos seguidos
  dias_calidos    días con T. máxima sobre el percentil 90 del distrito
  dias_aire_seco  días con humedad relativa bajo el percentil 20 del distrito
  dias_ventosos   días con viento sobre el percentil 80 del distrito
  hum_min_mes     humedad relativa diaria más baja del mes (%)
  dias_peligro    días secos + aire seco + viento fuerte a la vez

Los percentiles de cada distrito se calculan SOLO con el periodo de referencia
(config.ANIO_REF_FIN y anteriores), para no usar información de los años que
luego se evalúan.
"""
import json

import numpy as np
import pandas as pd

import config as C

COLUMNAS = ["dias_secos", "racha_seca", "dias_calidos", "dias_aire_seco",
            "dias_ventosos", "hum_min_mes", "dias_peligro"]


def _racha_maxima(serie_bool):
    """Longitud de la racha más larga de valores True seguidos."""
    v = serie_bool.to_numpy(dtype=int)
    if not v.any():
        return 0
    # posiciones donde cambia el valor → largo de cada tramo
    cortes = np.flatnonzero(np.diff(np.r_[0, v, 0]))
    return int((cortes[1::2] - cortes[::2]).max())


def indicadores_distrito(ubigeo, diario):
    """diario: DataFrame indexado por fecha con T2M_MAX, PRECTOTCORR, RH2M, WS2M."""
    d = diario.replace(-999.0, np.nan).dropna()
    ref = d[d.index.year <= C.ANIO_REF_FIN]
    if len(ref) < 365:
        ref = d                                   # historial corto: usa todo lo disponible
    t90, rh20, w80 = ref.T2M_MAX.quantile(0.9), ref.RH2M.quantile(0.2), ref.WS2M.quantile(0.8)

    seco = d.PRECTOTCORR < 1.0
    aire_seco = d.RH2M < rh20
    ventoso = d.WS2M > w80
    tabla = pd.DataFrame({
        "seco": seco, "calido": d.T2M_MAX > t90, "aire_seco": aire_seco,
        "ventoso": ventoso, "rh": d.RH2M,
        "peligro": seco & aire_seco & (d.WS2M > 0.8 * w80),
    })
    g = tabla.groupby([tabla.index.year, tabla.index.month])
    m = pd.DataFrame({
        "dias_secos": g.seco.sum(),
        "racha_seca": g.seco.apply(_racha_maxima),
        "dias_calidos": g.calido.sum(),
        "dias_aire_seco": g.aire_seco.sum(),
        "dias_ventosos": g.ventoso.sum(),
        "hum_min_mes": g.rh.min().round(1),
        "dias_peligro": g.peligro.sum(),
    })
    m.index.names = ["anio", "mes"]
    return m.reset_index().assign(ubigeo=ubigeo)


def calcular(ubigeos):
    """Indicadores diarios de todos los distritos a partir de la caché de NASA POWER."""
    partes, faltan = [], []
    for u in ubigeos:
        archivo = C.DATA_RAW / "cache" / f"power_{u}.json"
        if not archivo.exists():
            faltan.append(u)
            continue
        par = json.loads(archivo.read_text())["parametros"]
        diario = pd.DataFrame(par)
        diario.index = pd.to_datetime(diario.index, format="%Y%m%d")
        partes.append(indicadores_distrito(u, diario))
    if faltan:
        raise FileNotFoundError(
            f"Falta el clima diario de {len(faltan)} distrito(s) en data/raw/cache/. "
            "Ejecuta primero: python main.py --datos-reales")
    return pd.concat(partes, ignore_index=True)
