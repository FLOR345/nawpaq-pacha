"""
ÑAWPAQ PACHA — Configuración central del proyecto.

Aquí se definen rutas, variables del modelo y umbrales de riesgo.
Si cambias de zona piloto o agregas variables, este es el primer archivo que se edita.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
SALIDAS = RAIZ / "outputs"
DASHBOARD = RAIZ / "dashboard"

GEOJSON_DISTRITOS = DATA_RAW / "puno_distritos.geojson"
# Tabla mensual por distrito (la genera src/descargar_datos_reales.py)
PANEL_MENSUAL = DATA_RAW / "panel_mensual.csv"
DATASET_MODELO = DATA_PROC / "dataset_modelo.csv"
MODELO_ARCHIVO = MODELOS / "modelo_riesgo.joblib"

# --- Columnas que debe tener el panel mensual ------------------------------
# Una fila = un distrito en un mes.
COLUMNAS_PANEL = [
    "ubigeo",          # código INEI del distrito (6 dígitos)
    "anio", "mes",
    "temp_max",        # °C, promedio mensual de temperatura máxima
    "precip_mm",       # mm acumulados en el mes
    "humedad_rel",     # % humedad relativa media
    "viento_kmh",      # km/h velocidad media del viento
    "ndvi",            # índice de vegetación (0 a 1)
    "focos_calor",     # n.º de focos de calor / incendios detectados en el mes
]
# Características fijas del territorio (una fila por distrito)
COLUMNAS_TERRITORIO = [
    "ubigeo",
    "altitud_m",       # altitud media
    "pendiente_pct",   # pendiente media del terreno
    "pct_pastizal",    # % del distrito cubierto por pastizales/pajonal
    "pct_agricola",    # % de área agrícola
]
TERRITORIO = DATA_RAW / "territorio_distritos.csv"

# --- Variables que usa el modelo (se calculan en preparar_datos.py) --------
VARIABLES_MODELO = [
    "temp_max_lag1", "precip_lag1", "precip_3m", "anom_precip_3m",
    "humedad_lag1", "viento_lag1", "ndvi_lag1", "anom_ndvi_lag1",
    "focos_lag1", "focos_12m", "frec_hist_mes",
    "mes_sin", "mes_cos",
    "altitud_m", "pendiente_pct", "pct_pastizal", "pct_agricola",
    # v0.3: días peligrosos del mes anterior (clima diario, ver variables_diarias.py)
    "dias_secos_lag1", "racha_seca_lag1", "dias_calidos_lag1", "dias_aire_seco_lag1",
    "dias_ventosos_lag1", "hum_min_mes_lag1", "dias_peligro_lag1",
    # v0.3: fuego en los distritos vecinos y rezagos propios adicionales
    "vec_focos_lag1", "vec_focos_12m", "focos_lag2", "focos_mismo_mes_ant",
]
# Periodo de referencia para los percentiles de los indicadores diarios
# (solo años anteriores a la primera evaluación, para no filtrar información).
ANIO_REF_FIN = 2019

# Nombres legibles para explicar el riesgo a usuarios
NOMBRES_LEGIBLES = {
    "temp_max_lag1": "temperatura máxima del último mes",
    "precip_lag1": "lluvia del último mes",
    "precip_3m": "lluvia de los últimos 3 meses",
    "anom_precip_3m": "déficit de lluvia frente a lo normal",
    "humedad_lag1": "humedad del aire",
    "viento_lag1": "viento",
    "ndvi_lag1": "estado de la vegetación",
    "anom_ndvi_lag1": "vegetación más seca de lo normal",
    "focos_lag1": "focos de calor del último mes",
    "focos_12m": "incendios en los últimos 12 meses",
    "frec_hist_mes": "historial de incendios en este mes",
    "mes_sin": "época del año", "mes_cos": "época del año",
    "altitud_m": "altitud", "pendiente_pct": "pendiente del terreno",
    "pct_pastizal": "cobertura de pastizales",
    "pct_agricola": "área agrícola",
    "dias_secos_lag1": "días sin lluvia del último mes",
    "racha_seca_lag1": "racha seca más larga del último mes",
    "dias_calidos_lag1": "días muy calurosos del último mes",
    "dias_aire_seco_lag1": "días con aire muy seco del último mes",
    "dias_ventosos_lag1": "días ventosos del último mes",
    "hum_min_mes_lag1": "humedad mínima del último mes",
    "dias_peligro_lag1": "días secos, de aire seco y ventosos",
    "vec_focos_lag1": "focos en distritos vecinos el último mes",
    "vec_focos_12m": "focos en distritos vecinos en 12 meses",
    "focos_lag2": "focos de calor de hace dos meses",
    "focos_mismo_mes_ant": "focos en este mes del año pasado",
}

# --- Validación temporal (nunca se mezclan años) -----------------------------
# Evaluación "año por año" (origen móvil): para cada año de prueba se entrena
# solo con los años anteriores y se pronostica ese año completo. Así se mide
# el desempeño en 6 años distintos, secos y húmedos, y no en uno solo.
ANIOS_PRUEBA = list(range(2020, 2026))    # 2020–2025
# Calibración: probabilidades fuera de muestra (dejando un año fuera cada vez)
# sobre todos los años de entrenamiento, combinadas con la referencia
# climatológica en una regresión logística (ver entrenar_modelo.py).

# --- Niveles de riesgo -------------------------------------------------------
# El modelo entrega una probabilidad CALIBRADA (ajustada con el año de
# validación) de que ocurra al menos un incendio en el distrito ese mes.
# Esa probabilidad se traduce a 4 niveles fáciles de comunicar:
NIVELES = ["Bajo", "Medio", "Alto", "Muy alto"]
UMBRALES_NIVEL = [0.05, 0.15, 0.30]   # <5% Bajo · 5–15% Medio · 15–30% Alto · ≥30% Muy alto
NIVEL_ALERTA = "Alto"                 # desde este nivel (≥15 %) se genera alerta orientativa
# v0.3: umbrales más bajos que en v0.2 (10/30/55 %). En la evaluación 2020–2025
# la alerta desde 15 % avisó en ~2 de cada 3 meses con incendio; desde 30 % solo
# en ~4 de cada 10. Para prevenir conviene avisar antes y aceptar más falsas alarmas.
COLORES_NIVEL = {"Bajo": "#2e9e5b", "Medio": "#e8c62e", "Alto": "#ee7d1f", "Muy alto": "#c8262b"}

SEMILLA = 42

# --- Modelo base ----------------------------------------------------------------
# "random_forest" (por defecto desde v0.3.1) o "gradient_boosting" (v0.3.0).
# En la comparación 2020–2025 (src/ablacion.py) Random Forest obtuvo AUC 0.842,
# Gradient Boosting 0.830 y regresión logística 0.828, con la misma calibración.
MODELO_BASE = "random_forest"

# --- Alerta de corto plazo (próximos días) ----------------------------------
# El modelo estima el riesgo del MES. La alerta final se ajusta con:
#  1. Focos activos detectados por satélite (NASA FIRMS) en las últimas 48 h
#     → el distrito pasa al menos a "Alto".
#  2. Pronóstico del tiempo de 7 días (Open-Meteo): cada día suma puntos de
#     peligro por aire seco, viento fuerte y ausencia de lluvia. Si en los
#     próximos 3 días hay 2 o más días de peligro "Alto" o "Extremo", el
#     nivel sube un escalón (solo si el modelo ya indica "Medio" o más).
# Esta regla es orientativa (no está entrenada); el panel la aplica en vivo.
PELIGRO_TIEMPO = {
    "humedad_min": [35, 25],    # % → 1 punto si < 35, 2 si < 25
    "viento_max": [20, 30],     # km/h → 1 punto si ≥ 20, 2 si ≥ 30
    "lluvia_seca": 1.0,         # mm → 1 punto si llueve menos que esto
    "lluvia_anula": 5.0,        # mm → con esta lluvia el día no suma peligro
    "niveles": ["Bajo", "Moderado", "Alto", "Extremo"],
    "cortes": [2, 3, 4],        # puntos: <2 Bajo · 2 Moderado · 3 Alto · ≥4 Extremo
}
HORAS_FOCO_ACTIVO = 48
