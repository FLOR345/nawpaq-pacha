# ÑAWPAQ PACHA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355818.svg)](https://doi.org/10.5281/zenodo.23355818)
[![Pronóstico mensual](https://github.com/FLOR345/nawpaq-pacha/actions/workflows/pronostico.yml/badge.svg)](https://github.com/FLOR345/nawpaq-pacha/actions/workflows/pronostico.yml)


**Pronóstico a un mes del riesgo de incendios de vegetación y alertas en español y quechua para los Andes peruanos.**
Zona piloto: **los 108 distritos de la región Puno, Perú**.

*Read in English: [README.md](README.md)*

**Demo en línea (se actualiza sola cada mes):** https://flor345.github.io/nawpaq-pacha/

ÑAWPAQ PACHA («tiempo de antes» en quechua) es un programa abierto en Python que:

1. descarga datos satelitales y climáticos públicos (NASA FIRMS, NASA POWER, MODIS, Copernicus DEM, OpenStreetMap);
2. estima, para cada distrito, la probabilidad de al menos un foco de calor en el **mes siguiente**;
3. convierte cada probabilidad en un nivel de riesgo, hasta tres razones en lenguaje sencillo y recomendaciones en **español y quechua**;
4. lo presenta en un **panel web estático** que se abre con doble clic, sin servidor.

```
Datos públicos → variables (datos hasta el mes t−1) → modelo calibrado → riesgo por distrito → alerta → revisión humana
```

## Resultados (evaluación año por año 2020–2025)

Cada año de 2020 a 2025 se pronosticó con un modelo entrenado **solo con los años anteriores**
(7 704 distrito-mes, 758 con incendio). Todas las configuraciones usan la misma evaluación
(`src/ablacion.py`).

| Configuración | AUC | AUC jun–nov | PR-AUC | Brier |
|---|---|---|---|---|
| Referencia climatológica (solo historial del distrito) | 0,770 | 0,708 | 0,353 | 0,0773 |
| v0.2: 17 variables, calibración con un año | 0,775 | 0,653 | 0,304 | 0,0811 |
| Gradient Boosting, 28 variables, calibración multianual (v0.3.0) | 0,830 | 0,719 | 0,359 | 0,0750 |
| Regresión logística, mismas variables y calibración | 0,828 | 0,717 | 0,359 | 0,0748 |
| **Random Forest, mismas variables y calibración (por defecto, v0.3.1)** | **0,842** | **0,737** | **0,386** | **0,0731** |

El modelo por defecto supera a la referencia en los seis años de prueba (diferencia de AUC
0,072; IC 95 % bootstrap 0,060–0,085; Brier skill score 0,056). Con alerta desde 15 % de
probabilidad avisó con un mes de anticipación en el 70 % de los meses con incendio.

## Instalación

Requisitos: Python 3.10 o superior.

```bash
git clone https://github.com/FLOR345/nawpaq-pacha.git
cd nawpaq-pacha
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Clave NASA FIRMS (opcional, gratuita)

1. Pide tu clave (llega al correo): https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Copia `.env.ejemplo` como `.env` y escribe `FIRMS_MAP_KEY=tu_clave`. El archivo `.env` está en `.gitignore`; no lo compartas.

Sin clave el sistema funciona, pero los focos llegan solo hasta diciembre del año
anterior y el pronóstico queda para enero.

## Uso

```bash
python main.py --datos-reales   # descarga/actualiza datos, entrena, evalúa y pronostica (1.ª vez ≈ 25 min, luego usa caché)
python main.py                  # usa los datos ya descargados
python main.py --ablacion       # además, compara modelos (outputs/ablacion.json)
```

Luego abre `dashboard/index.html` en el navegador.

Salidas en `outputs/`: `pronostico_AAAA_MM.csv`, `alertas_AAAA_MM.csv`, `evaluacion_modelo.json`,
`predicciones_evaluacion.csv` (predicciones fuera de muestra 2020–2025) y `ablacion.json`.

### Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Cómo funciona

| Etapa | Módulo | Qué hace |
|---|---|---|
| 1. Datos | `src/descargar_datos_reales.py` | Descarga con caché focos FIRMS, clima diario NASA POWER, NDVI y cobertura MODIS, relieve Copernicus y capas OSM; asigna los focos a cada distrito |
| 2. Variables | `src/preparar_datos.py`, `src/variables_diarias.py` | 28 variables: clima del mes anterior, días secos/calurosos/ventosos desde el clima diario, NDVI y anomalías, historial de fuego propio y de los distritos vecinos, estacionalidad y relieve |
| 3. Modelo | `src/entrenar_modelo.py` | Random Forest (o Gradient Boosting); puntuaciones fuera de muestra combinadas con la referencia en una calibración logística multianual; evaluación 2020–2025 |
| 4. Pronóstico y alertas | `src/predecir.py`, `src/alertas.py` | Niveles (bajo < 5 %, medio 5–15 %, alto 15–30 %, muy alto ≥ 30 %), ajuste de corto plazo con focos de 48 h y pronóstico Open-Meteo de 7 días, razones y recomendaciones ES/QU |
| 5. Panel | `dashboard/index.html` | Mapa Leaflet, fichas por distrito, alertas, reportes con métricas y clima en vivo |

Toda la configuración está en `src/config.py`. Para otra región, reemplaza
`data/raw/puno_distritos.geojson` por los límites distritales de esa región (campos
`ubigeo`, `distrito`, `provincia`) y ajusta `BBOX` en `src/descargar_datos_reales.py`.

## Fuentes de datos

Focos de calor: NASA FIRMS (MODIS C6.1 y VIIRS) · Clima: NASA POWER · NDVI y cobertura:
MODIS MOD13Q1 y MCD12Q1 (ORNL DAAC) · Relieve: Copernicus DEM vía Open-Meteo ·
Comunidades y áreas protegidas: © OpenStreetMap · Pronóstico: Open-Meteo.
Límites distritales derivados de *juaneladio/peru-geojson* (base INEI, MPL 2.0).

## Licencia

Código: MIT. Límites distritales: MPL 2.0. Los datos de entrada pertenecen a sus proveedores.
