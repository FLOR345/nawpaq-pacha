# ÑAWPAQ PACHA — v0.3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355818.svg)](https://doi.org/10.5281/zenodo.23355818)

Inteligencia artificial para anticipar y prevenir incendios de vegetación en los Andes.
Zona piloto: **región Puno, por distritos** (108 distritos con límite disponible).

> **English summary.** ÑAWPAQ PACHA is an open-source Python pipeline that downloads
> public satellite and climate data (NASA FIRMS, NASA POWER, MODIS, Copernicus DEM),
> forecasts one month ahead the probability of at least one active-fire detection in
> each district, and publishes risk levels, plain-language reasons and Spanish/Quechua
> recommendations in a static web dashboard. Run `python main.py --datos-reales`.
> Tests: `python -m pytest -q`. Licence: MIT.

## Novedades de la v0.3

| Cambio | Por qué |
|---|---|
| 7 indicadores de **días peligrosos** del mes anterior (días sin lluvia, racha seca, días calurosos, de aire seco, ventosos, humedad mínima, días secos + aire seco + viento) calculados con el clima **diario** (`src/variables_diarias.py`) | Una quema se escapa en un día seco y ventoso, no "en promedio" |
| Focos en los **distritos vecinos** y rezagos propios (hace 2 meses, mismo mes del año pasado) | El fuego se agrupa en el espacio y se repite |
| **Calibración multianual** fuera de muestra (un año fuera cada vez) combinada con la referencia climatológica | En v0.2 se calibraba con un solo año y las probabilidades se desajustaban en años secos o húmedos |
| **Evaluación año por año 2020–2025** (cada año con un modelo que no lo vio), intervalos bootstrap y *Brier skill score* | Dos años de prueba no bastaban para confirmar diferencias |
| Umbrales de riesgo 5 / 15 / 30 % (antes 10 / 30 / 55 %) | Con alerta desde 30 % solo se avisaba en 4 de cada 10 meses con incendio |
| Pruebas automáticas (`tests/`), estudio de ablación (`src/ablacion.py`), licencia MIT y `CITATION.cff` | Reproducibilidad y publicación del software |

Resultados de la evaluación 2020–2025 (7776 distrito-mes, 758 con incendio):

| Modelo | AUC | AUC jun–nov | PR-AUC | Brier |
|---|---|---|---|---|
| Referencia climatológica | 0,770 | 0,708 | 0,353 | 0,0773 |
| v0.2 (17 variables, calibración 1 año) | 0,775 | 0,653 | 0,304 | 0,0811 |
| **v0.3** | **0,830** | **0,719** | **0,359** | **0,0750** |

La v0.3 supera a la referencia en los 6 años evaluados (diferencia de AUC 0,06; IC 95 % 0,05–0,07).
Con alerta desde 15 % avisó en el 67 % de los meses con incendio.

> **v0.2 — datos reales.** `python main.py --datos-reales` descarga clima, vegetación,
> cobertura, relieve y focos de calor reales (NASA, MODIS, Copernicus, OpenStreetMap),
> entrena el modelo y genera el panel con alertas. Para los focos de calor del **año en
> curso** necesitas una clave gratuita de NASA FIRMS (ver "Clave FIRMS").

## Qué hace

```
Datos meteorológicos y territoriales → Variables → Modelo de IA → Mapa de riesgo → Alerta orientativa → Validación humana
```

1. **Datos** (`data/raw/`): una fila por distrito y mes (temperatura, lluvia, humedad,
   viento, NDVI, focos de calor) + características fijas del territorio.
2. **Variables** (`src/preparar_datos.py`): para estimar el mes *t* solo usa información
   hasta el mes *t-1* (anticipación real de 1 mes): rezagos, lluvia acumulada de 3 meses,
   anomalías frente a años anteriores, historial de incendios, estacionalidad.
3. **Modelo de IA** (`src/entrenar_modelo.py`): Gradient Boosting + calibración multianual
   combinada con la referencia climatológica. Evaluación año por año 2020–2025: cada año
   se pronostica con un modelo entrenado solo con los años anteriores, y se compara con
   una referencia simple ("solo historial") para demostrar que la IA aporta.
4. **Riesgo y alertas** (`src/predecir.py`, `src/alertas.py`): 4 niveles
   (Bajo <5 %, Medio 5–15 %, Alto 15–30 %, Muy alto ≥30 %), razones en lenguaje simple y
   recomendaciones en español y quechua (quechua en **borrador**, requiere validación).
5. **Panel web** (`dashboard/index.html`): mapa por distrito, ficha con alerta, ranking
   por provincia y métricas del modelo. Funciona sin internet para el mapa.

## Cómo ejecutarlo en tu PC (VS Code)

Requisitos: Python 3.10 o superior.

```bash
# 1. Abre la carpeta nawpaq_pacha en VS Code y una terminal (Ctrl + ñ)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Instala dependencias
pip install -r requirements.txt

# 3. (Una vez) guarda tu clave FIRMS en un archivo .env en la raíz:
#    FIRMS_MAP_KEY=tu_clave

# 4. Descarga datos reales y ejecuta todo (1.ª vez ~25 min; luego usa caché)
python main.py --datos-reales

# (opcional) pruebas automáticas y comparación v0.2 vs v0.3
pip install -r requirements-dev.txt
python -m pytest -q
python main.py --ablacion

# 5. Abre dashboard/index.html con doble clic (o con la extensión Live Server)
```

Salidas en `outputs/`: `pronostico_AAAA_MM.csv`, `alertas_AAAA_MM.csv`, `evaluacion_modelo.json`,
`predicciones_evaluacion.csv` (predicciones fuera de muestra 2020–2025) y, con `--ablacion`, `ablacion.json`.

## Estructura

```
nawpaq_pacha/
├── main.py                      ejecuta todo
├── requirements.txt
├── src/
│   ├── config.py                rutas, variables, umbrales de riesgo
│   ├── descargar_datos_reales.py descarga datos reales (NASA, MODIS, OSM)
│   ├── agregar_focos_firms.py   convierte focos reales de NASA FIRMS a conteo por distrito
│   ├── preparar_datos.py        variables para el modelo (incl. distritos vecinos)
│   ├── variables_diarias.py     días secos, calurosos, ventosos… desde el clima diario
│   ├── ablacion.py              comparación v0.2 vs v0.3 (opcional)
│   ├── entrenar_modelo.py       entrenamiento, calibración y evaluación
│   ├── alertas.py               textos de alerta ES/QU y explicaciones
│   └── predecir.py              pronóstico y exportación al panel
├── tests/                       pruebas automáticas (pytest)
├── LICENSE                      licencia MIT
├── CITATION.cff                 cómo citar el software
├── data/raw/                    datos de entrada (+ límites distritales de Puno)
├── data/processed/              dataset con variables
├── models/                      modelo entrenado (.joblib)
├── outputs/                     pronósticos, alertas y métricas
└── dashboard/                   panel web (index.html + datos_riesgo.js)
```

## Clave FIRMS

1. Pide tu clave (gratis, llega al correo): https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Crea el archivo `.env` en la raíz del proyecto con `FIRMS_MAP_KEY=tu_clave`
   (está en `.gitignore`; no lo compartas).

Sin clave el sistema funciona, pero los focos llegan solo hasta diciembre del año
anterior y el pronóstico queda para enero.

## Panel web

- **Inicio**: mapa satelital con el nivel de alerta por distrito, focos activos (7 días),
  incendios de los últimos 12 meses, áreas agrícolas, pastizales, comunidades y áreas
  protegidas; tarjeta de alerta, zonas en riesgo y condiciones actuales **en vivo**.
- **Alertas**: distritos con alerta y sus motivos. **Reportes**: ranking, CSV, métricas
  del modelo. **Configuración**: idioma, tema, mapa base, reglas y fuentes.
- La alerta final = riesgo mensual de la IA + focos activos (48 h) + pronóstico del
  tiempo de 7 días (Open-Meteo, se actualiza en vivo cada 30 min con internet).
  La regla está en `PELIGRO_TIEMPO` (`src/config.py`).

## Datos reales

`src/descargar_datos_reales.py` descarga (con caché en `data/raw/cache/`):

| Variable | Fuente que usa el sistema | Nota |
|---|---|---|
| Focos de calor | NASA FIRMS MODIS C6.1 (archivo anual de Perú + API) | confianza ≥ 30 |
| Focos activos 7 días | NASA FIRMS MODIS + VIIRS | para la alerta inmediata |
| Temperatura, lluvia, humedad, viento | NASA POWER (diario → mensual) | punto representativo del distrito |
| NDVI (vegetación) | MODIS MOD13Q1 250 m (ORNL DAAC) | ventana de 5×5 km |
| Cobertura (pastizal, agrícola) | MODIS MCD12Q1 2024 (IGBP) | ventana de 13×13 km |
| Altitud y pendiente | Copernicus DEM 90 m (Open-Meteo) | pendiente aproximada |
| Comunidades, áreas protegidas | OpenStreetMap (Overpass) | capas del mapa |
| Pronóstico y condiciones actuales | Open-Meteo | en vivo en el panel |

Mejoras futuras: SENAMHI/PISCO para lluvia, MCD64A1 (área quemada), MINAM (cobertura),
INEI (población) e INDECI-SINPAD para validar con emergencias reportadas.

Límites distritales: GeoJSON público basado en INEI (repositorio *juaneladio/peru-geojson*).
Para uso oficial conviene reemplazarlo por los límites vigentes del INEI.
