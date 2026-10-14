# ÑAWPAQ PACHA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355818.svg)](https://doi.org/10.5281/zenodo.23355818)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Pronóstico mensual](https://github.com/FLOR345/nawpaq-pacha/actions/workflows/pronostico.yml/badge.svg)](https://github.com/FLOR345/nawpaq-pacha/actions/workflows/pronostico.yml)


**One-month-ahead vegetation fire risk forecasting and bilingual (Spanish/Quechua) alerts for the Peruvian Andes.**
Pilot area: the **108 districts of the Puno region, Peru**.

*Leer en español: [README.es.md](README.es.md)*

**Live demo (updated automatically every month):** https://flor345.github.io/nawpaq-pacha/

ÑAWPAQ PACHA ("the time before" in Quechua) is an open-source Python pipeline that:

1. downloads public satellite and climate data (NASA FIRMS, NASA POWER, MODIS, Copernicus DEM, OpenStreetMap);
2. estimates, for every district, the probability of at least one active-fire detection in the **next month**;
3. turns each probability into a risk level, up to three plain-language reasons and recommended actions in **Spanish and Quechua**;
4. publishes everything in a **static web dashboard** that opens from the file system, with no server.

```
Public data → features (data up to month t−1) → calibrated model → district risk → alert → human review
```

## Results (rolling evaluation 2020–2025)

Each year from 2020 to 2025 was forecast by a model trained **only on earlier years**
(7,704 district-months, 758 with fire). All configurations use the same evaluation
(`src/ablacion.py`).

| Configuration | AUC | AUC Jun–Nov | PR-AUC | Brier |
|---|---|---|---|---|
| Climatological baseline (district fire history only) | 0.770 | 0.708 | 0.353 | 0.0773 |
| v0.2: 17 predictors, calibrated on one year | 0.775 | 0.653 | 0.304 | 0.0811 |
| Gradient boosting, 28 predictors, multi-year calibration | 0.830 | 0.719 | 0.359 | 0.0750 |
| Logistic regression, same predictors and calibration | 0.828 | 0.717 | 0.359 | 0.0748 |
| **Random forest, same predictors and calibration (default model)** | **0.842** | **0.737** | **0.386** | **0.0731** |

The default model beats the baseline in each of the six test years (AUC difference
0.072, 95 % bootstrap CI 0.060–0.085; Brier skill score 0.056). With alerts from a 15 %
probability it flagged 70 % of fire months one month in advance.

## Installation

Requirements: Python ≥ 3.10.

```bash
git clone https://github.com/FLOR345/nawpaq-pacha.git
cd nawpaq-pacha
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### NASA FIRMS key (optional, free)

1. Request a key (it arrives by e-mail): https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Copy `.env.ejemplo` to `.env` and write `FIRMS_MAP_KEY=your_key`. The `.env` file is git-ignored; never share it.

Without a key the system still runs, but fire detections stop at December of the
previous year and the forecast is made for January.

## Usage

```bash
python main.py --datos-reales   # download/update data, train, evaluate, forecast (first run ≈ 25 min, then cached)
python main.py                  # reuse downloaded data
python main.py --ablacion       # also run the model comparison (outputs/ablacion.json)
```

Then open `dashboard/index.html` in a web browser.

Outputs in `outputs/`:

| File | Content |
|---|---|
| `pronostico_YYYY_MM.csv` | Probability, model level, final level and reasons for every district |
| `alertas_YYYY_MM.csv` | Districts with an alert (high or very high) |
| `evaluacion_modelo.json` | Metrics by year, bootstrap CIs, thresholds, calibration, variable importance |
| `predicciones_evaluacion.csv` | Out-of-sample monthly predictions 2020–2025 |
| `ablacion.json` | Model comparison (with `--ablacion`) |

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests run without internet or downloaded data. Among other things they check
that no predictor of month *t* uses information from month *t* or later.

## How it works

| Stage | Module | What it does |
|---|---|---|
| 1. Data acquisition | `src/descargar_datos_reales.py` | Downloads and caches FIRMS fires, NASA POWER daily climate, MODIS NDVI and land cover, Copernicus DEM terrain and OSM layers; assigns fires to districts |
| 2. Features | `src/preparar_datos.py`, `src/variables_diarias.py` | 28 predictors: weather of the previous month, dry/hot/windy-day counts from daily data, NDVI and anomalies, own and neighbouring-district fire history, seasonality, terrain |
| 3. Model | `src/entrenar_modelo.py` | Random forest (or gradient boosting); out-of-sample scores combined with the climatological baseline in a logistic calibration across years; rolling 2020–2025 evaluation |
| 4. Forecast and alerts | `src/predecir.py`, `src/alertas.py` | Risk levels (low < 5 %, medium 5–15 %, high 15–30 %, very high ≥ 30 %), short-term adjustment with 48-h active fires and the 7-day Open-Meteo forecast, reasons and ES/QU recommendations |
| 5. Dashboard | `dashboard/index.html` | Leaflet map, district cards, alert list, reports with model metrics, live weather |

All settings (paths, predictors, thresholds, alert rules, model type) are in
`src/config.py`. To use another region, replace `data/raw/puno_distritos.geojson`
with that region's district boundaries (fields `ubigeo`, `distrito`, `provincia`)
and adjust the bounding box `BBOX` in `src/descargar_datos_reales.py`.

## Data sources

| Variable | Source | Processing |
|---|---|---|
| Active fires | NASA FIRMS MODIS C6.1 | confidence ≥ 30, counted per district and month |
| Active fires, last 7 days | NASA FIRMS MODIS + VIIRS | short-term alert |
| Temperature, rainfall, humidity, wind | NASA POWER (daily) | representative point of each district |
| NDVI | MODIS MOD13Q1 250 m (ORNL DAAC) | 5 × 5 km window |
| Land cover | MODIS MCD12Q1 2024 (IGBP) | 13 × 13 km window |
| Elevation and slope | Copernicus DEM 90 m (via Open-Meteo) | approximate slope |
| Settlements, protected areas | OpenStreetMap (Overpass) | map layers |
| Weather forecast | Open-Meteo | live in the dashboard |

District boundaries are derived from *juaneladio/peru-geojson* (INEI-based, MPL 2.0;
see `data/raw/LICENCIA_LIMITES_DISTRITALES.txt`).

## Automatic monthly forecast

The workflow `.github/workflows/pronostico.yml` runs on the 3rd of every month (or on
demand from the *Actions* tab): it runs the tests, downloads the new data, retrains,
evaluates and forecasts, and publishes the dashboard and the CSV outputs to GitHub Pages.
To enable it in a fork: add the secret `FIRMS_MAP_KEY` (*Settings → Secrets and variables →
Actions*) and set *Settings → Pages → Source* to *GitHub Actions*.

## Citation

If you use this software, please cite it through the Zenodo DOI above or the
`CITATION.cff` file.

## License

Code: MIT (see `LICENSE`). District boundaries: MPL 2.0. Input data belong to their providers.
