# Cambios / Changelog

## [Sin publicar] v0.4 (en desarrollo)
- Nuevo `src/fuentes_nasa.py`: NDVI (MOD13A2), temperatura de superficie (MOD11A2), humedad del
  suelo (SMAP SPL3SMP_E), área quemada (MCD64A1) y cobertura anual (MCD12Q1) promediados sobre
  el polígono de cada distrito con NASA AppEEARS; focos VIIRS S-NPP 375 m desde 2015.
- `config.ETIQUETA`: la variable a pronosticar puede definirse con MODIS, VIIRS, ambos o MCD64A1.
- `ablacion.py`: configuraciones G (cada fuente NASA por separado) y
  `outputs/sensibilidad_etiqueta.json`.
- `python main.py --nasa`. El modelo por defecto no cambia (mismos resultados que v0.3).

## v0.3.1 (2026-11-20)
Revisión del artículo; el modelo y sus resultados no cambian.
- Intervalos de confianza por bootstrap **por bloques de distrito** (antes, distrito-meses sueltos).
- Importancia por permutación en los seis años de prueba y por grupos de variables correlacionadas.
- Tabla de umbrales con la tasa de falsas alarmas (meses sin incendio que recibieron alerta).
- `src/analisis_revision.py`: evaluación retrospectiva de la regla de 48 h y etiqueta de
  sensibilidad con focos MODIS de confianza alta (≥ 80).
- Panel: vuelve el aviso de borrador en las recomendaciones en quechua; Leaflet incluido en
  `dashboard/lib/` para abrir el panel sin internet.
- Incluye el código de fuentes NASA opcionales (`src/fuentes_nasa.py`, `python main.py --nasa`),
  desactivado por defecto (`USAR_VARIABLES_NASA = False`); ver la sección v0.4.

## v0.3.0 (2026-10-13)
- Random Forest por defecto, 28 variables, calibración multianual con la referencia
  climatológica, evaluación año por año 2020–2025, pronóstico mensual automático.
