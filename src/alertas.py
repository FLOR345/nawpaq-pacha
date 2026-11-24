"""
ÑAWPAQ PACHA — Textos de alertas y recomendaciones (español, quechua e inglés).

Los textos en quechua (collao) fueron validados con hablantes locales en
noviembre de 2026. Si se cambia o agrega un texto en quechua, vuelve a
validarlo y pon QUECHUA_VALIDADO = True mientras tanto: el panel mostrará
el aviso de borrador.
El inglés es para lectores externos (revisores, organismos internacionales).
"""

QUECHUA_VALIDADO = True

NIVEL_QU = {"Bajo": "Pisi", "Medio": "Chawpi", "Alto": "Hatun", "Muy alto": "Ancha hatun"}
NIVEL_EN = {"Bajo": "Low", "Medio": "Medium", "Alto": "High", "Muy alto": "Very high"}

RECOMENDACIONES = {
    "Muy alto": {
        "es": ["Evita toda quema de pastizales, ichu y rastrojos.",
               "Coordina la vigilancia con autoridades y brigadas comunales.",
               "Protege ganado y cultivos; ten agua y herramientas a la mano.",
               "Si ves humo o fuego, llama al 116 (Bomberos) y avisa a tu autoridad."],
        "qu": ["Ama ichhuta, chakra puchuqkunata kanaychu.",
               "Kamachikuqkunawan, ayllu brigadakunawan qhawariy.",
               "Uywaykita, chakraykita waqaychay; yakuta, llamk'anakunata wakichiy.",
               "Q'usnita ninatapas rikuspaqa, 116 yupayman waqyay."],
        "en": ["Avoid any burning of grassland, ichu grass or crop residues.",
               "Coordinate surveillance with local authorities and community brigades.",
               "Protect livestock and crops; keep water and tools at hand.",
               "If you see smoke or fire, call 116 (Fire Department) and notify your local authority."],
    },
    "Alto": {
        "es": ["Posterga las quemas agrícolas hasta que bajen las condiciones de riesgo.",
               "Limpia de pasto seco los alrededores de viviendas y corrales.",
               "Mantente atento a los avisos de tu municipalidad."],
        "qu": ["Ama kay p'unchaykunapi chakrata kanaychu.",
               "Wasiyki, kanchayki muyuriqta ch'aki q'achuta pichay.",
               "Llaqta kamachikuqpa willakuyninkunata uyariy."],
        "en": ["Postpone agricultural burning until risk conditions decrease.",
               "Clear dry grass around homes and livestock pens.",
               "Stay alert to warnings from your municipality."],
    },
    "Medio": {
        "es": ["Vigila las quemas y no las dejes sin control.",
               "Revisa los cortafuegos alrededor de viviendas y corrales."],
        "qu": ["Kanaykunata qhawariy, ama sapallanta saqiychu.",
               "Wasiyki muyuriqpi nina harkaykunata qhawariy."],
        "en": ["Supervise any burning and never leave it unattended.",
               "Check the firebreaks around homes and livestock pens."],
    },
    "Bajo": {
        "es": ["Condiciones de riesgo bajas. Mantén las buenas prácticas."],
        "qu": ["Kunanqa pisi riesgo kachkan. Allin ruraykunata hinalla ruray."],
        "en": ["Low risk conditions. Keep up good practices."],
    },
}

TITULO = {"es": "Alerta orientativa de incendio", "qu": "Ninamanta willakuy",
          "en": "Indicative fire alert"}
AVISO = {
    "es": "Estimación orientativa: no afirma que un incendio ocurrirá. "
          "Debe revisarse con el criterio de autoridades y comunidades.",
    "qu": "Kayqa yanapaq willakuylla, manam nichkanchu nina kanqapuni nispa.",
    "en": "Indicative estimate: it does not state that a fire will occur. "
          "It must be reviewed with the judgement of authorities and communities.",
}


def explicar(fila, normal):
    """Devuelve hasta 3 razones en lenguaje simple de por qué el riesgo es ese."""
    r = []
    if normal is not None:
        if normal.precip_3m > 5 and fila.precip_3m < 0.7 * normal.precip_3m:
            r.append(f"Lluvia de los últimos 3 meses {100 * (1 - fila.precip_3m / normal.precip_3m):.0f}% por debajo de lo normal")
        if fila.humedad_lag1 < normal.humedad_lag1 - 4:
            r.append(f"Aire más seco de lo normal ({fila.humedad_lag1:.0f}% de humedad)")
        if fila.temp_max_lag1 > normal.temp_max_lag1 + 0.8:
            r.append(f"Temperatura máxima {fila.temp_max_lag1 - normal.temp_max_lag1:.1f} °C sobre lo normal")
        if fila.ndvi_lag1 < normal.ndvi_lag1 - 0.03:
            r.append("Vegetación más seca de lo normal (satélite)")
        if "dias_secos_lag1" in normal and fila.dias_secos_lag1 >= normal.dias_secos_lag1 + 4:
            r.append(f"{int(fila.dias_secos_lag1)} días sin lluvia el mes pasado "
                     f"(lo normal son {normal.dias_secos_lag1:.0f})")
        if fila.viento_lag1 > normal.viento_lag1 + 2:
            r.append(f"Vientos más fuertes de lo normal ({fila.viento_lag1:.0f} km/h)")
    if getattr(fila, "dias_peligro_lag1", 0) >= 3:
        r.append(f"{int(fila.dias_peligro_lag1)} días secos, de aire seco y con viento el mes pasado")
    if fila.focos_lag1 > 0:
        r.append(f"{int(fila.focos_lag1)} foco(s) de calor detectados el mes pasado")
    if getattr(fila, "vec_focos_lag1", 0) >= 3:
        r.append(f"{int(fila.vec_focos_lag1)} foco(s) en distritos vecinos el mes pasado")
    if fila.frec_hist_mes >= 0.4:
        r.append(f"Historial: hubo incendios en este mes en {round(fila.frec_hist_mes * 10)} de cada 10 años")
    if fila.pct_pastizal >= 55:
        r.append(f"Alta cobertura de pastizales ({fila.pct_pastizal:.0f}% del distrito)")
    return r[:3] if r else ["Condiciones similares a lo normal para la época"]


# --- Alerta de corto plazo (ver PELIGRO_TIEMPO en config.py) -----------------
# La misma regla está escrita en dashboard/index.html (función peligroDia) para
# recalcular la alerta en vivo; si cambias una, cambia la otra.
def peligro_dia(hum_min, viento_max, lluvia, regla):
    if lluvia is None or hum_min is None or viento_max is None:
        return "Bajo"
    if lluvia >= regla["lluvia_anula"]:
        return "Bajo"
    pts = (hum_min < regla["humedad_min"][0]) + (hum_min < regla["humedad_min"][1])
    pts += (viento_max >= regla["viento_max"][0]) + (viento_max >= regla["viento_max"][1])
    pts += lluvia < regla["lluvia_seca"]
    return regla["niveles"][sum(pts >= c for c in regla["cortes"])]


def nivel_final(nivel_modelo, peligros_3d, focos_48h, niveles):
    """Combina el riesgo mensual del modelo con focos activos y el tiempo de los próximos días."""
    i = niveles.index(nivel_modelo)
    motivos = []
    if i >= 1 and sum(p in ("Alto", "Extremo") for p in peligros_3d) >= 2:
        i = min(i + 1, len(niveles) - 1)
        motivos.append("Pronóstico: aire seco y viento fuerte en los próximos 3 días")
    if focos_48h > 0 and i < 2:
        i = 2
    if focos_48h > 0:
        motivos.append(f"{focos_48h} foco(s) activo(s) detectado(s) por satélite en las últimas 48 h")
    return niveles[i], motivos
