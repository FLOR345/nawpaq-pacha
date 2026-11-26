"""
ÑAWPAQ PACHA — Capturas del panel para las figuras del artículo.

Genera, con el mismo tamaño y encuadre cada vez:
  docs/figuras/fig2_inicio.png        vista de inicio (mapa, alerta regional, zonas en riesgo)
  docs/figuras/fig3_ficha_juliaca.png ficha del distrito de Juliaca

Uso:
  pip install playwright
  python -m playwright install chromium
  python tools/capturas_articulo.py                      # panel publicado (con mapa base)
  python tools/capturas_articulo.py --url dashboard/index.html   # panel local
  python tools/capturas_articulo.py --idioma en          # recomendaciones en inglés
"""
import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "docs" / "figuras"
URL_WEB = "https://flor345.github.io/nawpaq-pacha/"
UBIGEO_JULIACA = "211101"


def resolver_url(url):
    if url.startswith("http"):
        return url.rstrip("#")
    ruta = (RAIZ / url).resolve() if not Path(url).is_absolute() else Path(url)
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. Genera el panel con: python main.py --datos-reales")
    return ruta.as_uri()


async def capturar(url, idioma, ancho, alto, escala):
    SALIDA.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        nav = await p.chromium.launch()
        ctx = await nav.new_context(viewport={"width": ancho, "height": alto},
                                    device_scale_factor=escala, color_scheme="light",
                                    locale="es-PE")
        # Idioma de las alertas y tema claro, guardados como lo hace el propio panel
        await ctx.add_init_script(
            f"try{{localStorage.setItem('nawpaq.lang','{idioma}');"
            f"localStorage.removeItem('nawpaq.theme')}}catch(e){{}}")
        pagina = await ctx.new_page()
        errores = []
        pagina.on("pageerror", lambda e: errores.append(str(e)))

        # Figura 2: vista de inicio
        await pagina.goto(url, wait_until="networkidle", timeout=120_000)
        await pagina.wait_for_timeout(4000)          # mapa base y condiciones en vivo
        destino = SALIDA / "fig2_inicio.png"
        await pagina.screenshot(path=str(destino))
        print("Guardado", destino.relative_to(RAIZ))

        # Figura 3: ficha de Juliaca (enlace directo #ficha=UBIGEO)
        await pagina.goto(f"{url}#ficha={UBIGEO_JULIACA}", wait_until="networkidle", timeout=120_000)
        await pagina.reload(wait_until="networkidle")
        await pagina.wait_for_timeout(3000)
        dialogo = pagina.locator("#dlg")
        if not await dialogo.evaluate("d => d.open"):
            raise SystemExit("No se abrió la ficha de Juliaca (¿cambió el ubigeo o el panel?)")
        # Mostrar la ficha completa, sin barra de desplazamiento
        await pagina.add_style_tag(content="#dlg{max-height:none!important;height:auto!important}"
                                           "#dlg *{max-height:none!important}")
        alto_ficha = await dialogo.evaluate("d => Math.ceil(d.scrollHeight) + 80")
        await pagina.set_viewport_size({"width": ancho, "height": max(alto, alto_ficha)})
        await pagina.wait_for_timeout(800)
        destino = SALIDA / "fig3_ficha_juliaca.png"
        await dialogo.screenshot(path=str(destino))
        print("Guardado", destino.relative_to(RAIZ))

        await nav.close()
        if errores:
            print("Avisos de JavaScript en el panel:", *errores, sep="\n  ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL_WEB, help="URL del panel o ruta a dashboard/index.html")
    ap.add_argument("--idioma", default="es", choices=["es", "qu", "en"])
    ap.add_argument("--ancho", type=int, default=1440)
    ap.add_argument("--alto", type=int, default=900)
    ap.add_argument("--escala", type=float, default=2, help="2 = imagen nítida para impresión")
    a = ap.parse_args()
    asyncio.run(capturar(resolver_url(a.url), a.idioma, a.ancho, a.alto, a.escala))


if __name__ == "__main__":
    main()
