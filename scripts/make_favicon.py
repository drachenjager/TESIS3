"""Genera favicon.ico + favicon.png a partir del diseño UNAM.

Ejecutar una sola vez desde la raíz del proyecto:
    python scripts/make_favicon.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

STATIC = Path(__file__).parent.parent / "app" / "static"

UNAM_BLUE = (0, 48, 135, 255)
UNAM_GOLD = (200, 168, 0, 255)
UNAM_GOLD_FAINT = (200, 168, 0, 89)


def _draw_favicon(size: int) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fondo azul con esquinas redondeadas
    r = max(1, round(6 * s / 32))
    draw.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=UNAM_BLUE)

    # Escala coordenadas SVG (diseñadas en 32x32) al tamaño actual
    def sc(v):
        return round(v * s / 32)

    # Línea de tendencia: 4,25 → 9,19 → 14,21 → 20,12 → 28,7
    pts = [(sc(4), sc(25)), (sc(9), sc(19)), (sc(14), sc(21)), (sc(20), sc(12)), (sc(28), sc(7))]
    lw = max(1, round(2.8 * s / 32))
    draw.line(pts, fill=UNAM_GOLD, width=lw)

    # Punto de pronóstico (círculo dorado)
    cr = max(1, round(2.8 * s / 32))
    cx, cy = sc(28), sc(7)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=UNAM_GOLD)

    # Eje horizontal sutil
    draw.line([(sc(4), sc(27)), (sc(28), sc(27))], fill=UNAM_GOLD_FAINT, width=1)

    return img


if __name__ == "__main__":
    img32 = _draw_favicon(32)
    img16 = _draw_favicon(16)

    # favicon.ico con ambos tamaños embebidos
    ico_path = STATIC / "favicon.ico"
    img32.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32)], append_images=[img16])
    print(f"Generado: {ico_path}")

    # favicon.png (32x32, respaldo adicional)
    png_path = STATIC / "favicon.png"
    img32.save(png_path, format="PNG")
    print(f"Generado: {png_path}")
