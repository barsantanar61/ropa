#!/usr/bin/env python3
"""
quitar_fondo_ropa_mejorado.py

Elimina el fondo de fotografías de ropa colocadas sobre telas/sábanas.

La versión está pensada para fotos como:
    - prenda clara sobre fondo gris
    - prenda centrada
    - textura visible tanto en la prenda como en el fondo

No depende de APIs ni de modelos externos.
Requisitos:
    pip install opencv-python numpy pillow

Ejemplo:
    python quitar_fondo_ropa_mejorado.py -i foto.jpg -o resultado.png

Carpeta:
    python quitar_fondo_ropa_mejorado.py -i ./fotos -o ./salida

Para obtener solo la prenda recortada:
    --sin-estabilizar
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image


EXTENSIONES = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
)


# ------------------------------------------------------------
# CARGA
# ------------------------------------------------------------

def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")

    return img


# ------------------------------------------------------------
# MÁSCARA INICIAL
# ------------------------------------------------------------

def mascara_inicial_ropa(img):
    """
    Crea las semillas para GrabCut.

    La clave de esta versión es que NO intenta averiguar el fondo
    únicamente por su color medio. En una sábana con pliegues eso
    falla fácilmente.

    En cambio:
      - gris poco saturado -> fondo seguro
      - zonas claramente cromáticas/crema -> primer plano probable
      - zonas ambiguas -> GrabCut decide
      - bordes de la fotografía -> fondo seguro
    """

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 1. BORDES: fondo seguro
    # --------------------------------------------------------

    borde = max(
        8,
        int(min(h, w) * 0.025)
    )

    mask[:borde, :] = cv2.GC_BGD
    mask[-borde:, :] = cv2.GC_BGD
    mask[:, :borde] = cv2.GC_BGD
    mask[:, -borde:] = cv2.GC_BGD

    # --------------------------------------------------------
    # 2. FONDO GRIS: baja saturación
    #
    # No usamos simplemente S < 10 para toda la foto porque
    # algunas sombras y objetos oscuros pueden tener saturación
    # baja. Lo combinamos con un brillo razonable.
    # --------------------------------------------------------

    fondo_gris = (
        (saturation < 10)
        & (value > 65)
    )

    mask[fondo_gris] = cv2.GC_BGD

    # --------------------------------------------------------
    # 3. PRENDA PROBABLE
    #
    # Una prenda beige/crema tiene bastante más saturación
    # que la tela gris.
    # --------------------------------------------------------

    prenda_probable = (
        (saturation > 18)
        & (value > 80)
    )

    mask[prenda_probable] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # 4. PRENDA MUY SEGURA
    # --------------------------------------------------------

    prenda_segura = (
        (saturation > 30)
        & (value > 90)
    )

    mask[prenda_segura] = cv2.GC_FGD

    # --------------------------------------------------------
    # 5. RECTÁNGULO CENTRAL COMO INFORMACIÓN ADICIONAL
    #
    # Evita que GrabCut descarte zonas poco saturadas de la
    # propia prenda por sombras.
    # --------------------------------------------------------

    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    y0 = int(h * 0.10)
    y1 = int(h * 0.88)

    region = mask[y0:y1, x0:x1]

    # Dentro de la zona donde esperamos encontrar la ropa,
    # los píxeles ambiguos quedan como probable foreground,
    # pero mantenemos como fondo seguro los grises claros.
    ambiguos = (
        (region == cv2.GC_PR_BGD)
    )

    region[ambiguos] = cv2.GC_PR_FGD

    gris_interno = (
        (saturation[y0:y1, x0:x1] < 8)
        & (value[y0:y1, x0:x1] > 100)
    )

    region[gris_interno] = cv2.GC_PR_BGD

    mask[y0:y1, x0:x1] = region

    # Los bordes siguen siendo fondo seguro.
    mask[:borde, :] = cv2.GC_BGD
    mask[-borde:, :] = cv2.GC_BGD
    mask[:, :borde] = cv2.GC_BGD
    mask[:, -borde:] = cv2.GC_BGD

    return mask


# ------------------------------------------------------------
# GRABCUT
# ------------------------------------------------------------

def segmentar(img, iteraciones=10):
    """
    Ejecuta GrabCut con máscara inicial.

    Se usa la imagen original, no un desenfoque fuerte,
    para conservar el borde tejido de la ropa.
    """

    mask = mascara_inicial_ropa(img)

    modelo_fondo = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    modelo_frente = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Un pequeño suavizado bilateral ayuda con el ruido de la
    # textura sin destruir completamente el contorno.
    img_gc = cv2.bilateralFilter(
        img,
        7,
        35,
        35
    )

    cv2.grabCut(
        img_gc,
        mask,
        None,
        modelo_fondo,
        modelo_frente,
        iteraciones,
        cv2.GC_INIT_WITH_MASK
    )

    foreground = (
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD)
    )

    return foreground.astype(np.uint8)


# ------------------------------------------------------------
# COMPONENTES
# ------------------------------------------------------------

def componente_principal(mask):
    """
    Conserva el componente principal de la prenda.

    A diferencia del código anterior, NO rellena el contorno
    con drawContours(). Conservamos los píxeles reales de la
    segmentación para no volver a introducir fondo.
    """

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if n <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    return (labels == idx).astype(np.uint8)


def limpiar_mascara(mask):
    """
    Limpieza conservadora.
    """

    h, w = mask.shape

    k = max(
        3,
        int(min(h, w) * 0.003)
    )

    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    # Solo una operación de cada tipo:
    # demasiada morfología destruye mangas y bordes.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Eliminar componentes diminutos.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    area_min = h * w * 0.00015

    limpio = np.zeros_like(mask)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= area_min:
            limpio[labels == i] = 1

    # Finalmente, componente principal.
    limpio = componente_principal(limpio)

    return limpio


# ------------------------------------------------------------
# BORDE / ALFA
# ------------------------------------------------------------

def crear_alfa(mask, radio=2):
    """
    Feather muy pequeño.

    El código anterior utilizaba un GaussianBlur relativamente
    grande. En prendas tejidas eso puede hacer que el borde quede
    lavado. Aquí la transición se limita al contorno.
    """

    mask255 = (mask * 255).astype(np.uint8)

    interior = cv2.distanceTransform(
        mask255,
        cv2.DIST_L2,
        5
    )

    exterior = cv2.distanceTransform(
        255 - mask255,
        cv2.DIST_L2,
        5
    )

    signed = interior - exterior

    radio = max(1.0, float(radio))

    alpha = np.clip(
        128 + signed * (127 / radio),
        0,
        255
    ).astype(np.uint8)

    alpha[interior >= radio] = 255
    alpha[exterior >= radio] = 0

    return alpha


# ------------------------------------------------------------
# RECORTE
# ------------------------------------------------------------

def bounding_box(mask, padding=15):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)

    x1 = min(
        mask.shape[1] - 1,
        int(xs.max()) + padding
    )

    y1 = min(
        mask.shape[0] - 1,
        int(ys.max()) + padding
    )

    return (
        x0,
        y0,
        x1 - x0 + 1,
        y1 - y0 + 1
    )


# ------------------------------------------------------------
# RGBA
# ------------------------------------------------------------

def crear_rgba(img, alpha):

    b, g, r = cv2.split(img)

    rgba = cv2.merge(
        (r, g, b, alpha)
    )

    return Image.fromarray(
        rgba,
        mode="RGBA"
    )


# ------------------------------------------------------------
# LIENZO
# ------------------------------------------------------------

def estabilizar(imagen, size=(1000, 1300)):

    canvas = Image.new(
        "RGBA",
        size,
        (0, 0, 0, 0)
    )

    cw, ch = size
    iw, ih = imagen.size

    margen = 0.92

    escala = min(
        (cw * margen) / iw,
        (ch * margen) / ih
    )

    # Solo reducimos.
    if escala < 1:

        imagen = imagen.resize(
            (
                max(1, int(iw * escala)),
                max(1, int(ih * escala))
            ),
            Image.Resampling.LANCZOS
        )

    x = (cw - imagen.width) // 2
    y = (ch - imagen.height) // 2

    canvas.alpha_composite(
        imagen,
        (x, y)
    )

    return canvas


# ------------------------------------------------------------
# PROCESAR
# ------------------------------------------------------------

def procesar(
    entrada,
    salida,
    iteraciones=10,
    estabilizar_resultado=True,
    size=(1000, 1300)
):

    img = cargar_imagen(entrada)

    mask = segmentar(
        img,
        iteraciones=iteraciones
    )

    mask = limpiar_mascara(mask)

    if not np.any(mask):
        raise ValueError(
            "No se pudo detectar la prenda."
        )

    alpha = crear_alfa(
        mask,
        radio=2
    )

    x, y, w, h = bounding_box(
        mask,
        padding=15
    )

    img_crop = img[
        y:y+h,
        x:x+w
    ]

    alpha_crop = alpha[
        y:y+h,
        x:x+w
    ]

    resultado = crear_rgba(
        img_crop,
        alpha_crop
    )

    if estabilizar_resultado:
        resultado = estabilizar(
            resultado,
            size=size
        )

    resultado.save(
        salida,
        "PNG"
    )


# ------------------------------------------------------------
# ARCHIVOS
# ------------------------------------------------------------

def buscar_imagenes(ruta):

    if not os.path.isdir(ruta):
        return [ruta]

    archivos = []

    for ext in EXTENSIONES:
        archivos.extend(
            glob.glob(
                os.path.join(
                    ruta,
                    "*" + ext
                )
            )
        )

        archivos.extend(
            glob.glob(
                os.path.join(
                    ruta,
                    "*" + ext.upper()
                )
            )
        )

    return sorted(set(archivos))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extrae automáticamente ropa "
            "sobre fondos de tela."
        )
    )

    parser.add_argument(
        "-i",
        "--entrada",
        required=True
    )

    parser.add_argument(
        "-o",
        "--salida",
        required=True
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=10
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[1000, 1300],
        metavar=("ANCHO", "ALTO")
    )

    parser.add_argument(
        "--sin-estabilizar",
        action="store_true"
    )

    args = parser.parse_args()

    imagenes = buscar_imagenes(
        args.entrada
    )

    if not imagenes:
        print("No se encontraron imágenes.")
        sys.exit(1)

    carpeta = os.path.isdir(
        args.entrada
    )

    if carpeta:
        os.makedirs(
            args.salida,
            exist_ok=True
        )

    for entrada in imagenes:

        nombre = os.path.splitext(
            os.path.basename(entrada)
        )[0]

        if carpeta:
            salida = os.path.join(
                args.salida,
                nombre + "_sin_fondo.png"
            )
        else:
            salida = args.salida

        try:

            procesar(
                entrada,
                salida,
                iteraciones=args.iter,
                estabilizar_resultado=(
                    not args.sin_estabilizar
                ),
                size=tuple(args.size)
            )

            print(
                f"OK -> {salida}"
            )

        except Exception as e:

            print(
                f"ERROR -> {entrada}: {e}"
            )


if __name__ == "__main__":
    main()
