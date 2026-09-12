#!/usr/bin/env python3
"""
quitar_fondo_ropa_mejorado_v2.py

Versión optimizada para detectar prendas blancas/claras sobre fondos claros.
Utiliza detección de bordes y relleno de contornos en lugar de solo color/saturación.
"""

import argparse
import glob
import os
import sys

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Faltan dependencias. {e}")
    print("Ejecuta: pip install opencv-python numpy pillow")
    sys.exit(1)

EXTENSIONES = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
)

def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return img

def segmentar_prenda_clara(img):
    """
    Enfoque robusto para prendas claras sobre fondos claros:
    Utiliza detección de bordes Canny para definir la silueta física,
    luego la rellena morfológicamente.
    """
    h, w = img.shape[:2]

    # 1. Convertir a Gris y Suavizar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Detección de Bordes Canny
    # El umbral bajo (30) es clave para detectar la camiseta blanca
    edged = cv2.Canny(blurred, 30, 100)

    # 3. Dilatación Morfológica: Unir bordes
    # Un kernel grande ayuda a cerrar la silueta
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edged, kernel_dilate, iterations=2)

    # 4. Operación de Cierre Morfológico
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 5. Relleno de Contornos (Flood Fill)
    # Rellenar agujeros interiores para asegurar que el cuerpo de la camiseta se mantiene.
    flood_fill_mask = closed.copy()
    h_ff, w_ff = flood_fill_mask.shape[:2]
    # Crear una máscara para floodFill, necesita ser 2px más ancha y alta
    ff_mask = np.zeros((h_ff + 2, w_ff + 2), np.uint8)
    
    # Rellenar desde las esquinas (se asume fondo)
    pts_fondo = [(0, 0), (w_ff - 1, 0), (0, h_ff - 1), (w_ff - 1, h_ff - 1)]
    for pt in pts_fondo:
        # Asegurarse de que el punto inicial es 0 (negro/fondo) antes de rellenar
        if flood_fill_mask[pt[1], pt[0]] == 0:
             cv2.floodFill(flood_fill_mask, ff_mask, pt, 255)

    # Invertir el flood fill para obtener los agujeros interiores
    inv_flood_fill_mask = cv2.bitwise_not(flood_fill_mask)

    # Combinar la imagen cerrada original con los agujeros rellenos
    filled_mask = closed | inv_flood_fill_mask

    # 6. Forzar Fondo en Bordes (GrabCut Style)
    borde = max(8, int(min(h, w) * 0.02))
    filled_mask[:borde, :] = 0
    filled_mask[-borde:, :] = 0
    filled_mask[:, :borde] = 0
    filled_mask[:, -borde:] = 0

    return filled_mask

def componente_principal(mask):
    """
    Conserva solo el componente más grande (la prenda).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)

def crear_alfa(mask, radio=2):
    """
    Feathering muy sutil para bordes limpios pero realistas.
    """
    mask255 = (mask * 255).astype(np.uint8)
    
    # Si la máscara es muy pequeña o está vacía
    if not np.any(mask255):
        return np.zeros_like(mask255)

    # Calcular la distancia transform
    interior = cv2.distanceTransform(mask255, cv2.DIST_L2, 5)
    exterior = cv2.distanceTransform(255 - mask255, cv2.DIST_L2, 5)

    # Crear el alpha signed
    signed = interior - exterior
    radio = max(1.0, float(radio))

    alpha = np.clip(128 + signed * (127 / radio), 0, 255).astype(np.uint8)
    alpha[interior >= radio] = 255
    alpha[exterior >= radio] = 0
    return alpha

def bounding_box(mask, padding=15):
    """
    Calcula el recorte ajustado.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + padding)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + padding)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1

def crear_rgba(img, alpha):
    """
    Combina la imagen original y la máscara alfa en un PNG.
    """
    # OpenCV usa BGR, PIL usa RGBA (Red, Green, Blue, Alpha)
    b, g, r = cv2.split(img)
    rgba = cv2.merge((r, g, b, alpha))
    return Image.fromarray(rgba, mode="RGBA")

def estabilizar(imagen, size=(1000, 1300)):
    """
    Coloca la prenda recortada en el centro de un lienzo transparente estándar.
    """
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    cw, ch = size
    iw, ih = imagen.size
    margen = 0.92
    escala = min((cw * margen) / iw, (ch * margen) / ih)
    
    if escala < 1:
        imagen = imagen.resize(
            (max(1, int(iw * escala)), max(1, int(ih * escala))),
            Image.Resampling.LANCZOS
        )

    x = (cw - imagen.width) // 2
    y = (ch - imagen.height) // 2
    canvas.alpha_composite(imagen, (x, y))
    return canvas

def procesar(entrada, salida, estabilizar_resultado=True, size=(1000, 1300)):
    """
    Flujo principal de procesamiento.
    """
    img = cargar_imagen(entrada)
    
    # NUEVA SEGMENTACIÓN
    mask = segmentar_prenda_clara(img)
    
    # Conservar componente principal
    mask = componente_principal(mask)
    
    if not np.any(mask):
        raise ValueError("No se pudo detectar la prenda clara en la imagen.")

    # Crear Alfa y Recorte
    alpha = crear_alfa(mask, radio=2)
    x, y, w, h = bounding_box(mask, padding=15)
    
    img_crop = img[y:y+h, x:x+w]
    alpha_crop = alpha[y:y+h, x:x+w]
    
    resultado = crear_rgba(img_crop, alpha_crop)

    if estabilizar_resultado:
        resultado = estabilizar(resultado, size=size)

    # Guardar
    resultado.save(salida, "PNG")

def buscar_imagenes(ruta):
    """
    Busca imágenes compatibles en la ruta o carpeta dada.
    """
    if not os.path.isdir(ruta):
        return [ruta]
    archivos = []
    for ext in EXTENSIONES:
        archivos.extend(glob.glob(os.path.join(ruta, "*" + ext)))
        archivos.extend(glob.glob(os.path.join(ruta, "*" + ext.upper())))
    return sorted(set(archivos))

def main():
    parser = argparse.ArgumentParser(description="Extrae ropa clara sobre fondos claros (versión v2 Edge-Detection)")
    parser.add_argument("-i", "--entrada", required=True, help="Imagen de entrada o carpeta")
    parser.add_argument("-o", "--salida", required=True, help="Imagen PNG de salida o carpeta")
    parser.add_argument("--size", type=int, nargs=2, default=[1000, 1300], metavar=("ANCHO", "ALTO"))
    parser.add_argument("--sin-estabilizar", action="store_true", help="Obtener solo el recorte ajustado")
    args = parser.parse_args()

    imagenes = buscar_imagenes(args.entrada)
    if not imagenes:
        print("No se encontraron imágenes.")
        sys.exit(1)

    carpeta_salida = os.path.isdir(args.salida)
    if carpeta_salida:
        os.makedirs(args.salida, exist_ok=True)
    elif len(imagenes) > 1 and not carpeta_salida:
        # Si se procesan varias imágenes pero la salida es un archivo, forzar carpeta
        os.makedirs(args.salida, exist_ok=True)
        carpeta_salida = True

    for entrada in imagenes:
        nombre = os.path.splitext(os.path.basename(entrada))[0]
        if carpeta_salida:
            salida = os.path.join(args.salida, nombre + "_sin_fondo.png")
        else:
            salida = args.salida

        try:
            procesar(
                entrada,
                salida,
                estabilizar_resultado=(not args.sin_estabilizar),
                size=tuple(args.size)
            )
            print(f"OK -> {salida}")
        except Exception as e:
            print(f"ERROR -> {entrada}: {e}")

if __name__ == "__main__":
    main()