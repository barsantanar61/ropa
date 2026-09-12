#!/usr/bin/env python3
"""
quitar_fondo_ropa_mejorado_v3.py

1. Elimina el fondo con Inteligencia Artificial (rembg / u2net).
2. Corrije automáticamente la iluminación y atenúa sombras duras (CLAHE en espacio LAB).
3. Genera versiones en fondo transparente y fondo blanco de catálogo.

Requisitos:
    pip install rembg pillow opencv-python numpy
"""

import argparse
import glob
import os
import sys
import numpy as np

try:
    import cv2
    from PIL import Image
    from rembg import remove, new_session
except ImportError:
    print("\n[ERROR] Faltan librerías requeridas.")
    print("Instálalas ejecutando en CMD:")
    print("    pip install rembg pillow opencv-python numpy\n")
    sys.exit(1)

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def corregir_iluminacion_y_color(img_bgr, alpha_mask):
    """
    Equilibrate la luz y atenúa sombras duras en la prenda usando CLAHE en espacio de color LAB.
    """
    # Convertir BGR a LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE para suavizar gradientes de sombras y mejorar el contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corregido = clahe.apply(l_channel)

    # Recombinar canales
    lab_corregido = cv2.merge((l_corregido, a_channel, b_channel))
    bgr_corregido = cv2.cvtColor(lab_corregido, cv2.COLOR_LAB2BGR)

    # Mezclar resultado según la máscara de transparencia
    alpha_3d = (alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    resultado_bgr = (bgr_corregido * alpha_3d + img_bgr * (1.0 - alpha_3d)).astype(np.uint8)

    return resultado_bgr


def estabilizar_lienzo(imagen_rgba, tamano_lienzo=(1000, 1300), margen_porcentaje=0.90, fondo_blanco=False):
    """
    Centra la prenda en un lienzo estándar (1000x1300).
    """
    cw, ch = tamano_lienzo
    iw, ih = imagen_rgba.size

    # Crear lienzo base
    if fondo_blanco:
        canvas = Image.new("RGBA", tamano_lienzo, (255, 255, 255, 255))
    else:
        canvas = Image.new("RGBA", tamano_lienzo, (0, 0, 0, 0))

    # Escalar manteniendo proporción
    escala = min((cw * margen_porcentaje) / iw, (ch * margen_porcentaje) / ih)
    nuevo_ancho = max(1, int(iw * escala))
    nuevo_alto = max(1, int(ih * escala))

    imagen_resaltada = imagen_rgba.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    pos_x = (cw - nuevo_ancho) // 2
    pos_y = (ch - nuevo_alto) // 2

    canvas.alpha_composite(imagen_resaltada, (pos_x, pos_y))
    return canvas


def procesar_imagen(ruta_entrada, carpeta_destino, nombre_base, session, tamano_lienzo=(1000, 1300)):
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No existe el archivo: {ruta_entrada}")

    # 1. Cargar imagen original
    img_pil = Image.open(ruta_entrada).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 2. Extracción de fondo con IA
    resultado_rgba = remove(img_pil, session=session)
    arr_rgba = np.array(resultado_rgba)
    alpha_mask = arr_rgba[:, :, 3]

    # 3. Corrección de iluminación/sombras en la prenda
    bgr_corregido = corregir_iluminacion_y_color(img_bgr, alpha_mask)
    rgb_corregido = cv2.cvtColor(bgr_corregido, cv2.COLOR_BGR2RGB)

    # Unir canales corregidos con el canal Alfa
    rgba_corregido = np.dstack((rgb_corregido, alpha_mask))
    img_corregida = Image.fromarray(rgba_corregido, mode="RGBA")

    # 4. Auto-recorte de transparencias vacías
    bbox = img_corregida.getbbox()
    if bbox:
        img_recortada = img_corregida.crop(bbox)
    else:
        raise ValueError("No se pudo detectar ninguna prenda en la imagen.")

    # 5. Guardar versión con Fondo Transparente
    salida_transparente = os.path.join(carpeta_destino, f"{nombre_base}_transparente.png")
    lienzo_transparente = estabilizar_lienzo(img_recortada, tamano_lienzo=tamano_lienzo, fondo_blanco=False)
    lienzo_transparente.save(salida_transparente, "PNG")

    # 6. Guardar versión con Fondo Blanco
    salida_blanco = os.path.join(carpeta_destino, f"{nombre_base}_blanco.png")
    lienzo_blanco = estabilizar_lienzo(img_recortada, tamano_lienzo=tamano_lienzo, fondo_blanco=True)
    lienzo_blanco.convert("RGB").save(salida_blanco, "JPEG", quality=95)


def buscar_imagenes(ruta_entrada):
    if not os.path.isdir(ruta_entrada):
        return [ruta_entrada]

    archivos = []
    for ext in EXTENSIONES:
        archivos.extend(glob.glob(os.path.join(ruta_entrada, "*" + ext)))
        archivos.extend(glob.glob(os.path.join(ruta_entrada, "*" + ext.upper())))
    return sorted(set(archivos))


def main():
    parser = argparse.ArgumentParser(description="Procesamiento completo de ropa: IA + Corrección de Luz/Sombras.")
    parser.add_argument("-i", "--entrada", required=True, help="Imagen de entrada o carpeta")
    parser.add_argument("-o", "--salida", required=True, help="Carpeta de destino para los resultados")
    args = parser.parse_args()

    imagenes = buscar_imagenes(args.entrada)
    if not imagenes:
        print(f"No se encontraron imágenes válidas en: {args.entrada}")
        sys.exit(1)

    os.makedirs(args.salida, exist_ok=True)

    print("Cargando modelo de IA...")
    session = new_session("u2net")

    for entrada in imagenes:
        nombre_base = os.path.splitext(os.path.basename(entrada))[0]
        try:
            print(f"Procesando: {os.path.basename(entrada)} ...", end=" ", flush=True)
            procesar_imagen(
                ruta_entrada=entrada,
                carpeta_destino=args.salida,
                nombre_base=nombre_base,
                session=session
            )
            print("[OK]")
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()