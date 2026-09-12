#!/usr/bin/env python3
"""
quitar_fondo_ropa_mejorado_v5.py

Elimina sombras proyectadas duras en la tela e impide que queden trozos de fondo entre las piernas.
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
    print("Instálalas ejecutando: pip install rembg pillow opencv-python numpy\n")
    sys.exit(1)

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def eliminar_sombra_dura(img_bgr, alpha_mask):
    """
    Elimina sombras proyectadas dividiendo la luminancia original 
    por su mapa de iluminación estimado (Gaussian Blur grande).
    """
    # Convertir a flotante
    img_float = img_bgr.astype(np.float32) / 255.0

    # Separar canales B, G, R
    channels = cv2.split(img_float)
    channels_corrected = []

    # Radio amplio para capturar la forma de la sombra diagonal
    h, w = img_bgr.shape[:2]
    ksize = int(min(h, w) * 0.25)
    if ksize % 2 == 0:
        ksize += 1

    for ch in channels:
        # Estimación de la luz ambiente
        bg_illum = cv2.GaussianBlur(ch, (ksize, ksize), 0)
        
        # Evitar división por cero
        bg_illum[bg_illum < 0.01] = 0.01

        # Dividir la imagen por el mapa de luz y normalizar
        ch_corr = ch / bg_illum
        mean_val = np.mean(ch[alpha_mask > 128]) if np.any(alpha_mask > 128) else 0.5
        ch_corr = ch_corr * mean_val

        channels_corrected.append(ch_corr)

    # Fusionar canales y reescalar a uint8
    img_corr = cv2.merge(channels_corrected)
    img_corr = np.clip(img_corr * 255.0, 0, 255).astype(np.uint8)

    # Aplicar la corrección únicamente sobre la zona de la prenda
    alpha_3d = (alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    resultado = (img_corr * alpha_3d + img_bgr * (1.0 - alpha_3d)).astype(np.uint8)

    return resultado


def limpiar_mascara(alpha_mask):
    """
    Elimina artefactos pequeños de fondo (como la sábana colada entre las piernas).
    """
    binaria = (alpha_mask > 100).astype(np.uint8) * 255

    # Conectar componentes grandes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:
        return alpha_mask

    # Conservar únicamente el área/componente principal
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_max = 1 + int(np.argmax(areas))

    mask_limpia = np.zeros_like(binaria)
    mask_limpia[labels == idx_max] = 255

    # Aplicar un pequeño suavizado de bordes
    mask_limpia = cv2.GaussianBlur(mask_limpia, (3, 3), 0)

    return mask_limpia


def estabilizar_lienzo(imagen_rgba, tamano_lienzo=(1000, 1300), margen_porcentaje=0.90, fondo_blanco=False):
    cw, ch = tamano_lienzo
    iw, ih = imagen_rgba.size

    if fondo_blanco:
        canvas = Image.new("RGBA", tamano_lienzo, (255, 255, 255, 255))
    else:
        canvas = Image.new("RGBA", tamano_lienzo, (0, 0, 0, 0))

    escala = min((cw * margen_porcentaje) / iw, (ch * margen_porcentaje) / ih)
    nuevo_ancho = max(1, int(iw * escala))
    nuevo_alto = max(1, int(ih * escala))

    imagen_resaltada = imagen_rgba.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    pos_x = (cw - nuevo_ancho) // 2
    pos_y = (ch - nuevo_alto) // 2

    canvas.alpha_composite(imagen_resaltada, (pos_x, pos_y))
    return canvas


def procesar_imagen(ruta_entrada, carpeta_destino, nombre_base, session, tamano_lienzo=(1000, 1300)):
    # 1. Cargar imagen
    img_pil = Image.open(ruta_entrada).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 2. IA Rembg
    resultado_rgba = remove(img_pil, session=session)
    arr_rgba = np.array(resultado_rgba)
    alpha_raw = arr_rgba[:, :, 3]

    # 3. Limpiar máscara para eliminar la sábana entre piernas
    alpha_mask = limpiar_mascara(alpha_raw)

    # 4. Eliminar la sombra diagonal dura
    bgr_sin_sombra = eliminar_sombra_dura(img_bgr, alpha_mask)
    rgb_sin_sombra = cv2.cvtColor(bgr_sin_sombra, cv2.COLOR_BGR2RGB)

    # 5. Reconstruir imagen corregida
    rgba_corregido = np.dstack((rgb_sin_sombra, alpha_mask))
    img_corregida = Image.fromarray(rgba_corregido, mode="RGBA")

    # 6. Auto-recorte
    bbox = img_corregida.getbbox()
    if bbox:
        img_recortada = img_corregida.crop(bbox)
    else:
        raise ValueError("No se pudo detectar ninguna prenda.")

    # 7. Guardar Transparente
    salida_transparente = os.path.join(carpeta_destino, f"{nombre_base}_transparente.png")
    lienzo_transparente = estabilizar_lienzo(img_recortada, tamano_lienzo=tamano_lienzo, fondo_blanco=False)
    lienzo_transparente.save(salida_transparente, "PNG")

    # 8. Guardar Blanco
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
    parser = argparse.ArgumentParser(description="Procesado avanzado de ropa sin sombras.")
    parser.add_argument("-i", "--entrada", required=True)
    parser.add_argument("-o", "--salida", required=True)
    args = parser.parse_args()

    imagenes = buscar_imagenes(args.entrada)
    if not imagenes:
        print(f"No hay imágenes en: {args.entrada}")
        sys.exit(1)

    os.makedirs(args.salida, exist_ok=True)
    session = new_session("u2net")

    for entrada in imagenes:
        nombre_base = os.path.splitext(os.path.basename(entrada))[0]
        try:
            print(f"Procesando: {os.path.basename(entrada)} ...", end=" ", flush=True)
            procesar_imagen(entrada, args.salida, nombre_base, session)
            print("[OK]")
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()