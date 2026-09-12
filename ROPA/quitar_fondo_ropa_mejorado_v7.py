#!/usr/bin/env python3
"""
quitar_fondo_ropa_mejorado_v7.py

- Elimina fondos y preserva prendas completas sin recortar estampados.
- Elimina sombras proyectadas duras y trozos de fondo entre extremidades.
- Genera versión en fondo transparente y fondo blanco de catálogo.
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


def asegurar_prenda_completa(img_bgr, alpha_ia):
    """
    Verifica que la máscara de la IA no haya seleccionado solo un estampado interior.
    Si la máscara es muy pequeña comparada con el área de la prenda, usa umbralizado de color.
    """
    h, w = img_bgr.shape[:2]
    area_total = h * w
    area_ia = np.sum(alpha_ia > 100)

    # Si la IA recortó menos del 10% de la imagen, es probable que solo tomara el dibujo
    if area_ia < (area_total * 0.08):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Buscar la prenda por diferencia de contraste con el fondo
        _, mask_auto = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_auto = cv2.morphologyEx(mask_auto, cv2.MORPH_CLOSE, kernel)
        
        # Combinar la máscara calculada con la IA
        alpha_final = cv2.bitwise_or(alpha_ia, mask_auto)
        return alpha_final

    return alpha_ia


def perfeccionar_mascara(alpha_mask):
    """
    Elimina artefactos sueltos, realiza erosión fina en bordes y quita sombras entre piernas.
    """
    _, binaria = cv2.threshold(alpha_mask, 40, 255, cv2.THRESH_BINARY)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_max = 1 + int(np.argmax(areas))
        mask_principal = np.zeros_like(binaria)
        mask_principal[labels == idx_max] = 255
    else:
        mask_principal = binaria

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_erosionada = cv2.erode(mask_principal, kernel, iterations=1)
    mask_suave = cv2.GaussianBlur(mask_erosionada, (3, 3), 0)

    return cv2.bitwise_and(alpha_mask, mask_suave)


def eliminar_sombra_dura(img_bgr, alpha_mask):
    """
    Iguala la iluminación en la prenda eliminando sombras proyectadas duras.
    """
    img_float = img_bgr.astype(np.float32) / 255.0
    channels = cv2.split(img_float)
    channels_corrected = []

    h, w = img_bgr.shape[:2]
    ksize = int(min(h, w) * 0.25)
    if ksize % 2 == 0:
        ksize += 1

    for ch in channels:
        bg_illum = cv2.GaussianBlur(ch, (ksize, ksize), 0)
        bg_illum[bg_illum < 0.01] = 0.01

        ch_corr = ch / bg_illum
        mean_val = np.mean(ch[alpha_mask > 128]) if np.any(alpha_mask > 128) else 0.5
        ch_corr = ch_corr * mean_val

        channels_corrected.append(ch_corr)

    img_corr = cv2.merge(channels_corrected)
    img_corr = np.clip(img_corr * 255.0, 0, 255).astype(np.uint8)

    alpha_3d = (alpha_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    return (img_corr * alpha_3d + img_bgr * (1.0 - alpha_3d)).astype(np.uint8)


def estabilizar_lienzo(imagen_rgba, tamano_lienzo=(1000, 1300), margen_porcentaje=0.88, fondo_blanco=False):
    cw, ch = tamano_lienzo
    iw, ih = imagen_rgba.size

    canvas = Image.new("RGBA", tamano_lienzo, (255, 255, 255, 255) if fondo_blanco else (0, 0, 0, 0))

    escala = min((cw * margen_porcentaje) / iw, (ch * margen_porcentaje) / ih)
    nuevo_ancho = max(1, int(iw * escala))
    nuevo_alto = max(1, int(ih * escala))

    imagen_resaltada = imagen_rgba.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    pos_x = (cw - nuevo_ancho) // 2
    pos_y = (ch - nuevo_alto) // 2

    canvas.alpha_composite(imagen_resaltada, (pos_x, pos_y))
    return canvas


def procesar_imagen(ruta_entrada, carpeta_destino, nombre_base, session, tamano_lienzo=(1000, 1300)):
    img_pil = Image.open(ruta_entrada).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 1. Extracción con rembg
    resultado_rgba = remove(img_pil, session=session)
    arr_rgba = np.array(resultado_rgba)
    alpha_raw = arr_rgba[:, :, 3]

    # 2. Validar que la prenda entera esté incluida
    alpha_completo = asegurar_prenda_completa(img_bgr, alpha_raw)

    # 3. Limpieza de bordes y sombras sueltas
    alpha_mask = perfeccionar_mascara(alpha_completo)

    # 4. Corrección de iluminación
    bgr_sin_sombra = eliminar_sombra_dura(img_bgr, alpha_mask)
    rgb_sin_sombra = cv2.cvtColor(bgr_sin_sombra, cv2.COLOR_BGR2RGB)

    # 5. Generar RGBA final
    rgba_corregido = np.dstack((rgb_sin_sombra, alpha_mask))
    img_corregida = Image.fromarray(rgba_corregido, mode="RGBA")

    # 6. Recorte automático
    bbox = img_corregida.getbbox()
    if bbox:
        img_recortada = img_corregida.crop(bbox)
    else:
        raise ValueError("No se pudo detectar la prenda en la imagen.")

    # 7. Guardar resultados
    salida_transparente = os.path.join(carpeta_destino, f"{nombre_base}_transparente.png")
    lienzo_transparente = estabilizar_lienzo(img_recortada, tamano_lienzo=tamano_lienzo, fondo_blanco=False)
    lienzo_transparente.save(salida_transparente, "PNG")

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
    parser = argparse.ArgumentParser(description="Procesamiento completo de ropa sin errores de estampado.")
    parser.add_argument("-i", "--entrada", required=True)
    parser.add_argument("-o", "--salida", required=True)
    args = parser.parse_args()

    imagenes = buscar_imagenes(args.entrada)
    if not imagenes:
        print(f"No hay imágenes en: {args.entrada}")
        sys.exit(1)

    os.makedirs(args.salida, exist_ok=True)

    print("Cargando modelo de alta precisión...")
    session = new_session("isnet-general-use")

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