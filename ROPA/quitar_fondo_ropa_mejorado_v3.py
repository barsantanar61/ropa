#!/usr/bin/env python3
"""
quitar_fondo_definitivo.py

Elimina el fondo de prendas de vestir utilizando Inteligencia Artificial (rembg / u2net).
Funciona perfecto con fondos estampados, sábanas arrugadas, ropa clara/oscura y objetos alrededor.

Requisitos:
    pip install rembg pillow

Uso:
    python quitar_fondo_definitivo.py -i chaqueta.jpg -o chaqueta_sin_fondo.png
    python quitar_fondo_definitivo.py -i ./fotos -o ./salida
"""

import argparse
import glob
import os
import sys
from PIL import Image

try:
    from rembg import remove, new_session
except ImportError:
    print("\n[ERROR] Falta la librería 'rembg'.")
    print("Por favor, instala la librería ejecutando en tu consola:")
    print("    pip install rembg pillow\n")
    sys.exit(1)

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def estabilizar_lienzo(imagen_rgba, tamano_lienzo=(1000, 1300), margen_porcentaje=0.90):
    """
    Centra la prenda recortada en un lienzo transparente de tamaño fijo.
    """
    canvas = Image.new("RGBA", tamano_lienzo, (0, 0, 0, 0))
    cw, ch = tamano_lienzo
    iw, ih = imagen_rgba.size

    escala = min((cw * margen_porcentaje) / iw, (ch * margen_porcentaje) / ih)
    
    nuevo_ancho = max(1, int(iw * escala))
    nuevo_alto = max(1, int(ih * escala))
    
    imagen_resaltada = imagen_rgba.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    pos_x = (cw - nuevo_ancho) // 2
    pos_y = (ch - nuevo_alto) // 2

    canvas.alpha_composite(imagen_resaltada, (pos_x, pos_y))
    return canvas

def procesar_imagen(ruta_entrada, ruta_salida, session, estabilizar=True, tamano_lienzo=(1000, 1300)):
    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"No existe el archivo: {ruta_entrada}")

    # 1. Cargar imagen
    img_original = Image.open(ruta_entrada).convert("RGB")

    # 2. Remover fondo con Red Neuronal
    resultado_rgba = remove(img_original, session=session)

    # 3. Recortar bordes transparentes sobrantes (Bounding Box)
    bbox = resultado_rgba.getbbox()
    if bbox:
        resultado_rgba = resultado_rgba.crop(bbox)
    else:
        raise ValueError("No se pudo detectar ningún objeto principal en la imagen.")

    # 4. Estabilizar / Centrar en lienzo estándar
    if estabilizar:
        resultado_final = estabilizar_lienzo(resultado_rgba, tamano_lienzo=tamano_lienzo)
    else:
        resultado_final = resultado_rgba

    # 5. Guardar como PNG transparente
    resultado_final.save(ruta_salida, "PNG")

def buscar_imagenes(ruta_entrada):
    if not os.path.isdir(ruta_entrada):
        return [ruta_entrada]
    
    archivos = []
    for ext in EXTENSIONES:
        archivos.extend(glob.glob(os.path.join(ruta_entrada, "*" + ext)))
        archivos.extend(glob.glob(os.path.join(ruta_entrada, "*" + ext.upper())))
    return sorted(set(archivos))

def main():
    parser = argparse.ArgumentParser(description="Extracción perfecta de ropa mediante IA.")
    parser.add_argument("-i", "--entrada", required=True, help="Ruta de la imagen o carpeta de entrada")
    parser.add_argument("-o", "--salida", required=True, help="Ruta del PNG de salida o carpeta de destino")
    parser.add_argument("--sin-estabilizar", action="store_true", help="Desactiva el centrado en lienzo de 1000x1300")
    args = parser.parse_args()

    imagenes = buscar_imagenes(args.entrada)
    if not imagenes:
        print(f"No se encontraron imágenes válidas en: {args.entrada}")
        sys.exit(1)

    es_carpeta_salida = os.path.isdir(args.salida) or len(imagenes) > 1
    if es_carpeta_salida:
        os.makedirs(args.salida, exist_ok=True)

    # Crear sesión de IA para procesar más rápido
    print("Cargando modelo de Inteligencia Artificial (u2net)...")
    session = new_session("u2net")

    for entrada in imagenes:
        nombre = os.path.splitext(os.path.basename(entrada))[0]
        if es_carpeta_salida:
            salida = os.path.join(args.salida, f"{nombre}_sin_fondo.png")
        else:
            salida = args.salida

        try:
            print(f"Procesando: {os.path.basename(entrada)} ...", end=" ", flush=True)
            procesar_imagen(
                ruta_entrada=entrada,
                ruta_salida=salida,
                session=session,
                estabilizar=not args.sin_estabilizar
            )
            print("[OK]")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()