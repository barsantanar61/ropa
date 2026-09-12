#!/usr/bin/env python3
"""
armario_virtual.py

Crea un armario interactivo para combinar prendas procesadas
y aplicar un acabado estético de catálogo sin errores de rango.
"""

import os
import glob
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance

st.set_page_config(page_title="Mi Armario Virtual", layout="wide")

# --- FUNCIONES DE MEJORA ESTÉTICA ---

def aplicar_sombra_y_estetica(img_rgba, color_fondo=(245, 244, 240), offset=(0, 15), blur=25):
    alpha = img_rgba.split()[3]
    sombra_mask = alpha.filter(ImageFilter.GaussianBlur(blur))
    sombra = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    sombra_negra = Image.new("RGBA", img_rgba.size, (40, 40, 40, 90))
    sombra.paste(sombra_negra, offset, sombra_mask)
    
    fondo = Image.new("RGBA", img_rgba.size, color_fondo + (255,))
    fondo.paste(sombra, (0, 0), sombra)
    fondo.paste(img_rgba, (0, 0), img_rgba)
    
    enhancer_color = ImageEnhance.Color(fondo)
    return enhancer_color.enhance(1.05).convert("RGB")

# --- CARGA Y CATEGORIZACIÓN ---

CARPETA_RESULTADOS = "./resultados"

def cargar_prendas():
    if not os.path.exists(CARPETA_RESULTADOS):
        return [], []
    
    archivos = glob.glob(os.path.join(CARPETA_RESULTADOS, "*_transparente.png"))
    
    arriba = []
    abajo = []
    
    for ruta in archivos:
        nombre = os.path.basename(ruta).lower()
        # Identificar prendas inferiores por palabras clave
        if any(kw in nombre for kw in ["pant", "pantalon", "falda", "short", "jean", "22.29.19"]):
            abajo.append(ruta)
        else:
            arriba.append(ruta)
            
    return sorted(arriba), sorted(abajo)

# --- INTERFAZ DE STREAMLIT ---

st.title("✨ Armario Virtual & Lookbook")

partes_arriba, partes_abajo = cargar_prendas()

# Configuración de Sidebar
with st.sidebar:
    st.header("⚙️ Estilo Estético")
    estilo_fondo = st.selectbox(
        "Fondo de catálogo",
        ["Crema / Lino (#F5F4F0)", "Blanco Estudio (#FFFFFF)", "Gris Cemento (#E5E5E5)", "Rosa Pastel (#FCE4EC)"]
    )
    colores = {
        "Crema / Lino (#F5F4F0)": (245, 244, 240),
        "Blanco Estudio (#FFFFFF)": (255, 255, 255),
        "Gris Cemento (#E5E5E5)": (229, 229, 229),
        "Rosa Pastel (#FCE4EC)": (252, 228, 236)
    }
    color_elegido = colores[estilo_fondo]

if not partes_arriba and not partes_abajo:
    st.warning("No hay imágenes procesadas en la carpeta './resultados'.")
else:
    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        st.subheader("👕 Parte de Arriba")
        if partes_arriba:
            nombres_arriba = [os.path.basename(p).replace("_transparente.png", "") for p in partes_arriba]
            sel_arriba = st.selectbox("Seleccionar prenda superior:", nombres_arriba, key="sb_up")
            idx_arriba = nombres_arriba.index(sel_arriba)
        else:
            st.info("No hay partes de arriba disponibles.")
            idx_arriba = None

        st.subheader("👖 Parte de Abajo")
        if partes_abajo:
            nombres_abajo = [os.path.basename(p).replace("_transparente.png", "") for p in partes_abajo]
            sel_abajo = st.selectbox("Seleccionar prenda inferior:", nombres_abajo, key="sb_down")
            idx_abajo = nombres_abajo.index(sel_abajo)
        else:
            st.info("No hay partes de abajo disponibles.")
            idx_abajo = None

    with col_der:
        st.subheader("🖼️ Vista Previa del Outfit")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            if idx_arriba is not None:
                img_up = Image.open(partes_arriba[idx_arriba]).convert("RGBA")
                st.image(aplicar_sombra_y_estetica(img_up, color_fondo=color_elegido), caption="Superior", use_container_width=True)

        with v_col2:
            if idx_abajo is not None:
                img_down = Image.open(partes_abajo[idx_abajo]).convert("RGBA")
                st.image(aplicar_sombra_y_estetica(img_down, color_fondo=color_elegido), caption="Inferior", use_container_width=True)