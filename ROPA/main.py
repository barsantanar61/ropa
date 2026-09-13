import os
import gc
import time
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = './uploads'
RESULTADOS_FOLDER = './resultados'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO = "barsantanar61/ropa"

# Sesión ligera de rembg para ejecución rápida en entornos limitados
session_rembg = new_session("u2netp")

# Ancho máximo con el que se procesa la imagen en el servidor.
# El frontend ya reduce a 1000px antes de enviar, pero en el servidor
# volvemos a bajarlo más (a 700px) porque aquí es donde de verdad
# se dispara el consumo de RAM: PIL + numpy + el modelo de rembg
# trabajan sobre esta copia, no sobre el archivo original.
MAX_ANCHO_SERVIDOR = 700


def redimensionar_si_hace_falta(img, max_ancho=MAX_ANCHO_SERVIDOR):
    """Reduce la imagen si supera max_ancho, manteniendo proporción."""
    if img.width <= max_ancho:
        return img
    ratio = max_ancho / float(img.width)
    nuevo_alto = int(img.height * ratio)
    return img.resize((max_ancho, nuevo_alto), Image.Resampling.LANCZOS)


def guardar_en_github_permanente(filepath_local, filename):
    """Sube la imagen procesada a la carpeta ROPA/resultados/ en GitHub."""
    if not GITHUB_TOKEN:
        print("Aviso: GITHUB_TOKEN no configurado. Guardado permanente omitido.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/ROPA/resultados/{filename}"

    try:
        with open(filepath_local, "rb") as file:
            content = base64.b64encode(file.read()).decode('utf-8')

        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        data = {
            "message": f"Añadir nueva prenda {filename} desde la app web",
            "content": content,
            "branch": "main"
        }

        response = requests.put(url, json=data, headers=headers)
        if response.status_code in [200, 201]:
            print(f"✅ Imagen {filename} guardada en GitHub")
        else:
            print(f"❌ Error GitHub ({response.status_code}): {response.text}")

        # 'content' puede pesar bastante (base64 del PNG entero).
        # La liberamos explícitamente en cuanto termina la petición.
        del content
    except Exception as e:
        print(f"❌ Excepción al subir a GitHub: {str(e)}")
    finally:
        gc.collect()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/api/garments', methods=['GET'])
def get_garments():
    files = os.listdir(RESULTADOS_FOLDER) if os.path.exists(RESULTADOS_FOLDER) else []
    valid_files = [f"/resultados/{f}" for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    tops = [f for f in valid_files if 'pantalon' not in f.lower()]
    bottoms = [f for f in valid_files if 'pantalon' in f.lower()]

    return jsonify({'tops': tops, 'bottoms': bottoms})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo'}), 400

    file = request.files['file']
    prenda_tipo = request.form.get('type', 'prenda')

    filename = f"{prenda_tipo}_{int(time.time())}.png"
    filepath_upload = os.path.join(UPLOAD_FOLDER, filename)
    filepath_resultado = os.path.join(RESULTADOS_FOLDER, filename)

    file.save(filepath_upload)

    input_image = None
    output_image = None

    try:
        print(f"[upload] Abriendo imagen {filename}...")
        # Abrir y convertir a RGB para asegurar compatibilidad con capturas de móvil (JPEG, HEIC, PNG)
        input_image = Image.open(filepath_upload).convert('RGB')

        print(f"[upload] Tamaño original: {input_image.size}")
        input_image = redimensionar_si_hace_falta(input_image)
        print(f"[upload] Tamaño tras redimensionar: {input_image.size}")

        print("[upload] Ejecutando rembg...")
        output_image = remove(input_image, session=session_rembg)

        print("[upload] Guardando resultado en disco...")
        output_image.save(filepath_resultado)

        # Liberamos las imágenes en memoria ANTES de subir a GitHub,
        # que ya de por sí vuelve a leer el archivo de disco.
        del input_image
        del output_image
        input_image = None
        output_image = None
        gc.collect()

        print("[upload] Subiendo a GitHub...")
        # Persistencia en GitHub
        guardar_en_github_permanente(filepath_resultado, filename)

        print("[upload] Completado OK")
        return jsonify({'message': 'Foto procesada correctamente', 'filename': filename}), 200
    except Exception as e:
        print(f"[upload] ERROR procesando imagen: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Por si el except saltó antes de llegar a la limpieza de arriba
        if input_image is not None:
            del input_image
        if output_image is not None:
            del output_image
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
