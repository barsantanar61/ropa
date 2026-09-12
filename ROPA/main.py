import os
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

# Carga la sesión ligera de IA (u2netp) para procesamiento ultra-rápido y bajo consumo de memoria
session_rembg = new_session("u2netp")

def guardar_en_github_permanente(filepath_local, filename):
    """Sube la imagen procesada a la carpeta resultados/ en GitHub."""
    if not GITHUB_TOKEN:
        print("Aviso: GITHUB_TOKEN no encontrado. Guardado permanente omitido.")
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
    except Exception as e:
        print(f"❌ Excepción al subir a GitHub: {str(e)}")


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
    
    try:
        # Recorte directo en memoria con el modelo liviano
        input_image = Image.open(filepath_upload)
        output_image = remove(input_image, session=session_rembg)
        output_image.save(filepath_resultado)
        
        # Persistencia en GitHub
        guardar_en_github_permanente(filepath_resultado, filename)

        return jsonify({'message': 'Foto procesada correctamente', 'filename': filename}), 200
    except Exception as e:
        print(f"Error procesando imagen: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
