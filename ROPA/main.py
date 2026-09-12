import os
import time
import subprocess
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = './uploads'
RESULTADOS_FOLDER = './resultados'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

# Variables para autenticación y guardado en GitHub
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPO = "barsantanar61/ropa"

def guardar_en_github_permanente(filepath_local, filename):
    """Sube la imagen procesada a la carpeta resultados/ en GitHub para que no se pierda al reiniciar Render."""
    if not GITHUB_TOKEN:
        print("Aviso: GITHUB_TOKEN no encontrado. La imagen solo persistirá durante la sesión actual de Render.")
        return

    # Ruta en GitHub donde se creará/guardará la imagen
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
            print(f"✅ Imagen {filename} guardada permanentemente en GitHub")
        else:
            print(f"❌ Error al guardar en GitHub ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Excepción al subir a GitHub: {str(e)}")


@app.route('/')
def index():
    """Sirve la página principal de la aplicación."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Sirve archivos estáticos como imágenes de resultados, CSS o JS."""
    return send_from_directory('.', path)


@app.route('/api/garments', methods=['GET'])
def get_garments():
    """Devuelve dinámicamente la lista de prendas procesadas que hay en la carpeta /resultados."""
    files = os.listdir(RESULTADOS_FOLDER) if os.path.exists(RESULTADOS_FOLDER) else []
    
    # Filtrar solo archivos con extensiones de imagen válidas
    valid_files = [f"/resultados/{f}" for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # Clasificar las prendas entre partes superiores (tops) e inferiores (bottoms)
    tops = [f for f in valid_files if 'pantalon' not in f.lower()]
    bottoms = [f for f in valid_files if 'pantalon' in f.lower()]
    
    return jsonify({'tops': tops, 'bottoms': bottoms})


@app.route('/upload', methods=['POST'])
def upload_file():
    """Recibe la imagen capturada por la cámara, la procesa con rembg y la guarda."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha proporcionado ningún archivo'}), 400
    
    file = request.files['file']
    prenda_tipo = request.form.get('type', 'prenda')
    
    # Nombrar el archivo con marca de tiempo para evitar duplicados
    filename = f"{prenda_tipo}_{int(time.time())}.png"
    filepath_upload = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath_upload)
    
    try:
        # 1. Recortar el fondo con el script de Python local
        subprocess.run(["python", "quitar_fondo_ropa_v6_pro.py", "-i", filepath_upload, "-o", RESULTADOS_FOLDER], check=True)
        
        filepath_resultado = os.path.join(RESULTADOS_FOLDER, filename)
        
        # 2. Guardar automáticamente en el repositorio de GitHub de forma permanente
        guardar_en_github_permanente(filepath_resultado, filename)

        return jsonify({'message': 'Foto procesada y guardada correctamente', 'filename': filename}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
