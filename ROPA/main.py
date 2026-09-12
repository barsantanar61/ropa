import os
import time
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = './uploads'
RESULTADOS_FOLDER = './resultados'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

# Sirve la página principal
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Sirve archivos estáticos (imágenes de resultados, CSS, JS)
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# Endpoint para subir y procesar fotos
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo'}), 400
    
    file = request.files['file']
    prenda_tipo = request.form.get('type', 'prenda')
    filename = f"{prenda_tipo}_{int(time.time())}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Procesar con tu script de quitar fondo
    try:
        subprocess.run(["python", "quitar_fondo_ropa_v6_pro.py", "-i", filepath, "-o", RESULTADOS_FOLDER], check=True)
        return jsonify({'message': 'Foto procesada con éxito', 'filename': filename}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)