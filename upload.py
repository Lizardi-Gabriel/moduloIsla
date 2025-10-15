import cv2
import time
import requests
import os
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

TOKENSAS = os.getenv("TOKENSAS")

# --- CONFIGURACIÓN ---
RTSP_URL = "rtsp://lizardi:zenobia16@10.3.56.116/cam/realmonitor?channel=2&subtype=0"

CONF_THRESHOLD = 0.6
CAPTURE_INTERVAL = 5  # segundos entre detecciones
TEMP_DIR = "./frames"

# --- ASEGURAR CARPETA TEMPORAL ---
os.makedirs(TEMP_DIR, exist_ok=True)

# --- INICIAR CAPTURA ---
#cap = cv2.VideoCapture(RTSP_URL)
cap = cv2.VideoCapture(0)  # usar cámara local para pruebas
if not cap.isOpened():
    print("Error: No se pudo conectar al stream RTSP.")
    exit()

print("Capturando frames y subiendo a Azure Blob...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame. Reintentando...")
            time.sleep(1)
            continue

        # Nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(TEMP_DIR, filename)

        # Guardar imagen temporal
        cv2.imwrite(filepath, frame)

        # URL de subida (añadir nombre del blob antes del token SAS)
        blob_url = f"https://thermalalmacen.blob.core.windows.net/fotos/{filename}?{TOKENSAS}"

        # Subir a Azure Blob
        with open(filepath, "rb") as f:
            response = requests.put(blob_url, data=f, headers={'x-ms-blob-type': 'BlockBlob'})

        if response.status_code == 201:
            print(f"Subido: {filename}")
            os.remove(filepath)  # eliminar frame local
        else:
            print(f"Error al subir {filename}: {response.status_code} - {response.text}")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\nCaptura detenida por el usuario.")
finally:
    cap.release()
