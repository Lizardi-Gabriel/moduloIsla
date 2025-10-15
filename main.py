import cv2
import time
import requests
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIGURACIÓN Y CREDENCIALES =====================

# Obtener Token SAS de las variables de entorno
tokenSas = os.getenv("TOKENSAS")
if not tokenSas:
    raise ValueError("Error: La variable de entorno 'TOKENSAS' no estar definida.")

# Configuración de Azure (URL base del contenedor)
# Asegurar que la URL base incluya el nombre de la cuenta y el contenedor
AZURE_CONTAINER_URL = "https://thermalalmacen.blob.core.windows.net/fotos"
API_URL = "http://4.155.33.198:8000/images/with-detections" # Ajustar la URL de tu API

# Configuración del modelo y captura
RTSP_URL = "rtsp://lizardi:zenobia16@10.3.56.116/cam/realmonitor?channel=2&subtype=0"
RUTA_MODELO = "./modelos/beta01.pt" # Asegurar que esta ruta sea correcta
CONF_THRESHOLD = 0.6
INTERVALO_CAPTURAR = 5  # Segundos entre detecciones
TEMPORAL_DIR = "./frames_temp"


# Asegurar carpeta temporal
os.makedirs(TEMPORAL_DIR, exist_ok=True)

# Cargar el modelo YOLO
try:
    modelo = YOLO(RUTA_MODELO)
except Exception as error:
    print(f"Error al cargar el modelo YOLO: {error}")
    exit()


def subirAzure(rutaLocal, nombreArchivo):
    """Subir la imagen anotada a Azure Blob Storage usando el Token SAS."""
    # Construir la URL completa para la subida PUT (blob_url)
    # Se añade el nombre del archivo y el token SAS
    blobUrl = f"{AZURE_CONTAINER_URL}/{nombreArchivo}?{tokenSas}"

    try:
        with open(rutaLocal, "rb") as archivo:
            # Usar requests.put para subir el archivo (necesario con SAS)
            response = requests.put(
                blobUrl,
                data=archivo,
                headers={
                    'x-ms-blob-type': 'BlockBlob', # Tipo de blob necesario para la subida
                    'Content-Type': 'image/jpeg' # Tipo de contenido
                }
            )

        if response.status_code == 201:
            print(f"[Azure] Subir con éxito: {nombreArchivo}")
            return f"{AZURE_CONTAINER_URL}/{nombreArchivo}" # Devolver la URL pública sin el token SAS
        else:
            print(f"[Azure] Error {response.status_code} al subir {nombreArchivo}: {response.text}")
            return None
    except Exception as error:
        print(f"[Azure] Excepción al subir a Azure: {error}")
        return None


def enviarApi(urlImagen, listaDetecciones):
    """Enviar datos de la detección (URL y Bounding Boxes) a la API del servidor."""
    payload = {
        "urlImagen": urlImagen,
        "detecciones": listaDetecciones
    }
    try:
        respuesta = requests.post(API_URL, json=payload)
        if respuesta.status_code in [200, 201]:
            print(f"[API] Registro correcto para {os.path.basename(urlImagen)}")
        else:
            print(f"[API] Error {respuesta.status_code} al registrar: {respuesta.text}")
    except Exception as error:
        print(f"[API] Error al comunicar con el servidor: {error}")


def ejecutarCaptura():
    """Bucle principal de captura, detección y subida."""
    # cap = cv2.VideoCapture(RTSP_URL) # Usar stream real
    cap = cv2.VideoCapture(0) # Usar cámara local para pruebas

    if not cap.isOpened():
        print("Error: No poder conectar al stream de video.")
        return

    print("Capturando frames, detectando y subiendo a Azure...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No poder leer frame. Reintentar...")
                time.sleep(1)
                continue

            # 1. Preparar nombres y rutas
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombreArchivo = f"frame_{timestamp}_{uuid.uuid4().hex[:6]}.jpg" # ID único para evitar colisiones
            rutaLocal = os.path.join(TEMPORAL_DIR, nombreArchivo)

            # 2. Guardar imagen temporal
            cv2.imwrite(rutaLocal, frame)

            # 3. Ejecutar modelo YOLO
            resultados = modelo(rutaLocal, conf=CONF_THRESHOLD, iou=0.45, verbose=False)
            deteccionesYolo = resultados[0].boxes

            listaDetecciones = []

            if len(deteccionesYolo) > 0:
                print(f"🚬 Detectar {len(deteccionesYolo)} objeto(s). Procesar...")

                # Anotar el frame y guardar la imagen anotada
                frameAnotado = resultados[0].plot()
                cv2.imwrite(rutaLocal, frameAnotado)

                # Construir la lista de detecciones para la API
                for det in deteccionesYolo:
                    confianza = float(det.conf[0])
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    clase = int(det.cls[0])
                    listaDetecciones.append({
                        "clase": clase,
                        "confianza": confianza,
                        "bbox": [x1, y1, x2, y2]
                    })

                # 4. Subir imagen a Azure
                urlAzure = subirAzure(rutaLocal, nombreArchivo)

                # 5. Enviar a la API
                if urlAzure:
                    enviarApi(urlAzure, listaDetecciones)

            # 6. Limpiar archivo temporal (siempre)
            #os.remove(rutaLocal)

            time.sleep(INTERVALO_CAPTURAR)

    except KeyboardInterrupt:
        print("\nCaptura detener por el usuario.")
    finally:
        cap.release()
        # Limpiar cualquier archivo temporal restante
        for archivo in os.listdir(TEMPORAL_DIR):
            os.remove(os.path.join(TEMPORAL_DIR, archivo))

if __name__ == "__main__":
    ejecutarCaptura()