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

# obtener token SAS de las variables de entorno
tokenSas = os.getenv("TOKENSAS")
if not tokenSas:
    raise ValueError("Error: La variable de entorno 'TOKENSAS' no esta definida.")

# configurar URL base del contenedor de Azure
azureContainerUrl = "https://thermalalmacen.blob.core.windows.net/fotos"
apiUrl = "http://4.155.33.198:8000/images/with-detections"

# configurar modelo y captura
rtspUrl = "rtsp://lizardi:zenobia16@10.3.56.116/cam/realmonitor?channel=2&subtype=0"
modelPath = "./modelos/beta01.pt"
confThreshold = 0.5
captureInterval = 10
temporalDir = "./frames_temp"


# asegurar que la carpeta temporal exista
os.makedirs(temporalDir, exist_ok=True)

# cargar el modelo YOLO
try:
    model = YOLO(modelPath)
except Exception as error:
    print(f"Error al cargar el modelo YOLO: {error}")
    exit()


def subirAzure(localPath, fileName):
    """Subir la imagen a Azure Blob Storage usando el Token SAS."""
    blobUrl = f"{azureContainerUrl}/{fileName}?{tokenSas}"

    try:
        with open(localPath, "rb") as file:
            # usar requests.put para subir el archivo
            response = requests.put(
                blobUrl,
                data=file,
                headers={
                    'x-ms-blob-type': 'BlockBlob',
                    'Content-Type': 'image/jpeg'
                }
            )

        if response.status_code == 201:
            print(f"[Azure] Subida exitosa: {fileName}")
            # devolver la URL publica sin el token
            return f"{azureContainerUrl}/{fileName}"
        else:
            print(f"[Azure] Error {response.status_code} al subir {fileName}: {response.text}")
            return None
    except Exception as error:
        print(f"[Azure] Excepcion al subir a Azure: {error}")
        return None


def enviarApi(imageUrl, detectionList):
    """Enviar datos de la deteccion (URL y Bounding Boxes) a la API."""
    try:
        response = requests.post(f"{apiUrl}?image_path={imageUrl}", json=detectionList)

        if response.status_code in [200, 201]:
            print(f"[API] Registro correcto para {os.path.basename(imageUrl)}")
        else:
            print(f"[API] Error {response.status_code} al registrar: {response.text}")
    except Exception as error:
        print(f"[API] Error al comunicar con el servidor: {error}")


def ejecutarCaptura():
    """Bucle principal de captura, deteccion y subida."""
    # cap = cv2.VideoCapture(RTSP_URL) # Usar stream real
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo conectar al stream de video.")
        return

    print("Capturando frames, detectando y subiendo a Azure...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame. Reintentando...")
                time.sleep(1)
                continue

            # 1. Preparar nombres y rutas
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fileName = f"frame_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
            localPath = os.path.join(temporalDir, fileName)

            # 2. Guardar imagen temporal
            cv2.imwrite(localPath, frame)

            # 3. Ejecutar modelo YOLO
            results = model(localPath, conf=confThreshold, iou=0.45, verbose=False)
            yoloDetections = results[0].boxes

            detectionList = []

            if len(yoloDetections) > 0:
                print(f"Cigarros Detectados {len(yoloDetections)} objeto(s). Procesando...")

                # anotar el frame y guardar la imagen anotada
                annotatedFrame = results[0].plot()
                cv2.imwrite(localPath, annotatedFrame)

                for det in yoloDetections:
                    confidence = float(det.conf[0])
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    detectionList.append({
                        "confianza": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                # 4. Subir imagen a Azure
                azureUrl = subirAzure(localPath, fileName)

                # 5. Enviar a la API
                if azureUrl:
                    enviarApi(azureUrl, detectionList)

            # 6. Limpiar archivo temporal
            os.remove(localPath)

            time.sleep(captureInterval)

    except KeyboardInterrupt:
        print("\nCaptura detenida.")
    finally:
        cap.release()
        # limpiar cualquier archivo temporal restante al salir
        for file in os.listdir(temporalDir):
            os.remove(os.path.join(temporalDir, file))


if __name__ == "__main__":
    ejecutarCaptura()