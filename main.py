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
temporalDir = "./frames_temp"

# intervalo entre frames para analizar
frameInterval = 5

# asegurar que la carpeta temporal exista
os.makedirs(temporalDir, exist_ok=True)

# cargar el modelo YOLO
try:
    model = YOLO(modelPath)
except Exception as error:
    print(f"Error al cargar el modelo YOLO: {error}")
    exit()


def subirAzure(localPath, fileName):
    """subir la imagen a Azure Blob Storage usando el Token SAS"""
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
    """enviar datos de la deteccion a la API"""
    try:
        response = requests.post(f"{apiUrl}?image_path={imageUrl}", json=detectionList)

        if response.status_code in [200, 201]:
            print(f"[API] Registro correcto para {os.path.basename(imageUrl)}")
        else:
            print(f"[API] Error {response.status_code} al registrar: {response.text}")
    except Exception as error:
        print(f"[API] Error al comunicar con el servidor: {error}")


def ejecutarCaptura():
    """bucle principal de captura con deteccion por cambio de cantidad"""
    # cap = cv2.VideoCapture(rtspUrl) # usar stream real
    cap = cv2.VideoCapture(rtspUrl)

    if not cap.isOpened():
        print("Error: No se pudo conectar al stream de video.")
        return

    print("Capturando frames y detectando cambios en cantidad de cigarros...")

    # variable para rastrear la cantidad anterior de detecciones
    cantidadAnterior = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame. Reintentando...")
                time.sleep(1)
                continue

            # preparar ruta temporal para analisis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fileName = f"frame_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
            localPath = os.path.join(temporalDir, fileName)

            # guardar imagen temporal para analisis
            cv2.imwrite(localPath, frame)

            # ejecutar modelo YOLO
            results = model(localPath, conf=confThreshold, iou=0.45, verbose=False)
            yoloDetections = results[0].boxes
            cantidadActual = len(yoloDetections)

            # verificar si hubo cambio en la cantidad
            if cantidadActual != cantidadAnterior:
                print(f"Cambio detectado: {cantidadAnterior} -> {cantidadActual} cigarros")

                # solo procesar si hay al menos una deteccion
                if cantidadActual > 0:
                    detectionList = []

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

                    # subir imagen a Azure
                    azureUrl = subirAzure(localPath, fileName)

                    # enviar a la API
                    if azureUrl:
                        enviarApi(azureUrl, detectionList)
                else:
                    print("Cambio a 0 cigarros detectado (no se guarda imagen)")

                # actualizar cantidad anterior
                cantidadAnterior = cantidadActual

            # limpiar archivo temporal
            if os.path.exists(localPath):
                os.remove(localPath)

            time.sleep(frameInterval)

    except KeyboardInterrupt:
        print("\nCaptura detenida.")
    finally:
        cap.release()
        # limpiar cualquier archivo temporal restante al salir
        for file in os.listdir(temporalDir):
            filePath = os.path.join(temporalDir, file)
            if os.path.exists(filePath):
                os.remove(filePath)


if __name__ == "__main__":
    ejecutarCaptura()