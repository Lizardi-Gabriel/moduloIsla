import cv2
import time
import requests
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import calendar

load_dotenv()

# ===================== CONFIGURACIÓN Y CREDENCIALES =====================

tokenSas = os.getenv("TOKENSAS")
if not tokenSas:
    raise ValueError("Error: La variable de entorno 'TOKENSAS' no está definida.")

azureContainerUrl = "https://thermalalmacen.blob.core.windows.net/fotos"
apiUrl = "http://4.155.33.198:8000/images/with-detections"

rtspUrl = "rtsp://lizardi:zenobia16@10.3.56.116/cam/realmonitor?channel=2&subtype=0"
modelPath = "./modelos/beta01.pt"
confThreshold = 0.5
temporalDir = "./frames_temp"

# Intervalos base
intervaloNormal = 10  # segundos cuando no hay fumador
intervaloFumador = 5  # segundos cuando hay fumador
sinDeteccionesLimite = 3  # fotos seguidas sin fumador antes de volver a 10s

os.makedirs(temporalDir, exist_ok=True)

# Cargar modelo YOLO
try:
    model = YOLO(modelPath)
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    exit()


def dentroHorario():
    """Verifica si es lunes-viernes y entre 7am y 9pm."""
    ahora = datetime.now()
    if calendar.weekday(ahora.year, ahora.month, ahora.day) >= 5:  # 5=sábado, 6=domingo
        return False
    return 7 <= ahora.hour < 21


def conectar_rtsp():
    """Intenta conectar con el RTSP, reintentando si falla."""
    while True:
        cap = cv2.VideoCapture(rtspUrl)
        if cap.isOpened():
            print("[INFO] Conectado al stream RTSP.")
            return cap
        print("[WARN] No se pudo conectar al RTSP. Reintentando en 5s...")
        time.sleep(5)


def subirAzure(localPath, fileName):
    blobUrl = f"{azureContainerUrl}/{fileName}?{tokenSas}"
    try:
        with open(localPath, "rb") as file:
            response = requests.put(
                blobUrl,
                data=file,
                headers={'x-ms-blob-type': 'BlockBlob', 'Content-Type': 'image/jpeg'}
            )
        if response.status_code == 201:
            print(f"[Azure] Subida exitosa: {fileName}")
            return f"{azureContainerUrl}/{fileName}"
        else:
            print(f"[Azure] Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[Azure] Excepción al subir: {e}")
        return None


def enviarApi(imageUrl, detectionList):
    try:
        response = requests.post(f"{apiUrl}?image_path={imageUrl}", json=detectionList)
        if response.status_code in [200, 201]:
            print(f"[API] Registro correcto para {os.path.basename(imageUrl)}")
        else:
            print(f"[API] Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[API] Error al comunicar con el servidor: {e}")


def ejecutarCaptura():
    cap = conectar_rtsp()
    sinFumadoresContador = 0
    intervaloActual = intervaloNormal

    print("[INFO] Iniciando monitoreo...")

    while True:
        try:
            # Verificar horario
            if not dentroHorario():
                print("[INFO] Fuera de horario laboral (7am-9pm). Pausando...")
                time.sleep(60)
                continue

            ret, frame = cap.read()
            if not ret:
                print("[WARN] No se pudo leer frame. Reintentando conexión...")
                cap.release()
                time.sleep(3)
                cap = conectar_rtsp()
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fileName = f"frame_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
            localPath = os.path.join(temporalDir, fileName)
            cv2.imwrite(localPath, frame)

            results = model(localPath, conf=confThreshold, iou=0.45, verbose=False)
            yoloDetections = results[0].boxes
            cantidadActual = len(yoloDetections)

            if cantidadActual > 0:
                print(f"[DETECCIÓN] {cantidadActual} fumador(es) detectado(s)")
                sinFumadoresContador = 0
                intervaloActual = intervaloFumador

                annotatedFrame = results[0].plot()
                cv2.imwrite(localPath, annotatedFrame)

                detectionList = []
                for det in yoloDetections:
                    confidence = float(det.conf[0])
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    detectionList.append({
                        "confianza": confidence,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    })

                azureUrl = subirAzure(localPath, fileName)
                if azureUrl:
                    enviarApi(azureUrl, detectionList)

            else:
                sinFumadoresContador += 1
                print(f"[INFO] No se detectaron fumadores ({sinFumadoresContador}/{sinDeteccionesLimite})")

                if sinFumadoresContador >= sinDeteccionesLimite:
                    intervaloActual = intervaloNormal

            if os.path.exists(localPath):
                os.remove(localPath)

            time.sleep(intervaloActual)

        except KeyboardInterrupt:
            print("\n[INFO] Captura detenida manualmente.")
            break
        except Exception as e:
            print(f"[ERROR] Error inesperado: {e}")
            cap.release()
            time.sleep(5)
            cap = conectar_rtsp()

    cap.release()
    for f in os.listdir(temporalDir):
        os.remove(os.path.join(temporalDir, f))


if __name__ == "__main__":
    ejecutarCaptura()
