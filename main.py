import cv2
import time
import requests
import threading
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import YOLO
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thermal_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThermalMonitorClient:
    """Cliente para monitoreo térmico con detección automática de eventos"""

    def __init__(
            self,
            camera_source: str,
            api_base_url: str,
            username: str,
            password: str,
            model_path: str,
            confidence_threshold: float = 0.5,
            azure_container_url: str = None,
            azure_token_sas: str = None
    ):
        # Puede ser int (0, 1) o string (RTSP URL, archivo de video)
        self.camera_source = camera_source
        self.api_base_url = api_base_url.rstrip('/')
        self.username = username
        self.password = password
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # Configurar Azure desde variables de entorno o parámetros
        self.azure_container_url = azure_container_url or os.getenv("AZURE_CONTAINER_URL")
        self.azure_token_sas = azure_token_sas or os.getenv("TOKENSAS")

        if not self.azure_container_url or not self.azure_token_sas:
            logger.warning("Configuración de Azure no encontrada. Las imágenes no se subirán a Azure.")
        else:
            logger.info("Configuración de Azure cargada correctamente")

        # Estado del cliente
        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[YOLO] = None
        self.token: Optional[str] = None
        self.running = False

        # NUEVO: Thread de lectura continua
        self.frame_actual = None
        self.frame_lock = threading.Lock()
        self.thread_lectura = None
        self.thread_lectura_running = False

        # Estado de detección
        self.estado_actual = "sin_deteccion"
        self.id_evento_activo: Optional[int] = None
        self.contador_sin_deteccion = 0
        self.contador_con_deteccion = 0

        # Configuración de tiempos
        self.tiempo_foto_sin_deteccion = 5
        self.tiempo_foto_con_deteccion = 2
        self.umbral_crear_evento = 3
        self.umbral_cerrar_evento = 5

        # Directorio para guardar imágenes temporales
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)

    def inicializar_camara(self) -> bool:
        """Inicializar conexión con la cámara"""
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()

            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
            logger.info(f"Conectando a cámara RTSP/URL: {self.camera_source}")

            if not self.cap.isOpened():
                logger.error("No se pudo abrir la cámara")
                return False

            # Verificar que se pueda leer un frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("No se pudo leer frame de prueba")
                self.cap.release()
                return False

            logger.info(f"Cámara inicializada correctamente - Resolución: {frame.shape[1]}x{frame.shape[0]}")
            return True

        except Exception as e:
            logger.error(f"Error al inicializar cámara: {e}")
            return False

    def _leer_frames_continuamente(self):
        """
        Thread que lee frames continuamente del stream RTSP.
        Esto evita que el buffer se llene con frames viejos.
        """
        logger.info("Thread de lectura continua iniciado")

        while self.thread_lectura_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()

                    if ret and frame is not None:
                        # Actualizar el frame actual de forma thread-safe
                        with self.frame_lock:
                            self.frame_actual = frame.copy()
                    else:
                        logger.warning("No se pudo leer frame en thread continuo")
                        time.sleep(0.5)

                        # Intentar reconectar
                        if not self.cap.isOpened():
                            logger.warning("Stream cerrado, intentando reconectar...")
                            self.inicializar_camara()
                else:
                    logger.warning("Cámara no disponible en thread")
                    time.sleep(1)

                # Pausa mínima para no saturar CPU (lee ~100 FPS máximo)
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error en thread de lectura: {e}")
                time.sleep(1)

        logger.info("Thread de lectura continua detenido")

    def iniciar_lectura_continua(self):
        """Iniciar el thread de lectura continua"""
        if self.thread_lectura is not None and self.thread_lectura.is_alive():
            logger.warning("Thread de lectura ya está corriendo")
            return

        self.thread_lectura_running = True
        self.thread_lectura = threading.Thread(target=self._leer_frames_continuamente)
        self.thread_lectura.daemon = True
        self.thread_lectura.start()
        logger.info("Thread de lectura continua iniciado")

    def detener_lectura_continua(self):
        """Detener el thread de lectura continua"""
        if self.thread_lectura is not None:
            self.thread_lectura_running = False
            self.thread_lectura.join(timeout=3)
            logger.info("Thread de lectura continua detenido")

    def verificar_camara(self) -> bool:
        """Verificar si la cámara está activa y reconectarla si es necesario"""
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Cámara desconectada, intentando reconectar...")
            return self.inicializar_camara()
        return True

    def cargar_modelo(self) -> bool:
        """Cargar modelo YOLO"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Modelo YOLO cargado desde {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar modelo YOLO: {e}")
            return False

    def autenticar(self) -> bool:
        """Autenticar con la API y obtener token JWT"""
        try:
            response = requests.post(
                f"{self.api_base_url}/token",
                data={
                    "username": self.username,
                    "password": self.password
                }
            )

            if response.status_code == 200:
                self.token = response.json()["access_token"]
                logger.info("Autenticación exitosa")
                return True
            else:
                logger.error(f"Error de autenticación: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error al autenticar: {e}")
            return False

    def obtener_headers(self) -> Dict[str, str]:
        """Obtener headers con token de autenticación"""
        return {
            "Authorization": f"Bearer {self.token}"
        }

    def capturar_frame(self) -> Optional[np.ndarray]:
        """
        Obtener el frame más reciente del thread de lectura continua.
        Ya no lee directamente de la cámara.
        """
        with self.frame_lock:
            if self.frame_actual is not None:
                return self.frame_actual.copy()
            else:
                logger.warning("No hay frame disponible aún")
                return None

    def detectar_objetos(self, frame: np.ndarray) -> List[Dict]:
        """Ejecutar detección YOLO en un frame"""
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            detecciones = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confianza = float(box.conf[0])

                    detecciones.append({
                        "confianza": confianza,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

            return detecciones

        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return []

    def guardar_frame_temporal(self, frame: np.ndarray) -> Optional[str]:
        """Guardar frame en archivo temporal con timestamp y UUID"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            import uuid
            unique_id = uuid.uuid4().hex[:6]
            filename = self.temp_dir / f"frame_{timestamp}_{unique_id}.jpg"
            cv2.imwrite(str(filename), frame)
            return str(filename)
        except Exception as e:
            logger.error(f"Error al guardar frame: {e}")
            return None

    def crear_evento(self) -> Optional[int]:
        """Crear un nuevo evento en la API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/eventos",
                json={
                    "fecha_evento": date.today().isoformat(),
                    "descripcion": "Evento detectado automáticamente",
                    "estatus": "pendiente"
                }
            )

            if response.status_code == 201:
                evento_id = response.json()["evento_id"]
                logger.info(f"Evento creado con ID: {evento_id}")
                return evento_id
            else:
                logger.error(f"Error al crear evento: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error al crear evento: {e}")
            return None

    def subir_a_azure(self, local_path: str, file_name: str) -> Optional[str]:
        """Subir imagen a Azure Blob Storage"""
        if not self.azure_container_url or not self.azure_token_sas:
            logger.warning("Azure no configurado, saltando subida")
            return None

        blob_url = f"{self.azure_container_url}/{file_name}?{self.azure_token_sas}"

        try:
            with open(local_path, "rb") as file:
                response = requests.put(
                    blob_url,
                    data=file,
                    headers={
                        'x-ms-blob-type': 'BlockBlob',
                        'Content-Type': 'image/jpeg'
                    }
                )

            if response.status_code == 201:
                azure_url = f"{self.azure_container_url}/{file_name}"
                logger.info(f"Imagen subida a Azure: {file_name}")
                return azure_url
            else:
                logger.error(f"Error al subir a Azure: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Excepción al subir a Azure: {e}")
            return None

    def enviar_imagen_con_detecciones(
            self,
            imagen_path: str,
            detecciones: List[Dict],
            evento_id: int
    ):
        """Enviar imagen y detecciones a la API en un hilo separado"""
        def _enviar():
            try:
                # Extraer nombre del archivo
                file_name = Path(imagen_path).name

                # Subir imagen a Azure
                azure_url = self.subir_a_azure(imagen_path, file_name)

                if not azure_url:
                    logger.error("No se pudo subir imagen a Azure, abortando envío")
                    return

                # Preparar payload para la API
                payload = {
                    "imagen": {
                        "ruta_imagen": azure_url
                    },
                    "detecciones": detecciones
                }

                # Enviar a la API
                response = requests.post(
                    f"{self.api_base_url}/eventos/{evento_id}/imagenes",
                    json=payload,
                    headers=self.obtener_headers()
                )

                if response.status_code == 201:
                    logger.info(f"Imagen y detecciones enviadas al evento {evento_id}")
                else:
                    logger.error(f"Error al enviar a API: {response.status_code} - {response.text}")

                # Eliminar archivo temporal después de subir
                try:
                    if os.path.exists(imagen_path):
                        os.remove(imagen_path)
                except Exception as e:
                    logger.warning(f"No se pudo eliminar archivo temporal: {e}")

            except Exception as e:
                logger.error(f"Error al enviar imagen: {e}")

        # Ejecutar en hilo separado
        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def procesar_detecciones(
            self,
            frame: np.ndarray,
            detecciones: List[Dict]
    ):
        """Procesar detecciones y gestionar estados de eventos"""
        hay_deteccion = len(detecciones) > 0

        if not hay_deteccion:
            self.contador_con_deteccion = 0
            self.contador_sin_deteccion += 1

            # Cerrar evento si hay muchas tomas sin detección
            if self.contador_sin_deteccion >= self.umbral_cerrar_evento:
                if self.id_evento_activo is not None:
                    logger.info(f"Cerrando evento {self.id_evento_activo}")
                    self.id_evento_activo = None
                    self.estado_actual = "sin_deteccion"
                self.contador_sin_deteccion = 0

        else:
            self.contador_sin_deteccion = 0
            self.contador_con_deteccion += 1

            # Crear evento si hay suficientes detecciones seguidas
            if self.contador_con_deteccion >= self.umbral_crear_evento:
                if self.id_evento_activo is None:
                    self.id_evento_activo = self.crear_evento()
                    if self.id_evento_activo:
                        self.estado_actual = "evento_activo"
                        logger.info("Estado: EVENTO ACTIVO")

            # Si hay evento activo, enviar imagen con detecciones
            if self.id_evento_activo is not None:
                imagen_path = self.guardar_frame_temporal(frame)
                if imagen_path:
                    self.enviar_imagen_con_detecciones(
                        imagen_path,
                        detecciones,
                        self.id_evento_activo
                    )

    def obtener_tiempo_espera(self) -> int:
        """Obtener tiempo de espera según estado actual"""
        if self.estado_actual == "sin_deteccion":
            return self.tiempo_foto_sin_deteccion
        else:
            return self.tiempo_foto_con_deteccion

    def ejecutar_ciclo(self):
        """Ejecutar ciclo principal de monitoreo"""
        logger.info("Iniciando ciclo de monitoreo...")

        ultimo_capture = time.time()

        while self.running:
            try:
                tiempo_actual = time.time()
                tiempo_espera = self.obtener_tiempo_espera()

                # Capturar frame según intervalo
                if tiempo_actual - ultimo_capture >= tiempo_espera:
                    frame = self.capturar_frame()

                    if frame is not None:
                        # Detectar objetos
                        detecciones = self.detectar_objetos(frame)

                        logger.info(
                            f"Estado: {self.estado_actual} | "
                            f"Detecciones: {len(detecciones)} | "
                            f"Sin detección: {self.contador_sin_deteccion} | "
                            f"Con detección: {self.contador_con_deteccion}"
                        )

                        # Procesar resultados
                        self.procesar_detecciones(frame, detecciones)

                        ultimo_capture = tiempo_actual

                    else:
                        logger.warning("Frame no disponible, esperando...")
                        time.sleep(1)

                # Pequeña pausa para no saturar CPU
                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Interrupción por usuario")
                break
            except Exception as e:
                logger.error(f"Error en ciclo principal: {e}")
                time.sleep(5)

    def iniciar(self):
        """Iniciar cliente de monitoreo"""
        logger.info("Iniciando cliente de monitoreo térmico...")

        # Inicializar componentes
        if not self.inicializar_camara():
            logger.error("No se pudo inicializar la cámara")
            return

        if not self.cargar_modelo():
            logger.error("No se pudo cargar el modelo")
            return

        if not self.autenticar():
            logger.error("No se pudo autenticar con la API")
            return

        # Iniciar thread de lectura continua
        self.iniciar_lectura_continua()

        # Esperar un momento para que el thread capture el primer frame
        time.sleep(1)

        # Iniciar ciclo principal
        self.running = True
        try:
            self.ejecutar_ciclo()
        finally:
            self.detener()

    def detener(self):
        """Detener cliente y liberar recursos"""
        logger.info("Deteniendo cliente...")
        self.running = False

        # NUEVO: Detener thread de lectura
        self.detener_lectura_continua()

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
        logger.info("Cliente detenido")


def main():
    """Función principal para ejecutar el cliente"""

    cliente = ThermalMonitorClient(
        # rtspUrl = "rtsp://lizardi:zenobia16@10.3.57.103/cam/realmonitor?channel=2&subtype=0"
        camera_source="rtsp://192.168.0.208:8554/stream",
        api_base_url="http://4.155.33.198:8000",
        username="userweb",
        password="password",
        model_path="./modelos/beta02.pt",
        confidence_threshold=0.5
    )

    # Iniciar monitoreo
    try:
        cliente.iniciar()
    except KeyboardInterrupt:
        logger.info("Programa terminado por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")


if __name__ == "__main__":
    main()