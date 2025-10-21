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
            azure_token_sas: str = None,
            max_errores_consecutivos: int = 5,
            timeout_reconexion: int = 10,
            intervalo_heartbeat: int = 300,
            hora_inicio: int = 6,
            hora_fin: int = 21
    ):
        self.camera_source = camera_source
        self.api_base_url = api_base_url.rstrip('/')
        self.username = username
        self.password = password
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        self.azure_container_url = azure_container_url or os.getenv("AZURE_CONTAINER_URL")
        self.azure_token_sas = azure_token_sas or os.getenv("TOKENSAS")

        if not self.azure_container_url or not self.azure_token_sas:
            logger.warning("Configuración de Azure no encontrada. Las imágenes no se subirán a Azure.")
        else:
            logger.info("Configuración de Azure cargada correctamente")

        self.cap: Optional[cv2.VideoCapture] = None
        self.model: Optional[YOLO] = None
        self.token: Optional[str] = None
        self.running = False

        self.frame_actual = None
        self.frame_lock = threading.Lock()
        self.thread_lectura = None
        self.thread_lectura_running = False

        self.estado_actual = "sin_deteccion"
        self.id_evento_activo: Optional[int] = None
        self.contador_sin_deteccion = 0
        self.contador_con_deteccion = 0

        self.tiempo_foto_sin_deteccion = 5
        self.tiempo_foto_con_deteccion = 2
        self.umbral_crear_evento = 3
        self.umbral_cerrar_evento = 5

        self.max_errores_consecutivos = max_errores_consecutivos
        self.timeout_reconexion = timeout_reconexion
        self.errores_consecutivos = 0
        self.ultima_reconexion = 0

        self.intervalo_heartbeat = intervalo_heartbeat
        self.ultimo_heartbeat = 0
        self.thread_heartbeat = None
        self.thread_heartbeat_running = False

        self.hora_inicio = hora_inicio
        self.hora_fin = hora_fin

        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)

    def enviar_log_api(self, tipo: str, mensaje: str):
        """Enviar log al endpoint de la API"""
        def _enviar():
            try:
                if not self.token:
                    return

                response = requests.post(
                    f"{self.api_base_url}/logs",
                    json={
                        "tipo": tipo,
                        "mensaje": mensaje
                    },
                    headers=self.obtener_headers(),
                    timeout=10
                )

                if response.status_code != 201:
                    logger.warning(f"Error al enviar log a API: {response.status_code}")

            except Exception as e:
                logger.warning(f"Excepción al enviar log a API: {e}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def esta_en_horario_operacion(self) -> bool:
        """Verificar si se encuentra en horario de operación"""
        hora_actual = datetime.now().hour
        return self.hora_inicio <= hora_actual < self.hora_fin

    def _heartbeat_loop(self):
        """Enviar heartbeat periódico a la API"""
        logger.info("Thread de heartbeat iniciado")

        while self.thread_heartbeat_running:
            try:
                tiempo_actual = time.time()

                if tiempo_actual - self.ultimo_heartbeat >= self.intervalo_heartbeat:
                    en_horario = self.esta_en_horario_operacion()

                    if en_horario:
                        mensaje_heartbeat = (
                            f"Sistema operando normalmente. "
                            f"Estado: {self.estado_actual}, "
                            f"Errores stream: {self.errores_consecutivos}, "
                            f"Evento activo: {self.id_evento_activo is not None}"
                        )
                    else:
                        hora_actual = datetime.now().strftime("%H:%M")
                        mensaje_heartbeat = (
                            f"Sistema fuera de horario operativo ({hora_actual}). "
                            f"Horario: {self.hora_inicio:02d}:00 - {self.hora_fin:02d}:00. "
                            f"Monitoreo en espera, conexión activa."
                        )

                    self.enviar_log_api("info", mensaje_heartbeat)
                    logger.info(f"Heartbeat enviado a API - En horario: {en_horario}")
                    self.ultimo_heartbeat = tiempo_actual

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error en thread de heartbeat: {e}")
                time.sleep(30)

        logger.info("Thread de heartbeat detenido")

    def iniciar_heartbeat(self):
        """Iniciar thread de heartbeat"""
        if self.thread_heartbeat is not None and self.thread_heartbeat.is_alive():
            logger.warning("Thread de heartbeat ya está corriendo")
            return

        self.ultimo_heartbeat = time.time()
        self.thread_heartbeat_running = True
        self.thread_heartbeat = threading.Thread(target=self._heartbeat_loop)
        self.thread_heartbeat.daemon = True
        self.thread_heartbeat.start()
        logger.info("Thread de heartbeat iniciado")

    def detener_heartbeat(self):
        """Detener thread de heartbeat"""
        if self.thread_heartbeat is not None:
            self.thread_heartbeat_running = False
            self.thread_heartbeat.join(timeout=5)
            logger.info("Thread de heartbeat detenido")

    def inicializar_camara(self) -> bool:
        """Inicializar conexión con la cámara"""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"Error al liberar cámara anterior: {e}")
                self.cap = None

            tiempo_desde_ultima = time.time() - self.ultima_reconexion
            if tiempo_desde_ultima < 2:
                time.sleep(2 - tiempo_desde_ultima)

            logger.info(f"Conectando a cámara: {self.camera_source}")
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                error_msg = f"No se pudo abrir la cámara RTSP: {self.camera_source}"
                logger.error(error_msg)
                self.enviar_log_api("error", error_msg)
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                error_msg = "No se pudo leer frame de prueba de la cámara RTSP"
                logger.error(error_msg)
                self.enviar_log_api("error", error_msg)
                self.cap.release()
                self.cap = None
                return False

            logger.info(f"Cámara inicializada - Resolución: {frame.shape[1]}x{frame.shape[0]}")
            self.enviar_log_api("info", f"Cámara RTSP conectada exitosamente - Resolución: {frame.shape[1]}x{frame.shape[0]}")

            self.ultima_reconexion = time.time()
            self.errores_consecutivos = 0

            return True

        except Exception as e:
            error_msg = f"Error crítico al inicializar cámara RTSP: {str(e)}"
            logger.error(error_msg)
            self.enviar_log_api("error", error_msg)
            self.cap = None
            return False

    def _verificar_estado_stream(self) -> bool:
        """Verificar si el stream está en buen estado"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return False

            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if width <= 0 or height <= 0:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error al verificar estado del stream: {e}")
            return False

    def _intentar_reconexion(self) -> bool:
        """Intentar reconectar al stream"""
        advertencia_msg = f"Intentando reconexión al stream RTSP. Errores consecutivos: {self.errores_consecutivos}"
        logger.warning(advertencia_msg)
        self.enviar_log_api("advertencia", advertencia_msg)

        tiempo_actual = time.time()

        if tiempo_actual - self.ultima_reconexion < self.timeout_reconexion:
            tiempo_espera = self.timeout_reconexion - (tiempo_actual - self.ultima_reconexion)
            logger.info(f"Esperando {tiempo_espera:.1f}s antes de reconectar...")
            time.sleep(tiempo_espera)

        resultado = self.inicializar_camara()

        if not resultado:
            error_msg = f"Reconexión fallida al stream RTSP después de {self.errores_consecutivos} intentos"
            logger.error(error_msg)
            self.enviar_log_api("error", error_msg)

        return resultado

    def _leer_frames_continuamente(self):
        """Leer frames continuamente del stream RTSP con reconexión automática"""
        logger.info("Thread de lectura continua iniciado")

        while self.thread_lectura_running:
            try:
                # Verificar si está en horario de operación
                if not self.esta_en_horario_operacion():
                    # Fuera de horario, solo dormir y continuar
                    time.sleep(10)
                    continue

                if not self._verificar_estado_stream():
                    logger.warning("Stream en mal estado, intentando reconectar...")
                    if self._intentar_reconexion():
                        logger.info("Reconexión exitosa")
                        self.errores_consecutivos = 0
                        continue
                    else:
                        logger.error("Reconexión fallida, esperando antes de reintentar...")
                        self.errores_consecutivos += 1
                        time.sleep(5)
                        continue

                ret, frame = self.cap.read()

                if ret and frame is not None and frame.size > 0:
                    with self.frame_lock:
                        self.frame_actual = frame.copy()

                    self.errores_consecutivos = 0

                else:
                    self.errores_consecutivos += 1
                    logger.warning(f"No se pudo leer frame (error {self.errores_consecutivos}/{self.max_errores_consecutivos})")

                    if self.errores_consecutivos >= self.max_errores_consecutivos:
                        error_msg = f"Demasiados errores consecutivos ({self.errores_consecutivos}) en lectura de stream RTSP, forzando reconexión"
                        logger.error(error_msg)
                        self.enviar_log_api("error", error_msg)

                        if self._intentar_reconexion():
                            logger.info("Reconexión exitosa después de errores")
                            self.enviar_log_api("info", "Reconexión exitosa al stream RTSP después de múltiples errores")
                            self.errores_consecutivos = 0
                        else:
                            logger.error("Reconexión fallida después de errores")
                            time.sleep(5)
                    else:
                        time.sleep(0.5)

                time.sleep(0.01)

            except Exception as e:
                self.errores_consecutivos += 1
                error_msg = f"Excepción en thread de lectura RTSP ({self.errores_consecutivos}): {str(e)}"
                logger.error(error_msg)
                self.enviar_log_api("error", error_msg)

                if self.errores_consecutivos >= self.max_errores_consecutivos:
                    self._intentar_reconexion()

                time.sleep(1)

        logger.info("Thread de lectura continua detenido")

    def iniciar_lectura_continua(self):
        """Iniciar el thread de lectura continua"""
        if self.thread_lectura is not None and self.thread_lectura.is_alive():
            logger.warning("Thread de lectura ya está corriendo")
            return

        self.errores_consecutivos = 0
        self.thread_lectura_running = True
        self.thread_lectura = threading.Thread(target=self._leer_frames_continuamente)
        self.thread_lectura.daemon = True
        self.thread_lectura.start()
        logger.info("Thread de lectura continua iniciado")

    def detener_lectura_continua(self):
        """Detener el thread de lectura continua"""
        if self.thread_lectura is not None:
            self.thread_lectura_running = False
            self.thread_lectura.join(timeout=5)
            logger.info("Thread de lectura continua detenido")

    def verificar_camara(self) -> bool:
        """Verificar si la cámara está activa y reconectarla si es necesario"""
        if not self._verificar_estado_stream():
            logger.warning("Cámara desconectada, intentando reconectar...")
            return self._intentar_reconexion()
        return True

    def cargar_modelo(self) -> bool:
        """Cargar modelo YOLO"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Modelo YOLO cargado desde {self.model_path}")
            self.enviar_log_api("info", f"Modelo YOLO cargado exitosamente desde {self.model_path}")
            return True
        except Exception as e:
            error_msg = f"Error al cargar modelo YOLO: {str(e)}"
            logger.error(error_msg)
            self.enviar_log_api("error", error_msg)
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
                error_msg = f"Error de autenticación con API: {response.status_code}"
                logger.error(error_msg)
                return False

        except Exception as e:
            error_msg = f"Error al autenticar con API: {str(e)}"
            logger.error(error_msg)
            return False

    def obtener_headers(self) -> Dict[str, str]:
        """Obtener headers con token de autenticación"""
        return {
            "Authorization": f"Bearer {self.token}"
        }

    def capturar_frame(self) -> Optional[np.ndarray]:
        """Obtener el frame más reciente del thread de lectura continua"""
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
                self.enviar_log_api("info", f"Nuevo evento creado con ID: {evento_id}")
                return evento_id
            else:
                error_msg = f"Error al crear evento en API: {response.status_code}"
                logger.error(error_msg)
                self.enviar_log_api("error", error_msg)
                return None

        except Exception as e:
            error_msg = f"Excepción al crear evento: {str(e)}"
            logger.error(error_msg)
            self.enviar_log_api("error", error_msg)
            return None

    def subir_a_azure(self, local_path: str, file_name: str) -> Optional[str]:
        """Subir imagen a Azure Blob Storage"""
        if not self.azure_container_url or not self.azure_token_sas:
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
                return azure_url
            else:
                return None

        except Exception as e:
            logger.error(f"Excepción Azure: {e}")
            return None

    def enviar_imagen_con_detecciones(
            self,
            imagen_path: str,
            detecciones: List[Dict],
            evento_id: int
    ):
        """Enviar imagen y detecciones a la API en un hilo separado"""
        def _enviar():
            file_name = Path(imagen_path).name
            resultado = {
                'archivo': file_name,
                'evento_id': evento_id,
                'detecciones': len(detecciones),
                'azure_subida': False,
                'api_enviada': False,
                'archivo_eliminado': False,
                'error': None
            }

            try:
                azure_url = self.subir_a_azure(imagen_path, file_name)

                if not azure_url:
                    resultado['error'] = 'Fallo subida Azure'
                    logger.error(f"PIPELINE FALLIDO | {resultado}")
                    self.enviar_log_api("error", f"Fallo al subir imagen a Azure: {file_name}")
                    return

                resultado['azure_subida'] = True

                payload = {
                    "imagen": {
                        "ruta_imagen": azure_url
                    },
                    "detecciones": detecciones
                }

                response = requests.post(
                    f"{self.api_base_url}/eventos/{evento_id}/imagenes",
                    json=payload,
                    headers=self.obtener_headers()
                )

                if response.status_code == 201:
                    resultado['api_enviada'] = True
                else:
                    resultado['error'] = f'API error {response.status_code}'
                    logger.error(f"PIPELINE FALLIDO | {resultado}")
                    self.enviar_log_api("error", f"Error al enviar imagen a API: {response.status_code} - {file_name}")
                    return

                try:
                    if os.path.exists(imagen_path):
                        os.remove(imagen_path)
                        resultado['archivo_eliminado'] = True
                except Exception as e:
                    resultado['error'] = f'No se eliminó temp: {str(e)}'

                logger.info(f"PIPELINE EXITOSO | {resultado}")

            except Exception as e:
                resultado['error'] = str(e)
                logger.error(f"PIPELINE FALLIDO | {resultado}")
                self.enviar_log_api("error", f"Excepción en pipeline de imagen: {str(e)}")

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

            if self.contador_sin_deteccion >= self.umbral_cerrar_evento:
                if self.id_evento_activo is not None:
                    logger.info(f"Cerrando evento {self.id_evento_activo}")
                    self.enviar_log_api("info", f"Evento cerrado: {self.id_evento_activo} - Sin detecciones por tiempo prolongado")
                    self.id_evento_activo = None
                    self.estado_actual = "sin_deteccion"
                self.contador_sin_deteccion = 0

        else:
            self.contador_sin_deteccion = 0
            self.contador_con_deteccion += 1

            if self.contador_con_deteccion >= self.umbral_crear_evento:
                if self.id_evento_activo is None:
                    self.id_evento_activo = self.crear_evento()
                    if self.id_evento_activo:
                        self.estado_actual = "evento_activo"
                        logger.info("Estado: EVENTO ACTIVO")

            if self.id_evento_activo is not None:
                imagen_path = self.guardar_frame_temporal(frame)
                if imagen_path:
                    logger.info(f"CAPTURA | Archivo: {Path(imagen_path).name} | Evento: {self.id_evento_activo} | Detecciones: {len(detecciones)}")
                    self.enviar_imagen_con_detecciones(
                        imagen_path,
                        detecciones,
                        self.id_evento_activo
                    )
                else:
                    logger.error(f"CAPTURA FALLIDA | Evento: {self.id_evento_activo} | Detecciones: {len(detecciones)}")

    def obtener_tiempo_espera(self) -> int:
        """Obtener tiempo de espera según estado actual"""
        if self.estado_actual == "sin_deteccion":
            return self.tiempo_foto_sin_deteccion
        else:
            return self.tiempo_foto_con_deteccion

    def ejecutar_ciclo(self):
        """Ejecutar ciclo principal de monitoreo"""
        logger.info("Iniciando ciclo de monitoreo...")
        self.enviar_log_api("info", "Sistema de monitoreo térmico iniciado correctamente")

        ultimo_capture = time.time()
        ultimo_log_fuera_horario = 0

        while self.running:
            try:
                # Verificar si está en horario de operación
                if not self.esta_en_horario_operacion():
                    # Mostrar mensaje periódico de que está fuera de horario
                    tiempo_actual = time.time()
                    if tiempo_actual - ultimo_log_fuera_horario >= 300:
                        hora_actual = datetime.now().strftime("%H:%M")
                        logger.info(f"Fuera de horario operativo ({hora_actual}). Esperando horario {self.hora_inicio:02d}:00 - {self.hora_fin:02d}:00")
                        ultimo_log_fuera_horario = tiempo_actual

                    time.sleep(30)
                    continue

                tiempo_actual = time.time()
                tiempo_espera = self.obtener_tiempo_espera()

                if tiempo_actual - ultimo_capture >= tiempo_espera:
                    frame = self.capturar_frame()

                    if frame is not None:
                        detecciones = self.detectar_objetos(frame)

                        logger.info(
                            f"Estado: {self.estado_actual} | "
                            f"Detecciones: {len(detecciones)} | "
                            f"Sin detección: {self.contador_sin_deteccion} | "
                            f"Con detección: {self.contador_con_deteccion} | "
                            f"Errores stream: {self.errores_consecutivos}"
                        )

                        self.procesar_detecciones(frame, detecciones)

                        ultimo_capture = tiempo_actual

                    else:
                        logger.warning("Frame no disponible, esperando...")
                        time.sleep(1)

                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Interrupción por usuario")
                self.enviar_log_api("advertencia", "Sistema detenido por usuario")
                break
            except Exception as e:
                error_msg = f"Error en ciclo principal: {str(e)}"
                logger.error(error_msg)
                self.enviar_log_api("error", error_msg)
                time.sleep(5)

    def iniciar(self):
        """Iniciar cliente de monitoreo"""
        logger.info("Iniciando cliente de monitoreo térmico...")
        logger.info(f"Horario de operación configurado: {self.hora_inicio:02d}:00 - {self.hora_fin:02d}:00")

        if not self.inicializar_camara():
            logger.error("No se pudo inicializar la cámara")
            return

        if not self.cargar_modelo():
            logger.error("No se pudo cargar el modelo")
            return

        if not self.autenticar():
            logger.error("No se pudo autenticar con la API")
            return

        self.iniciar_lectura_continua()
        self.iniciar_heartbeat()

        time.sleep(2)

        self.running = True
        try:
            self.ejecutar_ciclo()
        finally:
            self.detener()

    def detener(self):
        """Detener cliente y liberar recursos"""
        logger.info("Deteniendo cliente...")
        self.enviar_log_api("advertencia", "Sistema de monitoreo térmico detenido")
        self.running = False

        self.detener_heartbeat()
        self.detener_lectura_continua()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error al liberar cámara: {e}")
            self.cap = None

        cv2.destroyAllWindows()
        logger.info("Cliente detenido")


def main():
    """Ejecutar el cliente"""

    cliente = ThermalMonitorClient(
        # rtspUrl = "rtsp://lizardi:zenobia16@10.3.57.103/cam/realmonitor?channel=2&subtype=0"
        camera_source="rtsp://192.168.0.208:8554/stream",
        api_base_url="http://4.155.33.198:8000",
        username="userweb",
        password="password",
        model_path="./modelos/beta02.pt",
        confidence_threshold=0.5,
        max_errores_consecutivos=5,
        timeout_reconexion=10,
        intervalo_heartbeat=300,
        hora_inicio=6,
        hora_fin=21
    )

    try:
        cliente.iniciar()
    except KeyboardInterrupt:
        logger.info("Programa terminado por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")


if __name__ == "__main__":
    main()