import cv2
import time
import threading
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Gestionar conexion y lectura de stream RTSP"""

    def __init__(
            self,
            camera_source: str,
            max_errores_consecutivos: int = 5,
            timeout_reconexion: int = 10
    ):
        self.camera_source = camera_source
        self.max_errores_consecutivos = max_errores_consecutivos
        self.timeout_reconexion = timeout_reconexion

        self.cap: Optional[cv2.VideoCapture] = None

        self.frame_actual = None
        self.frame_lock = threading.Lock()

        self.thread_lectura = None
        self.thread_lectura_running = False

        self.errores_consecutivos = 0
        self.ultima_reconexion = 0

        self.on_error_callback = None
        self.on_reconnect_callback = None

    def set_callbacks(self, on_error=None, on_reconnect=None):
        """Configurar callbacks para eventos"""
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def inicializar(self) -> bool:
        """Inicializar conexion con la camara"""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"Error al liberar camara anterior: {e}")
                self.cap = None

            tiempo_desde_ultima = time.time() - self.ultima_reconexion
            if tiempo_desde_ultima < 2:
                time.sleep(2 - tiempo_desde_ultima)

            logger.info(f"Conectando a camara: {self.camera_source}")
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                error_msg = f"No se pudo abrir la camara RTSP: {self.camera_source}"
                logger.error(error_msg)
                if self.on_error_callback:
                    self.on_error_callback("error", error_msg)
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                error_msg = "No se pudo leer frame de prueba de la camara RTSP"
                logger.error(error_msg)
                if self.on_error_callback:
                    self.on_error_callback("error", error_msg)
                self.cap.release()
                self.cap = None
                return False

            logger.info(f"Camara inicializada - Resolucion: {frame.shape[1]}x{frame.shape[0]}")
            if self.on_reconnect_callback:
                self.on_reconnect_callback(f"Camara RTSP conectada exitosamente - Resolucion: {frame.shape[1]}x{frame.shape[0]}")

            self.ultima_reconexion = time.time()
            self.errores_consecutivos = 0

            return True

        except Exception as e:
            error_msg = f"Error critico al inicializar camara RTSP: {str(e)}"
            logger.error(error_msg)
            if self.on_error_callback:
                self.on_error_callback("error", error_msg)
            self.cap = None
            return False

    def _verificar_estado_stream(self) -> bool:
        """Verificar si el stream esta en buen estado"""
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
        advertencia_msg = f"Intentando reconexion al stream RTSP. Errores consecutivos: {self.errores_consecutivos}"
        logger.warning(advertencia_msg)
        if self.on_error_callback:
            self.on_error_callback("advertencia", advertencia_msg)

        tiempo_actual = time.time()

        if tiempo_actual - self.ultima_reconexion < self.timeout_reconexion:
            tiempo_espera = self.timeout_reconexion - (tiempo_actual - self.ultima_reconexion)
            logger.info(f"Esperando {tiempo_espera:.1f}s antes de reconectar...")
            time.sleep(tiempo_espera)

        resultado = self.inicializar()

        if not resultado:
            error_msg = f"Reconexion fallida al stream RTSP despues de {self.errores_consecutivos} intentos"
            logger.error(error_msg)
            if self.on_error_callback:
                self.on_error_callback("error", error_msg)

        return resultado

    def _leer_frames_continuamente(self, en_horario_callback):
        """Leer frames continuamente del stream RTSP con reconexion automatica"""
        logger.info("Thread de lectura continua iniciado")

        while self.thread_lectura_running:
            try:
                if not en_horario_callback():
                    time.sleep(10)
                    continue

                if not self._verificar_estado_stream():
                    logger.warning("Stream en mal estado, intentando reconectar...")
                    if self._intentar_reconexion():
                        logger.info("Reconexion exitosa")
                        self.errores_consecutivos = 0
                        continue
                    else:
                        logger.error("Reconexion fallida, esperando antes de reintentar...")
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
                        error_msg = f"Demasiados errores consecutivos ({self.errores_consecutivos}) en lectura de stream RTSP, forzando reconexion"
                        logger.error(error_msg)
                        if self.on_error_callback:
                            self.on_error_callback("error", error_msg)

                        if self._intentar_reconexion():
                            logger.info("Reconexion exitosa despues de errores")
                            if self.on_reconnect_callback:
                                self.on_reconnect_callback("Reconexion exitosa al stream RTSP despues de multiples errores")
                            self.errores_consecutivos = 0
                        else:
                            logger.error("Reconexion fallida despues de errores")
                            time.sleep(5)
                    else:
                        time.sleep(0.5)

                time.sleep(0.01)

            except Exception as e:
                self.errores_consecutivos += 1
                error_msg = f"Excepcion en thread de lectura RTSP ({self.errores_consecutivos}): {str(e)}"
                logger.error(error_msg)
                if self.on_error_callback:
                    self.on_error_callback("error", error_msg)

                if self.errores_consecutivos >= self.max_errores_consecutivos:
                    self._intentar_reconexion()

                time.sleep(1)

        logger.info("Thread de lectura continua detenido")

    def iniciar_lectura_continua(self, en_horario_callback):
        """Iniciar el thread de lectura continua"""
        if self.thread_lectura is not None and self.thread_lectura.is_alive():
            logger.warning("Thread de lectura ya esta corriendo")
            return

        self.errores_consecutivos = 0
        self.thread_lectura_running = True
        self.thread_lectura = threading.Thread(
            target=self._leer_frames_continuamente,
            args=(en_horario_callback,)
        )
        self.thread_lectura.daemon = True
        self.thread_lectura.start()
        logger.info("Thread de lectura continua iniciado")

    def detener_lectura_continua(self):
        """Detener el thread de lectura continua"""
        if self.thread_lectura is not None:
            self.thread_lectura_running = False
            self.thread_lectura.join(timeout=5)
            logger.info("Thread de lectura continua detenido")

    def obtener_frame(self) -> Optional[np.ndarray]:
        """Obtener el frame mas reciente del thread de lectura continua"""
        with self.frame_lock:
            if self.frame_actual is not None:
                return self.frame_actual.copy()
            else:
                logger.warning("No hay frame disponible aun")
                return None

    def liberar(self):
        """Liberar recursos de la camara"""
        self.detener_lectura_continua()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error al liberar camara: {e}")
            self.cap = None

        cv2.destroyAllWindows()
        logger.info("Camara liberada")