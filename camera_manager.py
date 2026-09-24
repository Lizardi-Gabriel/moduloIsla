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
            timeout_reconexion: int = 10,
            max_antiguedad_frame: float = 5
    ):
        self.camera_source = camera_source
        self.max_errores_consecutivos = max_errores_consecutivos
        self.timeout_reconexion = timeout_reconexion
        self.max_antiguedad_frame = max_antiguedad_frame
        self.ultimo_frame_recibido = None
        self._stop = threading.Event()

        self.cap: Optional[cv2.VideoCapture] = None

        self.frame_actual = None
        self.frame_lock = threading.Lock()

        self.thread_lectura = None
        self.thread_lectura_running = False

        self.errores_consecutivos = 0
        self.ultima_reconexion = None

        self._conexion_fallida = False
        self.on_error_callback = None
        self.on_reconnect_callback = None

    def set_callbacks(self, on_error=None, on_reconnect=None):
        """Configurar callbacks para eventos"""
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def _reportar_desconexion(self):
        self._invalidar_frame()
        if self._conexion_fallida:
            return
        self._conexion_fallida = True
        mensaje = "Camara sin conexion utilizable; intentando recuperar el stream"
        logger.error(mensaje, extra={"state_transition": True})
        if self.on_error_callback:
            self.on_error_callback("error", mensaje)

    def inicializar(self) -> bool:
        """Inicializar conexion con la camara"""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"Error al liberar camara anterior: {type(e).__name__}")
                self.cap = None

            self._invalidar_frame()
            self.ultima_reconexion = time.monotonic()
            logger.debug("Conectando a camara")
            self.cap = cv2.VideoCapture(
                self.camera_source, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
            )

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                error_msg = "No se pudo abrir la camara RTSP"
                logger.debug(error_msg)
                self._reportar_desconexion()
                self.cap.release()
                self.cap = None
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                error_msg = "No se pudo leer frame de prueba de la camara RTSP"
                logger.debug(error_msg)
                self._reportar_desconexion()
                self.cap.release()
                self.cap = None
                return False

            self._registrar_frame(frame)
            return True


        except Exception as e:
            error_msg = f"Error critico al inicializar camara RTSP: {type(e).__name__}"
            logger.debug(error_msg)
            self._reportar_desconexion()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
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
            logger.warning(f"Error al verificar estado del stream: {type(e).__name__}")
            return False

    def _intentar_reconexion(self) -> bool:
        """Intentar reconectar al stream"""
        self._reportar_desconexion()

        if self.ultima_reconexion is not None:
            espera = self.timeout_reconexion - (time.monotonic() - self.ultima_reconexion)
            if espera > 0 and self._stop.wait(espera):
                return False
        if self._stop.is_set():
            return False
        return self.inicializar()

    def _leer_frames_continuamente(self, en_horario_callback):
        """Leer frames continuamente del stream RTSP con reconexion automatica"""
        logger.debug("Thread de lectura continua iniciado")

        while self.thread_lectura_running:
            try:
                if not en_horario_callback():
                    self._stop.wait(10)
                    continue

                if not self._verificar_estado_stream():
                    logger.debug("Stream en mal estado, intentando reconectar...")
                    if self._intentar_reconexion():
                        logger.debug("Reconexion exitosa")
                        self.errores_consecutivos = 0
                        continue
                    else:
                        logger.debug("Reconexion fallida, esperando antes de reintentar...")
                        self.errores_consecutivos += 1
                        self._stop.wait(5)
                        continue

                ret, frame = self.cap.read()

                if ret and frame is not None and frame.size > 0:
                    self._registrar_frame(frame)

                else:
                    self._invalidar_frame()
                    self.errores_consecutivos += 1
                    logger.debug(f"No se pudo leer frame (error {self.errores_consecutivos}/{self.max_errores_consecutivos})")

                    if self.errores_consecutivos >= self.max_errores_consecutivos:
                        error_msg = f"Demasiados errores consecutivos ({self.errores_consecutivos}) en lectura de stream RTSP, forzando reconexion"
                        logger.debug(error_msg)
                        self._reportar_desconexion()

                        if self._intentar_reconexion():
                            logger.debug("Reconexion exitosa despues de errores")
                            self.errores_consecutivos = 0
                        else:
                            logger.debug("Reconexion fallida despues de errores")
                            self._stop.wait(5)
                    else:
                        self._stop.wait(0.5)

                self._stop.wait(0.01)

            except Exception as e:
                self.errores_consecutivos += 1
                error_msg = f"Excepcion en thread de lectura RTSP ({self.errores_consecutivos}): {type(e).__name__}"
                logger.debug(error_msg)
                self._reportar_desconexion()

                if self.errores_consecutivos >= self.max_errores_consecutivos:
                    self._intentar_reconexion()

                self._stop.wait(1)

        logger.debug("Thread de lectura continua detenido")

    def iniciar_lectura_continua(self, en_horario_callback):
        """Iniciar el thread de lectura continua"""
        if self.thread_lectura is not None and self.thread_lectura.is_alive():
            logger.debug("Thread de lectura ya esta corriendo")
            return

        self.errores_consecutivos = 0
        self._stop.clear()
        self.thread_lectura_running = True
        self.thread_lectura = threading.Thread(
            target=self._leer_frames_continuamente,
            args=(en_horario_callback,)
        )
        self.thread_lectura.daemon = True
        self.thread_lectura.start()
        logger.debug("Thread de lectura continua iniciado")

    def detener_lectura_continua(self):
        """Detener el thread de lectura continua"""
        self.thread_lectura_running = False
        self._stop.set()
        if self.thread_lectura is not None:
            self.thread_lectura.join(timeout=12)
            logger.debug("Thread de lectura continua detenido")

    def _invalidar_frame(self):
        with self.frame_lock:
            self.frame_actual = None

    def _registrar_frame(self, frame):
        with self.frame_lock:
            self.frame_actual = frame.copy()
            self.ultimo_frame_recibido = time.monotonic()
        if self._conexion_fallida:
            mensaje = "Conexion con camara recuperada"
            logger.info(mensaje)
            if self.on_reconnect_callback:
                self.on_reconnect_callback(mensaje)
        self._conexion_fallida = False
        self.errores_consecutivos = 0

    def obtener_estado(self):
        """Salud basada en recepcion local; no mide la fecha de captura del dispositivo."""
        with self.frame_lock:
            edad = (None if self.ultimo_frame_recibido is None else
                    max(0, time.monotonic() - self.ultimo_frame_recibido))
            disponible = self.frame_actual is not None and edad is not None and edad <= self.max_antiguedad_frame
            return {"disponible": disponible, "antiguedad_segundos": edad}

    def obtener_frame(self) -> Optional[np.ndarray]:
        """Entregar solamente imagenes recibidas dentro del limite de antiguedad."""
        with self.frame_lock:
            if self.frame_actual is None or self.ultimo_frame_recibido is None:
                return None
            if time.monotonic() - self.ultimo_frame_recibido > self.max_antiguedad_frame:
                self.frame_actual = None
                return None
            return self.frame_actual.copy()

    def liberar(self):
        """Liberar recursos de la camara"""
        self.detener_lectura_continua()
        self._invalidar_frame()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error al liberar camara: {type(e).__name__}")
            self.cap = None

        cv2.destroyAllWindows()
        logger.debug("Camara liberada")