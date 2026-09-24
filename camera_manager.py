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

        self._conexion_fallida = False
        self.on_error_callback = None
        self.on_reconnect_callback = None

    def set_callbacks(self, on_error=None, on_reconnect=None):
        """Configurar callbacks para eventos"""
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def _reportar_desconexion(self):
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

            tiempo_desde_ultima = time.time() - self.ultima_reconexion
            if tiempo_desde_ultima < 2:
                time.sleep(2 - tiempo_desde_ultima)

            logger.debug("Conectando a camara")
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                error_msg = "No se pudo abrir la camara RTSP"
                logger.debug(error_msg)
                self._reportar_desconexion()
                return False

            ret, frame = self.cap.read()
            if not ret or frame is None:
                error_msg = "No se pudo leer frame de prueba de la camara RTSP"
                logger.debug(error_msg)
                self._reportar_desconexion()
                self.cap.release()
                self.cap = None
                return False

            if self._conexion_fallida:
                mensaje = "Conexion con camara recuperada"
                logger.info(mensaje)
                if self.on_reconnect_callback:
                    self.on_reconnect_callback(mensaje)
            else:
                logger.debug("Camara inicializada")
            self._conexion_fallida = False

            self.ultima_reconexion = time.time()
            self.errores_consecutivos = 0

            return True

        except Exception as e:
            error_msg = f"Error critico al inicializar camara RTSP: {type(e).__name__}"
            logger.debug(error_msg)
            self._reportar_desconexion()
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

        tiempo_actual = time.time()

        if tiempo_actual - self.ultima_reconexion < self.timeout_reconexion:
            tiempo_espera = self.timeout_reconexion - (tiempo_actual - self.ultima_reconexion)
            logger.debug(f"Esperando {tiempo_espera:.1f}s antes de reconectar...")
            time.sleep(tiempo_espera)

        resultado = self.inicializar()

        return resultado

    def _leer_frames_continuamente(self, en_horario_callback):
        """Leer frames continuamente del stream RTSP con reconexion automatica"""
        logger.debug("Thread de lectura continua iniciado")

        while self.thread_lectura_running:
            try:
                if not en_horario_callback():
                    time.sleep(10)
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
                        time.sleep(5)
                        continue

                ret, frame = self.cap.read()

                if ret and frame is not None and frame.size > 0:
                    with self.frame_lock:
                        self.frame_actual = frame.copy()

                    if self._conexion_fallida:
                        mensaje = "Conexion con camara recuperada"
                        logger.info(mensaje)
                        if self.on_reconnect_callback:
                            self.on_reconnect_callback(mensaje)
                        self._conexion_fallida = False
                    self.errores_consecutivos = 0

                else:
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
                            time.sleep(5)
                    else:
                        time.sleep(0.5)

                time.sleep(0.01)

            except Exception as e:
                self.errores_consecutivos += 1
                error_msg = f"Excepcion en thread de lectura RTSP ({self.errores_consecutivos}): {type(e).__name__}"
                logger.debug(error_msg)
                self._reportar_desconexion()

                if self.errores_consecutivos >= self.max_errores_consecutivos:
                    self._intentar_reconexion()

                time.sleep(1)

        logger.debug("Thread de lectura continua detenido")

    def iniciar_lectura_continua(self, en_horario_callback):
        """Iniciar el thread de lectura continua"""
        if self.thread_lectura is not None and self.thread_lectura.is_alive():
            logger.debug("Thread de lectura ya esta corriendo")
            return

        self.errores_consecutivos = 0
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
        if self.thread_lectura is not None:
            self.thread_lectura_running = False
            self.thread_lectura.join(timeout=5)
            logger.debug("Thread de lectura continua detenido")

    def obtener_frame(self) -> Optional[np.ndarray]:
        """Obtener el frame mas reciente del thread de lectura continua"""
        with self.frame_lock:
            if self.frame_actual is not None:
                return self.frame_actual.copy()
            else:
                logger.debug("No hay frame disponible aun")
                return None

    def liberar(self):
        """Liberar recursos de la camara"""
        self.detener_lectura_continua()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error al liberar camara: {type(e).__name__}")
            self.cap = None

        cv2.destroyAllWindows()
        logger.debug("Camara liberada")