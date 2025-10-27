import cv2
import time
import threading
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

from config import Config
from camera_manager import CameraManager
from detection_service import DetectionService
from api_client import APIClient
from azure_storage import AzureStorage

logger = logging.getLogger(__name__)


class ThermalMonitor:
    """Sistema de monitoreo termico con deteccion automatica de eventos"""

    def __init__(self, config: Config):
        self.config = config

        self.camera = CameraManager(
            camera_source=config.camera_source,
            max_errores_consecutivos=config.max_errores_consecutivos,
            timeout_reconexion=config.timeout_reconexion
        )

        self.detector = DetectionService(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold
        )

        self.api = APIClient(
            api_base_url=config.api_base_url,
            username=config.username,
            password=config.password
        )

        self.azure = AzureStorage(
            container_url=config.azure_container_url,
            token_sas=config.azure_token_sas
        )

        self.camera.set_callbacks(
            on_error=self.api.enviar_log,
            on_reconnect=lambda msg: self.api.enviar_log("info", msg)
        )

        self.running = False

        self.estado_actual = "sin_deteccion"
        self.id_evento_activo: Optional[int] = None
        self.contador_sin_deteccion = 0
        self.contador_con_deteccion = 0

        self.thread_heartbeat = None
        self.thread_heartbeat_running = False
        self.ultimo_heartbeat = 0

        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def esta_en_horario_operacion(self) -> bool:
        """Verificar si se encuentra en horario de operacion"""
        hora_actual = datetime.now().hour
        return self.config.hora_inicio <= hora_actual < self.config.hora_fin

    def _heartbeat_loop(self):
        """Enviar heartbeat periodico a la API"""
        logger.info("Thread de heartbeat iniciado")

        while self.thread_heartbeat_running:
            try:
                tiempo_actual = time.time()

                if tiempo_actual - self.ultimo_heartbeat >= self.config.intervalo_heartbeat:
                    en_horario = self.esta_en_horario_operacion()

                    if en_horario:
                        mensaje_heartbeat = (
                            f"Sistema operando normalmente. "
                            f"Estado: {self.estado_actual}, "
                            f"Errores stream: {self.camera.errores_consecutivos}, "
                            f"Evento activo: {self.id_evento_activo is not None}"
                        )
                    else:
                        hora_actual_str = datetime.now().strftime("%H:%M")
                        mensaje_heartbeat = (
                            f"Sistema fuera de horario operativo ({hora_actual_str}). "
                            f"Horario: {self.config.hora_inicio:02d}:00 - {self.config.hora_fin:02d}:00. "
                            f"Monitoreo en espera, conexion activa."
                        )

                    self.api.enviar_heartbeat(mensaje_heartbeat)
                    self.ultimo_heartbeat = tiempo_actual

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error en thread de heartbeat: {e}")
                time.sleep(30)

        logger.info("Thread de heartbeat detenido")

    def iniciar_heartbeat(self):
        """Iniciar thread de heartbeat"""
        if self.thread_heartbeat is not None and self.thread_heartbeat.is_alive():
            logger.warning("Thread de heartbeat ya esta corriendo")
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

    def guardar_frame_temporal(self, frame: np.ndarray) -> Optional[str]:
        """Guardar frame en archivo temporal con timestamp y UUID"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]
            filename = self.temp_dir / f"frame_{timestamp}_{unique_id}.jpg"
            cv2.imwrite(str(filename), frame)
            return str(filename)
        except Exception as e:
            logger.error(f"Error al guardar frame: {e}")
            return None

    def procesar_imagen_con_detecciones(
            self,
            frame: np.ndarray,
            detecciones: List[Dict],
            evento_id: int
    ):
        """Procesar y enviar imagen con detecciones"""
        imagen_path = self.guardar_frame_temporal(frame)

        if not imagen_path:
            logger.error(f"CAPTURA FALLIDA | Evento: {evento_id} | Detecciones: {len(detecciones)}")
            return

        file_name = Path(imagen_path).name
        logger.info(f"CAPTURA | Archivo: {file_name} | Evento: {evento_id} | Detecciones: {len(detecciones)}")

        azure_url = self.azure.subir_imagen(imagen_path, file_name)

        if not azure_url:
            logger.error(f"Error al subir imagen a Azure: {file_name}")
            self.api.enviar_log("error", f"Fallo al subir imagen a Azure: {file_name}")
            return

        def eliminar_archivo_temporal():
            try:
                if os.path.exists(imagen_path):
                    os.remove(imagen_path)
                    logger.info(f"Archivo temporal eliminado: {file_name}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal: {e}")

        self.api.enviar_imagen_con_detecciones(
            evento_id=evento_id,
            azure_url=azure_url,
            detecciones=detecciones,
            callback_eliminar_archivo=eliminar_archivo_temporal
        )

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

            if self.contador_sin_deteccion >= self.config.umbral_cerrar_evento:
                if self.id_evento_activo is not None:
                    logger.info(f"Cerrando evento {self.id_evento_activo}")
                    self.api.enviar_log(
                        "info",
                        f"Evento cerrado: {self.id_evento_activo} - Sin detecciones por tiempo prolongado"
                    )
                    self.id_evento_activo = None
                    self.estado_actual = "sin_deteccion"
                self.contador_sin_deteccion = 0

        else:
            self.contador_sin_deteccion = 0
            self.contador_con_deteccion += 1

            if self.contador_con_deteccion >= self.config.umbral_crear_evento:
                if self.id_evento_activo is None:
                    self.id_evento_activo = self.api.crear_evento()
                    if self.id_evento_activo:
                        self.estado_actual = "evento_activo"
                        logger.info("Estado: EVENTO ACTIVO")

            if self.id_evento_activo is not None:
                self.procesar_imagen_con_detecciones(
                    frame,
                    detecciones,
                    self.id_evento_activo
                )

    def obtener_tiempo_espera(self) -> int:
        """Obtener tiempo de espera segun estado actual"""
        if self.estado_actual == "sin_deteccion":
            return self.config.tiempo_foto_sin_deteccion
        else:
            return self.config.tiempo_foto_con_deteccion

    def ejecutar_ciclo(self):
        """Ejecutar ciclo principal de monitoreo"""
        logger.info("Iniciando ciclo de monitoreo...")
        self.api.enviar_log("info", "Sistema de monitoreo termico iniciado correctamente")

        ultimo_capture = time.time()
        ultimo_log_fuera_horario = 0

        while self.running:
            try:
                if not self.esta_en_horario_operacion():
                    tiempo_actual = time.time()
                    if tiempo_actual - ultimo_log_fuera_horario >= 300:
                        hora_actual_str = datetime.now().strftime("%H:%M")
                        logger.info(
                            f"Fuera de horario operativo ({hora_actual_str}). "
                            f"Esperando horario {self.config.hora_inicio:02d}:00 - {self.config.hora_fin:02d}:00"
                        )
                        ultimo_log_fuera_horario = tiempo_actual

                    time.sleep(30)
                    continue

                tiempo_actual = time.time()
                tiempo_espera = self.obtener_tiempo_espera()

                if tiempo_actual - ultimo_capture >= tiempo_espera:
                    frame = self.camera.obtener_frame()

                    if frame is not None:
                        detecciones = self.detector.detectar(frame)

                        logger.info(
                            f"Estado: {self.estado_actual} | "
                            f"Detecciones: {len(detecciones)} | "
                            f"Sin deteccion: {self.contador_sin_deteccion} | "
                            f"Con deteccion: {self.contador_con_deteccion} | "
                            f"Errores stream: {self.camera.errores_consecutivos}"
                        )

                        self.procesar_detecciones(frame, detecciones)

                        ultimo_capture = tiempo_actual

                    else:
                        logger.warning("Frame no disponible, esperando...")
                        time.sleep(1)

                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("Interrupcion por usuario")
                self.api.enviar_log("advertencia", "Sistema detenido por usuario")
                break
            except Exception as e:
                error_msg = f"Error en ciclo principal: {str(e)}"
                logger.error(error_msg)
                self.api.enviar_log("error", error_msg)
                time.sleep(5)

    def iniciar(self):
        """Iniciar sistema de monitoreo"""
        logger.info("Iniciando sistema de monitoreo termico...")
        logger.info(f"Horario de operacion configurado: {self.config.hora_inicio:02d}:00 - {self.config.hora_fin:02d}:00")

        if not self.config.validar():
            logger.error("Configuracion invalida")
            return

        if not self.camera.inicializar():
            logger.error("No se pudo inicializar la camara")
            return

        if not self.detector.cargar_modelo():
            logger.error("No se pudo cargar el modelo")
            return

        if not self.api.autenticar():
            logger.error("No se pudo autenticar con la API")
            return

        self.camera.iniciar_lectura_continua(self.esta_en_horario_operacion)
        self.iniciar_heartbeat()

        time.sleep(2)

        self.running = True
        try:
            self.ejecutar_ciclo()
        finally:
            self.detener()

    def detener(self):
        """Detener sistema y liberar recursos"""
        logger.info("Deteniendo sistema...")
        self.api.enviar_log("advertencia", "Sistema de monitoreo termico detenido")
        self.running = False

        self.detener_heartbeat()
        self.camera.liberar()

        logger.info("Sistema detenido")