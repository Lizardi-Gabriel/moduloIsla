import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Parametros tecnicos compartidos. Tiempos en segundos salvo los sufijos _MS.
ENV_FILE = Path(__file__).resolve().parent / ".env"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "thermal_monitor.log"
LOG_MAX_BYTES = 5_000_000
LOG_BACKUP_COUNT = 3
LOG_REPEAT_INTERVAL = 300
LOG_REPEAT_CAPACITY = 512
API_LOG_TIMEOUT = 10
API_IMAGE_TIMEOUT = 30
CAMERA_IO_TIMEOUT_MS = 5000
CAMERA_BUFFER_SIZE = 1
CAMERA_SCHEDULE_POLL_INTERVAL = 10
CAMERA_RETRY_DELAY = 5
CAMERA_READ_FAILURE_DELAY = 0.5
CAMERA_READ_INTERVAL = 0.01
CAMERA_ERROR_DELAY = 1
CAMERA_STOP_TIMEOUT = 12
HEARTBEAT_POLL_INTERVAL = 30
HEARTBEAT_STOP_TIMEOUT = 5
MONITOR_SCHEDULE_POLL_INTERVAL = 30
MONITOR_NO_FRAME_DELAY = 1
MONITOR_POLL_INTERVAL = 0.1
MONITOR_ERROR_DELAY = 5
MONITOR_STARTUP_DELAY = 2


@dataclass
class Config:
    """Configuracion del sistema de monitoreo termico"""

    # Los nombres existentes de .env se conservan por compatibilidad.
    camera_source: str = field(default="", metadata={"env": "CAMARA_TERMICA"})
    api_base_url: str = field(default="", metadata={"env": "API_CONTROL"})
    username: str = field(default="", metadata={"env": "USER_API"})
    password: str = field(default="", repr=False, metadata={"env": "PASSWORD"})
    model_path: str = field(default="", metadata={"env": "MODEL_PATH"})

    confidence_threshold: float = 0.5

    max_errores_consecutivos: int = 5
    timeout_reconexion: int = 10
    max_antiguedad_frame: float = 5

    intervalo_heartbeat: int = 300

    hora_inicio: int = 6
    hora_fin: int = 23

    tiempo_foto_sin_deteccion: int = 5
    tiempo_foto_con_deteccion: int = 2
    umbral_crear_evento: int = 3
    umbral_cerrar_evento: int = 5

    temp_dir: str = "temp_images"

    @classmethod
    def from_env(cls, **kwargs) -> 'Config':
        """Cargar .env y convertir tipos; prioridad: kwargs > entorno > defaults.

        Los parametros sin alias usan su nombre en mayusculas en el entorno
        (por ejemplo HORA_INICIO o CONFIDENCE_THRESHOLD).
        """
        load_dotenv(ENV_FILE)
        config_dict = {}
        for item in fields(cls):
            if item.name in kwargs:
                continue
            env_name = item.metadata.get("env", item.name.upper())
            value = os.getenv(env_name)
            if value is not None:
                try:
                    config_dict[item.name] = item.type(value)
                except (TypeError, ValueError):
                    raise ValueError(f"{env_name} debe ser de tipo {item.type.__name__}") from None
        config_dict.update(kwargs)
        return cls(**config_dict)

    def validar(self) -> bool:
        """Validar configuracion basica"""
        if not self.camera_source:
            logger.error("camera_source no configurado")
            return False

        if not self.api_base_url:
            logger.error("api_base_url no configurado")
            return False

        if not self.model_path:
            logger.error("model_path no configurado")
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"Modelo no encontrado: {self.model_path}")
            return False

        if self.hora_inicio < 0 or self.hora_inicio > 23:
            logger.error("hora_inicio debe estar entre 0 y 23")
            return False

        if self.hora_fin < 0 or self.hora_fin > 23:
            logger.error("hora_fin debe estar entre 0 y 23")
            return False

        if self.max_antiguedad_frame <= 0:
            logger.error("max_antiguedad_frame debe ser mayor que cero")
            return False

        return True

    def __post_init__(self):
        """Normalizar valores despues de inicializacion"""
        self.api_base_url = (self.api_base_url or '').rstrip('/')
