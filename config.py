import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuracion del sistema de monitoreo termico"""

    camera_source: str
    api_base_url: str
    username: str
    password: str
    model_path: str

    confidence_threshold: float = 0.5

    azure_container_url: Optional[str] = None
    azure_token_sas: Optional[str] = None

    max_errores_consecutivos: int = 5
    timeout_reconexion: int = 10

    intervalo_heartbeat: int = 300

    hora_inicio: int = 6
    hora_fin: int = 21

    tiempo_foto_sin_deteccion: int = 5
    tiempo_foto_con_deteccion: int = 2
    umbral_crear_evento: int = 3
    umbral_cerrar_evento: int = 5

    temp_dir: str = "temp_images"

    @classmethod
    def from_env(cls, **kwargs) -> 'Config':
        """Crear configuracion desde variables de entorno y parametros"""
        load_dotenv()

        config_dict = {
            'azure_container_url': os.getenv("AZURE_CONTAINER_URL"),
            'azure_token_sas': os.getenv("TOKENSAS"),
        }

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

        if not self.azure_container_url or not self.azure_token_sas:
            logger.warning("Configuracion de Azure incompleta. Las imagenes no se subiran a Azure.")

        return True

    def __post_init__(self):
        """Normalizar valores despues de inicializacion"""
        self.api_base_url = self.api_base_url.rstrip('/')