import requests
import threading
import logging
from typing import Optional, Dict, List
from datetime import date

logger = logging.getLogger(__name__)


class APIClient:
    """Cliente para interactuar con la API del sistema"""

    def __init__(
            self,
            api_base_url: str,
            username: str,
            password: str
    ):
        self.api_base_url = api_base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token: Optional[str] = None

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
                logger.info("Autenticacion exitosa")
                return True
            else:
                logger.error(f"Error de autenticacion con API: {response.status_code}, {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error al autenticar con API: {str(e)}")
            return False

    def _obtener_headers(self) -> Dict[str, str]:
        """Obtener headers con token de autenticacion"""
        return {
            "Authorization": f"Bearer {self.token}"
        }

    def enviar_log(self, tipo: str, mensaje: str):
        """Enviar log al endpoint de la API de forma asincrona"""
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
                    headers=self._obtener_headers(),
                    timeout=10
                )

                if response.status_code != 201:
                    logger.warning(f"Error al enviar log a API: {response.status_code}")

            except Exception as e:
                logger.warning(f"Excepcion al enviar log a API: {e}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def crear_evento(self, descripcion: str = "Evento detectado automaticamente") -> Optional[int]:
        """Crear un nuevo evento en la API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/eventos",
                json={
                    "fecha_evento": date.today().isoformat(),
                    "descripcion": descripcion,
                    "estatus": "pendiente"
                },
                headers=self._obtener_headers()
            )

            if response.status_code == 201:
                evento_id = response.json()["evento_id"]
                logger.info(f"Evento creado con ID: {evento_id}")
                self.enviar_log("info", f"Nuevo evento creado con ID: {evento_id}")
                return evento_id
            else:
                logger.error(f"Error al crear evento en API: {response.status_code}")
                self.enviar_log("error", f"Error al crear evento en API: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Excepcion al crear evento: {str(e)}")
            self.enviar_log("error", f"Excepcion al crear evento: {str(e)}")
            return None

    def enviar_imagen_con_detecciones(
            self,
            evento_id: int,
            azure_url: str,
            detecciones: List[Dict],
            callback_eliminar_archivo
    ):
        """Enviar imagen y detecciones a la API de forma asincrona"""
        def _enviar():
            try:
                payload = {
                    "imagen": {
                        "ruta_imagen": azure_url
                    },
                    "detecciones": detecciones
                }

                response = requests.post(
                    f"{self.api_base_url}/eventos/{evento_id}/imagenes",
                    json=payload,
                    headers=self._obtener_headers()
                )

                if response.status_code == 201:
                    logger.info(f"Imagen enviada exitosamente a API - Evento: {evento_id}")
                    callback_eliminar_archivo()
                else:
                    logger.error(f"Error al enviar imagen a API: {response.status_code}")
                    self.enviar_log("error", f"Error al enviar imagen a API: {response.status_code}")

            except Exception as e:
                logger.error(f"Excepcion al enviar imagen a API: {str(e)}")
                self.enviar_log("error", f"Excepcion al enviar imagen a API: {str(e)}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def enviar_heartbeat(self, mensaje: str):
        """Enviar heartbeat a la API de forma asincrona"""
        def _enviar():
            try:
                if not self.token:
                    return

                self.enviar_log("info", mensaje)
                logger.info("Heartbeat enviado a API")

            except Exception as e:
                logger.error(f"Error al enviar heartbeat: {e}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()