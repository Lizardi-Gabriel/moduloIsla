import config as settings
from log_policy import LogLimiter
import requests
import threading
import logging
import base64
import cv2
from pathlib import Path
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
        self._log_limiter = LogLimiter()
        self._auth_lock = threading.Lock()
        self._auth_generation = 0
        self._auth_success = False

    def autenticar(self) -> bool:
        """Autenticar sin permitir renovaciones simultaneas del token."""
        with self._auth_lock:
            return self._autenticar()

    def _autenticar(self) -> bool:
        """Se invoca con _auth_lock adquirido; no usa solicitudes autenticadas."""
        self._auth_generation += 1
        self._auth_success = False
        try:
            response = requests.post(
                f"{self.api_base_url}/token",
                data={
                    "username": self.username,
                    "password": self.password
                },
                timeout=(settings.API_CONNECT_TIMEOUT, settings.API_AUTH_TIMEOUT)
            )

            if response.status_code == 200:
                token = response.json()["access_token"]
                if not isinstance(token, str) or not token.strip():
                    raise ValueError("Token invalido")
                self.token = token
                self._auth_success = True
                logger.debug("Autenticacion exitosa")
                return True
            else:
                logger.error(f"Error de autenticacion con API: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error al autenticar con API: {type(e).__name__}")
            return False

    def _obtener_headers(self) -> Dict[str, str]:
        """Obtener headers con token de autenticacion"""
        return {
            "Authorization": f"Bearer {self.token}"
        }

    def _post(self, url: str, *, timeout=settings.API_EVENT_TIMEOUT, **kwargs):
        """Repetir una sola vez tras 401; nunca repetir por timeout o error 5xx.

        Rebobinar archivos evita subir cuerpos vacios al renovar la sesion.
        La generacion permite compartir una renovacion entre solicitudes simultaneas,
        incluso si el servidor devuelve el mismo token o la renovacion falla.
        """
        positions = [(item[1], item[1].tell()) for item in kwargs.get("files", {}).values()]
        with self._auth_lock:
            generation = self._auth_generation
            headers = self._obtener_headers()
        response = requests.post(
            url, headers=headers,
            timeout=(settings.API_CONNECT_TIMEOUT, timeout), **kwargs
        )
        if response.status_code != 401:
            return response

        with self._auth_lock:
            if generation == self._auth_generation:
                if not self._autenticar():
                    return response
            elif not self._auth_success:
                return response
            retry_headers = self._obtener_headers()
        for file, position in positions:
            file.seek(position)
        response.close()
        return requests.post(
            url, headers=retry_headers,
            timeout=(settings.API_CONNECT_TIMEOUT, timeout), **kwargs
        )

    def enviar_log(self, tipo: str, mensaje: str, *, repetible: bool = False):
        """Enviar log al endpoint de la API de forma asincrona"""
        if not self.token:
            return
        if not repetible and not self._log_limiter.allow((tipo, mensaje)):
            return

        def _enviar():
            try:
                if not self.token:
                    return

                response = self._post(
                    f"{self.api_base_url}/logs",
                    json={
                        "tipo": tipo,
                        "mensaje": mensaje
                    },
                    timeout=settings.API_LOG_TIMEOUT
                )

                if response.status_code != 201:
                    logger.warning(f"Error al enviar log a API: {response.status_code}")

            except Exception as e:
                logger.warning(f"Excepcion al enviar log a API: {type(e).__name__}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def crear_evento(self, descripcion: str = "Evento detectado automaticamente") -> Optional[int]:
        """Crear un nuevo evento en la API"""
        try:
            response = self._post(
                f"{self.api_base_url}/eventos",
                json={
                    "fecha_evento": date.today().isoformat(),
                    "descripcion": descripcion,
                    "estatus": "pendiente"
                },
                timeout=settings.API_EVENT_TIMEOUT
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
            logger.error(f"Excepcion al crear evento: {type(e).__name__}")
            self.enviar_log("error", f"Excepcion al crear evento: {type(e).__name__}")
            return None

    def subir_imagen_evento(self, evento_id: int, imagen_path: str, file_name: Optional[str] = None) -> bool:
        """Subir imagen del evento al endpoint multipart/form-data del backend."""
        try:
            filename = file_name or Path(imagen_path).name
            mime_type = "image/jpeg"
            if filename.lower().endswith((".png", ".webp")):
                mime_type = "image/png" if filename.lower().endswith(".png") else "image/webp"

            with open(imagen_path, "rb") as file:
                files = {
                    "file": (filename, file, mime_type)
                }

                response = self._post(
                    f"{self.api_base_url}/eventos/{evento_id}/imagenes/upload",
                    files=files,
                    timeout=settings.API_IMAGE_TIMEOUT
                )

            if response.status_code in (200, 201):
                logger.debug(f"Imagen subida al endpoint de evento {evento_id}")
                return True

            logger.error(f"Error al subir imagen del evento {evento_id}: {response.status_code}")
            self.enviar_log("error", f"Error al subir imagen del evento {evento_id}: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Excepcion al subir imagen del evento {evento_id}: {type(e).__name__}")
            self.enviar_log("error", f"Excepcion al subir imagen del evento {evento_id}: {type(e).__name__}")
            return False

    def enviar_imagen_con_detecciones(
            self,
            evento_id: int,
            imagen_path: str,
            detecciones: List[Dict],
            callback_eliminar_archivo=None
    ):
        """Primero subir la imagen y luego enviar su ruta con detecciones al backend."""
        def _enviar():
            try:
                filename = Path(imagen_path).name
                mime_type = "image/jpeg"
                if filename.lower().endswith((".png", ".webp")):
                    mime_type = "image/png" if filename.lower().endswith(".png") else "image/webp"

                with open(imagen_path, "rb") as file:
                    files = {"file": (filename, file, mime_type)}
                    upload_response = self._post(
                        f"{self.api_base_url}/eventos/{evento_id}/imagenes/upload",
                        files=files,
                        timeout=settings.API_IMAGE_TIMEOUT
                    )

                if upload_response.status_code not in (200, 201):
                    logger.error(
                        f"Error al subir imagen al evento {evento_id}: "
                        f"{upload_response.status_code}"
                    )
                    self.enviar_log(
                        "error",
                        f"Error al subir imagen al evento {evento_id}: {upload_response.status_code}"
                    )
                    return

                logger.debug(f"Imagen subida correctamente al evento {evento_id}")


                upload_data = upload_response.json()

                ruta_imagen_servidor = upload_data["url"]

                payload = {
                    "imagen": {
                        "ruta_imagen": ruta_imagen_servidor
                    },
                    "detecciones": detecciones
                }

                response = self._post(
                    f"{self.api_base_url}/eventos/{evento_id}/imagenes",
                    json=payload,
                    timeout=settings.API_IMAGE_TIMEOUT
                )

                if response.status_code in (200, 201):
                    logger.debug(f"Imagen y detecciones enviadas exitosamente - Evento: {evento_id}")
                    if callback_eliminar_archivo:
                        callback_eliminar_archivo()
                else:
                    logger.error(
                        f"Error al registrar imagen y detecciones del evento {evento_id}: {response.status_code}"
                    )
                    self.enviar_log(
                        "error",
                        f"Error al registrar imagen y detecciones del evento {evento_id}: {response.status_code}"
                    )

            except Exception as e:
                logger.error(f"Excepcion al enviar imagen y detecciones del evento {evento_id}: {type(e).__name__}")
                self.enviar_log("error", f"Excepcion al enviar imagen y detecciones del evento {evento_id}: {type(e).__name__}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()

    def enviar_heartbeat(self, mensaje: str):
        """Mantener la señal de vida sin anunciar un envio aun no confirmado."""
        self.enviar_log("info", mensaje, repetible=True)

    def enviar_imagen_contexto_b64(self, evento_id: int, frame_contexto) -> bool:
        """Codificar imagen a Base64 y enviarla para descripcion"""
        def _enviar():
            try:
                # Codificar frame a JPG
                _, buffer = cv2.imencode('.jpg', frame_contexto)
                # Convertir a Base64 string
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                payload = {
                    "imagen_base64": jpg_as_text,
                }

                # endpoint
                url = f"{self.api_base_url}/eventos/{evento_id}/descripcion"

                logger.debug(f"Enviando imagen de contexto (Base64) para evento {evento_id}...")

                response = self._post(
                    url,
                    json=payload,
                    timeout=settings.API_IMAGE_TIMEOUT
                )

                if response.status_code in [200, 201]:
                    logger.debug(f"Imagen de contexto enviada con exito. Evento: {evento_id}")
                else:
                    logger.error(f"Error al enviar imagen contexto: {response.status_code}")

            except Exception as e:
                logger.error(f"Excepcion enviando imagen contexto: {type(e).__name__}")

        thread = threading.Thread(target=_enviar)
        thread.daemon = True
        thread.start()
        return True