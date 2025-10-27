import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AzureStorage:
    """Gestionar subida de imagenes a Azure Blob Storage"""

    def __init__(
            self,
            container_url: Optional[str] = None,
            token_sas: Optional[str] = None
    ):
        self.container_url = container_url
        self.token_sas = token_sas
        self.habilitado = bool(container_url and token_sas)

        if not self.habilitado:
            logger.warning("Azure Storage no configurado. Las imagenes no se subiran.")
        else:
            logger.info("Azure Storage configurado correctamente")

    def subir_imagen(self, local_path: str, file_name: str) -> Optional[str]:
        """Subir imagen a Azure Blob Storage"""
        if not self.habilitado:
            logger.warning("Azure Storage no habilitado")
            return None

        blob_url = f"{self.container_url}/{file_name}?{self.token_sas}"

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
                azure_url = f"{self.container_url}/{file_name}"
                logger.info(f"Imagen subida a Azure: {file_name}")
                return azure_url
            else:
                logger.error(f"Error al subir a Azure: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Excepcion al subir a Azure: {e}")
            return None