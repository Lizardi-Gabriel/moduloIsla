import logging
from typing import List, Dict, Optional
from ultralytics import YOLO
import numpy as np
from config import Config
logger = logging.getLogger(__name__)


class DetectionService:
    """Servicio de deteccion de objetos usando YOLO"""

    def __init__(
            self,
            model_path: str,
            confidence_threshold: float = Config.confidence_threshold
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None

    def cargar_modelo(self) -> bool:
        """Cargar modelo YOLO"""
        try:
            self.model = YOLO(self.model_path)
            logger.debug("Modelo YOLO cargado")
            return True
        except Exception as e:
            logger.error(f"Error al cargar modelo YOLO: {type(e).__name__}")
            return False

    def detectar(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """Devolver detecciones (incluida lista vacia) o None si el analisis falla."""
        if self.model is None:
            logger.error("Modelo no cargado")
            return None

        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
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
            logger.error(f"Error en deteccion: {type(e).__name__}")
            return None
