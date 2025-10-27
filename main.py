import logging
import signal
import sys
import os

from config import Config
from thermal_monitor import ThermalMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thermal_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

monitor = None


def signal_handler(sig, frame):
    """Manejar señales de interrupcion"""
    logger.info("Señal de interrupcion recibida")
    if monitor:
        monitor.detener()
    sys.exit(0)


def main():
    """Ejecutar el sistema de monitoreo termico"""
    global monitor

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    camera_source = os.getenv("CAMARA_TERMICA")
    api_base_url = os.getenv("API_CONTROL")
    username = os.getenv("USER_API")
    password = os.getenv("PASSWORD")
    model_path = os.getenv("MODEL_PATH")

    config = Config.from_env(
        camera_source=camera_source,
        api_base_url=api_base_url,
        username=username,
        password=password,
        model_path=model_path,
        confidence_threshold=0.5,
        max_errores_consecutivos=5,
        timeout_reconexion=10,
        intervalo_heartbeat=300,
        hora_inicio=6,
        hora_fin=23 # 21
    )

    monitor = ThermalMonitor(config)

    try:
        monitor.iniciar()
    except KeyboardInterrupt:
        logger.info("Programa terminado por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
    finally:
        if monitor:
            monitor.detener()


if __name__ == "__main__":
    main()