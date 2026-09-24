import logging
from logging.handlers import RotatingFileHandler
from log_policy import RepeatedWarningFilter
import signal
import sys

import config as settings
from config import Config
from thermal_monitor import ThermalMonitor

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        RotatingFileHandler(settings.LOG_FILE, maxBytes=settings.LOG_MAX_BYTES,
                            backupCount=settings.LOG_BACKUP_COUNT, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Cada salida mantiene su propio limite para mostrar el primer fallo en ambas.
for handler in logging.getLogger().handlers:
    handler.addFilter(RepeatedWarningFilter())
logging.getLogger("urllib3").setLevel(logging.WARNING)
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

    config = Config.from_env()

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
