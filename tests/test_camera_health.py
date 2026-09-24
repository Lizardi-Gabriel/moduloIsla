import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from camera_manager import CameraManager

# Estas pruebas no cargan YOLO ni pesos: verifican conexion y coordinacion.
with patch.dict(sys.modules, {'detection_service': SimpleNamespace(DetectionService=Mock())}):
    from thermal_monitor import ThermalMonitor


class CameraHealthTests(unittest.TestCase):
    def test_expired_frame_is_never_returned(self):
        camera = CameraManager('unused', max_antiguedad_frame=5)
        with patch('camera_manager.time.monotonic', return_value=10):
            camera._registrar_frame(np.zeros((2, 2, 3)))
        with patch('camera_manager.time.monotonic', return_value=14):
            self.assertIsNotNone(camera.obtener_frame())
        with patch('camera_manager.time.monotonic', return_value=16):
            self.assertIsNone(camera.obtener_frame())
            self.assertEqual(camera.obtener_estado(), {'disponible': False, 'antiguedad_segundos': 6})

    def test_disconnect_invalidates_frame_but_preserves_age(self):
        camera = CameraManager('unused')
        with patch('camera_manager.time.monotonic', return_value=10):
            camera._registrar_frame(np.zeros((2, 2, 3)))
            camera._reportar_desconexion()
            self.assertIsNone(camera.obtener_frame())
            self.assertEqual(camera.obtener_estado()['antiguedad_segundos'], 0)

    def test_reader_recovers_after_initial_open_failure(self):
        camera = CameraManager('unused', timeout_reconexion=0)
        failed, healthy = Mock(), Mock()
        failed.isOpened.return_value = False
        healthy.isOpened.return_value = True
        healthy.read.return_value = (True, np.zeros((2, 2, 3)))
        error, recovered = Mock(), Mock()
        camera.set_callbacks(error, recovered)
        camera.thread_lectura_running = True
        def schedule():
            if healthy.read.called:
                camera.thread_lectura_running = False
                return False
            return True
        with patch('camera_manager.cv2.VideoCapture', side_effect=[failed, healthy]), patch.object(camera._stop, 'wait', return_value=False):
            camera._leer_frames_continuamente(schedule)
        failed.release.assert_called_once()
        recovered.assert_called_once()
        error.assert_called_once()
        self.assertIsNotNone(camera.obtener_frame())

    def test_initial_frame_is_published_and_empty_frame_rejected(self):
        for frame, expected in [(np.zeros((2, 2, 3)), True), (np.empty((0, 0, 3)), False)]:
            camera = CameraManager('unused')
            with patch('camera_manager.cv2.VideoCapture') as capture:
                capture.return_value.isOpened.return_value = True
                capture.return_value.read.return_value = (True, frame)
                self.assertEqual(camera.inicializar(), expected)
                self.assertEqual(camera.obtener_estado()['disponible'], expected)

    def test_start_monitor_does_not_require_synchronous_camera_success(self):
        monitor = ThermalMonitor.__new__(ThermalMonitor)
        monitor.config = Mock(hora_inicio=6, hora_fin=23)
        monitor.camera = Mock()
        monitor.camera.inicializar.return_value = False
        monitor.detector = Mock()
        monitor.api = Mock()
        monitor.iniciar_heartbeat = Mock()
        monitor.ejecutar_ciclo = Mock()
        monitor.detener = Mock()
        with patch('thermal_monitor.time.sleep'):
            monitor.iniciar()
        monitor.camera.inicializar.assert_not_called()
        monitor.camera.iniciar_lectura_continua.assert_called_once()
        monitor.iniciar_heartbeat.assert_called_once()
        monitor.ejecutar_ciclo.assert_called_once()

    def test_heartbeat_health_and_outside_schedule(self):
        monitor = ThermalMonitor.__new__(ThermalMonitor)
        monitor.camera = CameraManager('unused')
        monitor.id_evento_activo = None
        monitor.esta_en_horario_operacion = Mock(return_value=True)
        self.assertIn('sin cuadros recibidos', monitor.obtener_mensaje_heartbeat())
        with patch('camera_manager.time.monotonic', return_value=10):
            monitor.camera._registrar_frame(np.zeros((2, 2, 3)))
            self.assertIn('camara recibiendo imagenes', monitor.obtener_mensaje_heartbeat())
        with patch('camera_manager.time.monotonic', return_value=20):
            self.assertIn('sin imagen reciente utilizable', monitor.obtener_mensaje_heartbeat())
            self.assertIn('10.0s', monitor.obtener_mensaje_heartbeat())
        monitor.esta_en_horario_operacion.return_value = False
        self.assertIn('camara no evaluada', monitor.obtener_mensaje_heartbeat())


if __name__ == '__main__':
    unittest.main()
