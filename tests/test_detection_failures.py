import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

# Probar el servicio real sin importar el motor ni cargar pesos.
with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock())}):
    from detection_service import DetectionService
from thermal_monitor import ThermalMonitor


class DetectionFailureTests(unittest.TestCase):
    def test_missing_model_is_failure(self):
        service = DetectionService('unused')
        with self.assertLogs('detection_service', level='ERROR'):
            self.assertIsNone(service.detectar(np.zeros((2, 2, 3))))

    def test_inference_and_result_errors_are_failures(self):
        for model in [Mock(side_effect=RuntimeError('failure')), Mock(return_value=[object()])]:
            service = DetectionService('unused')
            service.model = model
            with self.assertLogs('detection_service', level='ERROR'):
                self.assertIsNone(service.detectar(np.zeros((2, 2, 3))))

    def test_successful_analysis_without_objects_is_empty_list(self):
        service = DetectionService('unused')
        service.model = Mock(return_value=[SimpleNamespace(boxes=[])])
        self.assertEqual(service.detectar(np.zeros((2, 2, 3))), [])

    def test_failures_do_not_close_capture_but_valid_empty_sample_does(self):
        monitor = ThermalMonitor.__new__(ThermalMonitor)
        monitor.api = Mock()
        monitor.config = SimpleNamespace(umbral_cerrar_evento=5, umbral_crear_evento=3)
        monitor.id_evento_activo = 42
        monitor.estado_actual = 'evento_activo'
        monitor.contador_sin_deteccion = 4
        monitor.contador_con_deteccion = 0
        for _ in range(10):
            monitor.procesar_detecciones(None, None)
        self.assertEqual(monitor.contador_sin_deteccion, 4)
        self.assertEqual(monitor.id_evento_activo, 42)
        self.assertEqual(monitor.estado_actual, 'evento_activo')
        monitor.api.enviar_log.assert_called_with('error', 'Fallo del detector: muestra omitida; contadores de captura sin cambios')
        monitor.procesar_detecciones(None, [])
        self.assertIsNone(monitor.id_evento_activo)

    def test_failure_does_not_advance_opening_counter(self):
        monitor = ThermalMonitor.__new__(ThermalMonitor)
        monitor.api = Mock()
        monitor.contador_con_deteccion = 2
        monitor.contador_sin_deteccion = 0
        monitor.id_evento_activo = None
        monitor.procesar_detecciones(None, None)
        self.assertEqual(monitor.contador_con_deteccion, 2)
        monitor.api.crear_evento.assert_not_called()

    def test_main_loop_skips_failed_sample_and_keeps_capture(self):
        monitor = ThermalMonitor.__new__(ThermalMonitor)
        monitor.api = Mock()
        monitor.camera = Mock()
        monitor.detector = Mock()
        monitor.detector.detectar.return_value = None
        monitor.running = True
        monitor.esta_en_horario_operacion = Mock(return_value=True)
        monitor.obtener_tiempo_espera = Mock(return_value=2)
        monitor.id_evento_activo = 42
        monitor.contador_sin_deteccion = 4
        monitor.contador_con_deteccion = 0
        def stop(_):
            monitor.running = False
        with patch('thermal_monitor.time.time', side_effect=[0, 2]), patch('thermal_monitor.time.sleep', side_effect=stop):
            monitor.ejecutar_ciclo()
        self.assertEqual(monitor.contador_sin_deteccion, 4)
        self.assertEqual(monitor.id_evento_activo, 42)
        monitor.api.enviar_log.assert_called_with('error', 'Fallo del detector: muestra omitida; contadores de captura sin cambios')


if __name__ == '__main__':
    unittest.main()
