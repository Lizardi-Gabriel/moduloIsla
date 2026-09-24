import logging
import unittest
from unittest.mock import Mock, patch

from log_policy import LogLimiter, RepeatedWarningFilter
from api_client import APIClient
from camera_manager import CameraManager


class LoggingTests(unittest.TestCase):
    def test_first_failure_repeat_and_interval(self):
        limiter = LogLimiter()
        with patch('log_policy.time.monotonic', side_effect=[0, 1, 300]):
            self.assertTrue(limiter.allow('fallo'))
            self.assertFalse(limiter.allow('fallo'))
            self.assertTrue(limiter.allow('fallo'))

    def test_console_and_file_both_receive_first_failure(self):
        record = logging.LogRecord('camera', logging.ERROR, '', 0, 'fallo', (), None)
        filters = [RepeatedWarningFilter(), RepeatedWarningFilter()]
        for current in filters:
            self.assertTrue(current.filter(record))
            self.assertFalse(current.filter(record))
        record.state_transition = True
        for current in filters:
            self.assertTrue(current.filter(record))

    def test_remote_duplicates_and_heartbeat(self):
        api = APIClient('https://example.invalid', 'user', 'secret')
        api.token = 'token'
        with patch('api_client.threading.Thread') as thread:
            api.enviar_log('error', 'fallo')
            api.enviar_log('error', 'fallo')
            self.assertEqual(thread.call_count, 1)
            api.enviar_heartbeat('Proceso activo')
            api.enviar_heartbeat('Proceso activo')
            self.assertEqual(thread.call_count, 3)

    def test_camera_reports_one_outage_and_one_recovery(self):
        camera = CameraManager('rtsp://user:secret@example.invalid/stream')
        error, recovered = Mock(), Mock()
        camera.set_callbacks(error, recovered)
        with self.assertLogs('camera_manager', level='INFO') as logs:
            camera._reportar_desconexion()
            camera._reportar_desconexion()
            frame = Mock(shape=(480, 640, 3))
            with patch('camera_manager.cv2.VideoCapture') as capture:
                capture.return_value.isOpened.return_value = True
                capture.return_value.read.return_value = (True, frame)
                self.assertTrue(camera.inicializar())
        error.assert_called_once()
        recovered.assert_called_once()
        self.assertNotIn('secret', str(logs.output))
        self.assertFalse(camera._conexion_fallida)


if __name__ == '__main__':
    unittest.main()
