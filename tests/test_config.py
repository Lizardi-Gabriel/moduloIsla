import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import Config


class ConfigTests(unittest.TestCase):
    def test_dotenv_is_loaded_before_reading_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / '.env'
            env_file.write_text(
                'CAMARA_TERMICA=rtsp://example.invalid/stream\n'
                'API_CONTROL=https://example.invalid/\n'
                'USER_API=test-user\nPASSWORD=test-password\n'
                'MODEL_PATH=modelos/test.pt\n'
                'HORA_INICIO=8\nCONFIDENCE_THRESHOLD=0.75\n'
            )
            with patch.dict(os.environ, {}, clear=True), patch('config.ENV_FILE', env_file):
                config = Config.from_env()
        self.assertEqual(config.camera_source, 'rtsp://example.invalid/stream')
        self.assertEqual(config.api_base_url, 'https://example.invalid')
        self.assertEqual(config.username, 'test-user')
        self.assertEqual(config.password, 'test-password')
        self.assertEqual(config.model_path, 'modelos/test.pt')
        self.assertEqual(config.hora_inicio, 8)
        self.assertEqual(config.confidence_threshold, 0.75)
        self.assertEqual(config.intervalo_heartbeat, Config.intervalo_heartbeat)

    def test_explicit_values_override_environment_and_environment_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / '.env'
            env_file.write_text('HORA_INICIO=8\nHORA_FIN=20\n')
            with patch.dict(os.environ, {'HORA_INICIO': '9', 'HORA_FIN': 'invalid'}, clear=True), patch('config.ENV_FILE', env_file):
                config = Config.from_env(hora_fin=22)
        self.assertEqual(config.hora_inicio, 9)
        self.assertEqual(config.hora_fin, 22)

    def test_invalid_numeric_setting_names_the_variable(self):
        with patch.dict(os.environ, {'HORA_INICIO': 'invalid'}, clear=True), patch('config.load_dotenv'):
            with self.assertRaisesRegex(ValueError, 'HORA_INICIO debe ser de tipo int'):
                Config.from_env()

    def test_missing_environment_reaches_validation_without_crashing(self):
        with patch.dict(os.environ, {}, clear=True), patch('config.load_dotenv'):
            config = Config.from_env()
        with self.assertLogs('config', level='ERROR'):
            self.assertFalse(config.validar())


if __name__ == '__main__':
    unittest.main()
