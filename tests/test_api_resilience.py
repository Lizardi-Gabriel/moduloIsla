import io
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import requests
import config as settings
from api_client import APIClient


def response(status, data=None):
    return Mock(status_code=status, json=Mock(return_value=data))


class APIResilienceTests(unittest.TestCase):
    def setUp(self):
        self.api = APIClient('https://example.invalid', 'user', 'secret')
        self.api.token = 'old'

    def test_auth_and_event_timeouts(self):
        with patch('api_client.requests.post', return_value=response(200, {'access_token': 'new'})) as post:
            self.assertTrue(self.api.autenticar())
            self.assertEqual(post.call_args.kwargs['timeout'], (settings.API_CONNECT_TIMEOUT, settings.API_AUTH_TIMEOUT))
        with patch('api_client.requests.post', return_value=response(201, {'evento_id': 42})) as post, patch.object(self.api, 'enviar_log'):
            self.assertEqual(self.api.crear_evento(), 42)
            self.assertEqual(post.call_args.kwargs['timeout'], (settings.API_CONNECT_TIMEOUT, settings.API_EVENT_TIMEOUT))

    def test_401_renews_and_retries_once(self):
        with patch('api_client.requests.post', side_effect=[response(401), response(200, {'access_token': 'new'}), response(201, {'evento_id': 42})]) as post, patch.object(self.api, 'enviar_log'):
            self.assertEqual(self.api.crear_evento(), 42)
            self.assertEqual(post.call_count, 3)
            self.assertEqual(post.call_args_list[0].kwargs['headers']['Authorization'], 'Bearer old')
            self.assertEqual(post.call_args_list[2].kwargs['headers']['Authorization'], 'Bearer new')
            self.assertEqual(post.call_args_list[0].kwargs['json'], post.call_args_list[2].kwargs['json'])

    def test_second_401_is_not_retried(self):
        with patch('api_client.requests.post', side_effect=[response(401), response(200, {'access_token': 'new'}), response(401)]) as post:
            self.assertEqual(self.api._post('/eventos').status_code, 401)
            self.assertEqual(post.call_count, 3)

    def test_failed_auth_does_not_repeat_operation(self):
        for auth_result in [response(401), response(200, {}), response(200, {'access_token': ''}), requests.Timeout()]:
            with self.subTest(auth_result=auth_result), patch('api_client.requests.post', side_effect=[response(401), auth_result]) as post:
                self.assertEqual(self.api._post('/eventos').status_code, 401)
                self.assertEqual(post.call_count, 2)

    def test_timeout_and_server_error_do_not_retry(self):
        with patch('api_client.requests.post', side_effect=requests.Timeout()) as post:
            with self.assertRaises(requests.Timeout):
                self.api._post('/eventos')
            post.assert_called_once()
        for status in [403, 500, 503]:
            with patch('api_client.requests.post', return_value=response(status)) as post:
                self.assertEqual(self.api._post('/eventos').status_code, status)
                post.assert_called_once()

    def test_upload_retry_rewinds_file(self):
        contents = []
        def post(url, **kwargs):
            if url.endswith('/token'):
                return response(200, {'access_token': 'new'})
            contents.append(kwargs['files']['file'][1].read())
            return response(401 if len(contents) == 1 else 201)
        with patch('api_client.requests.post', side_effect=post):
            self.assertEqual(self.api._post('/upload', files={'file': ('image.jpg', io.BytesIO(b'image'), 'image/jpeg')}).status_code, 201)
        self.assertEqual(contents, [b'image', b'image'])

    def test_concurrent_401s_share_refresh_even_with_same_token(self):
        for success in [True, False]:
            with self.subTest(success=success):
                api = APIClient('https://example.invalid', 'user', 'secret')
                api.token = 'old'
                barrier = threading.Barrier(2)
                local = threading.local()
                auth_calls = []
                def post(url, **kwargs):
                    if url.endswith('/token'):
                        auth_calls.append(url)
                        return response(200, {'access_token': 'old'}) if success else response(401)
                    if not getattr(local, 'called', False):
                        local.called = True
                        barrier.wait(timeout=3)
                        return response(401)
                    return response(201)
                with patch('api_client.requests.post', side_effect=post), ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: api._post('/eventos').status_code, range(2)))
                self.assertEqual(results, [201, 201] if success else [401, 401])
                self.assertEqual(len(auth_calls), 1)


if __name__ == '__main__':
    unittest.main()
