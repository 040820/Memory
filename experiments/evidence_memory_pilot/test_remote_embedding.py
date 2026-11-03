import asyncio
import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PATH = Path(__file__).with_name('remote_embedding.py')
if PATH.exists():
    spec = importlib.util.spec_from_file_location('remote_embedding', PATH)
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
else:
    remote = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body['input'] == ['bad']:
            payload = {'data': [{'index': 0, 'embedding': [float('nan'), 1]}]}
        else:
            payload = {'data': [{'index': i, 'embedding': [float(i), 1.0]} for i in reversed(range(len(body['input'])))]}
        b = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *args):
        pass


class RemoteEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self):
        self.assertIsNotNone(remote, 'remote embedding client is missing')
        return remote.RemoteEmbeddingClient(f'http://127.0.0.1:{self.server.server_port}/v1', 'bge-m3', 2)

    def test_batch_restores_input_order(self):
        result = asyncio.run(self.client().embed(['first', 'second']))
        self.assertEqual(result, [[0.0, 1.0], [1.0, 1.0]])

    def test_rejects_nonfinite_vectors(self):
        with self.assertRaises(ValueError):
            asyncio.run(self.client().embed(['bad']))

    def test_empty_input_does_not_send_request(self):
        self.assertEqual(asyncio.run(self.client().embed([])), [])

    def test_adapter_preserves_manager_methods_and_tensor_contract(self):
        class FakeTorch:
            float32 = 'float32'
            class cuda:
                @staticmethod
                def is_available(): return False
            @staticmethod
            def device(name): return name
            @staticmethod
            def tensor(values, dtype, device): return (values, dtype, device)
        class Manager:
            def cosine_similarity(self, a, b): return 42
        remote.install_manager_adapter(Manager, self.client(), FakeTorch)
        instance = Manager(object())
        self.assertEqual(instance.cosine_similarity([], []), 42)
        self.assertEqual(asyncio.run(instance.get_embeddings(['first'])), [[0.0, 1.0]])
        tensor = asyncio.run(instance.get_embeddings(['first'], need_tensor=True))
        self.assertEqual(tensor, ([[0.0, 1.0]], 'float32', 'cpu'))


if __name__ == '__main__':
    unittest.main()
