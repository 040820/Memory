"""OpenAI-compatible dense embedding client for the isolated baseline adapter."""
import asyncio
import json
import math
import urllib.request


class RemoteEmbeddingClient:
    def __init__(self, base_url, model, dimensions, timeout=60, api_key=None):
        self.url = base_url.rstrip('/') + '/embeddings'
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.api_key = api_key
        # Experiment services are on the explicitly configured local network.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _embed(self, texts):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        payload = json.dumps({'model': self.model, 'input': texts}).encode('utf8')
        request = urllib.request.Request(self.url, data=payload, headers=headers, method='POST')
        with self.opener.open(request, timeout=self.timeout) as response:
            data = json.load(response).get('data', [])
        if len(data) != len(texts):
            raise ValueError('embedding response count does not match input')
        indexes = [item.get('index') for item in data]
        if sorted(indexes) != list(range(len(texts))):
            raise ValueError('embedding response indexes do not match input')
        result = []
        for item in sorted(data, key=lambda item: item['index']):
            vector = item.get('embedding')
            if not isinstance(vector, list) or len(vector) != self.dimensions or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise ValueError('embedding vector has incorrect dimensions or nonfinite values')
            result.append(vector)
        return result

    async def embed(self, texts):
        if not texts:
            return []
        return await asyncio.to_thread(self._embed, texts)


def install_manager_adapter(manager_class, client, torch_module):
    """Patch only this process; existing cosine methods remain on the class."""
    def initialize(self, config=None):
        self.device = torch_module.device('cuda' if torch_module.cuda.is_available() else 'cpu')
        self.model_name = client.model
        self.dimensions = client.dimensions

    async def get_embeddings(self, texts, need_tensor=False):
        vectors = await client.embed(texts)
        if need_tensor:
            return torch_module.tensor(vectors, dtype=torch_module.float32, device=self.device)
        return vectors

    manager_class.__init__ = initialize
    manager_class.get_embeddings = get_embeddings
