import asyncio
import unittest
from types import SimpleNamespace

import torch

from base.embeddings import EmbeddingManager


class _FakeEmbeddingsAPI:
    """模拟 OpenAI 兼容 /v1/embeddings 响应（不联网）。"""

    async def create(self, input, model):
        vectors = [[0.1, 0.2, 0.3] for _ in input]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=v) for v in vectors])


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsAPI()


def make_manager():
    config = SimpleNamespace(
        api_type='openai', api_key='EMPTY', model='bge-m3',
        base_url='http://127.0.0.1:9/v1', cache_dir=None,
        dimensions=3, max_token_size=100, embed_batch_size=2,
        embedding_func_max_async=2)
    manager = EmbeddingManager(config)
    manager.client = _FakeClient()   # 替换真实客户端，测试不联网
    return manager


class RemoteEmbeddingTypeTests(unittest.TestCase):
    """远程 embedding 返回类型必须与 hf 路径一致（need_tensor 语义）。"""

    def test_need_tensor_returns_tensor(self):
        manager = make_manager()
        result = asyncio.run(manager.get_embeddings(['a', 'b'], need_tensor=True))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual((2, 3), tuple(result.shape))

    def test_default_returns_list(self):
        manager = make_manager()
        result = asyncio.run(manager.get_embeddings(['a']))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_query_similarity_path_works(self):
        # 复现 query_processor 的调用序列：
        # get_embeddings(need_tensor=True) -> cosine_similarity_tensor
        manager = make_manager()
        query = asyncio.run(manager.get_embeddings(['q'], need_tensor=True))
        entities = manager.transfer_to_tensor(
            asyncio.run(manager.get_embeddings(['e1', 'e2'])))
        similarities = manager.cosine_similarity_tensor(query, entities)
        self.assertIsInstance(similarities, torch.Tensor)
        self.assertEqual((1, 2), tuple(similarities.shape))
        # 相似度值必须是数而不是报错回退的 0.0 标量
        self.assertEqual(2, similarities.numel())


if __name__ == '__main__':
    unittest.main()
