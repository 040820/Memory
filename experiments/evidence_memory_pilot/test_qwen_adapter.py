import asyncio
import importlib.util
import os
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('qwen_adapter.py')
if PATH.exists():
    spec = importlib.util.spec_from_file_location('qwen_adapter', PATH)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
else:
    adapter = None


class QwenAdapterTest(unittest.TestCase):
    def test_qwen_requests_disable_thinking_and_keep_other_parameters(self):
        self.assertIsNotNone(adapter, 'Qwen adapter is missing')
        class FakeLLM:
            model = 'qwen3-8b'
            async def _generate_internal(self, prompt, **kwargs):
                return kwargs
        adapter.install_non_thinking_adapter(FakeLLM, 'qwen3-8b')
        result = asyncio.run(FakeLLM()._generate_internal('hello', temperature=0.2,
            extra_body={'foo': 1, 'chat_template_kwargs': {'enable_thinking': True, 'other': 2}}))
        self.assertEqual(result['temperature'], 0.2)
        self.assertEqual(result['extra_body']['foo'], 1)
        self.assertEqual(result['extra_body']['chat_template_kwargs'], {'enable_thinking': False, 'other': 2})

    def test_other_models_are_unmodified(self):
        self.assertIsNotNone(adapter, 'Qwen adapter is missing')
        class FakeLLM:
            model = 'other'
            async def _generate_internal(self, prompt, **kwargs): return kwargs
        adapter.install_non_thinking_adapter(FakeLLM, 'qwen3-8b')
        self.assertEqual(asyncio.run(FakeLLM()._generate_internal('hi')), {})

    def test_proxy_is_removed_only_while_client_is_constructed(self):
        self.assertIsNotNone(adapter, 'Qwen adapter is missing')
        class FakeLLM:
            def __init__(self):
                self.saw_proxy = 'ALL_PROXY' in os.environ
        previous = os.environ.get('ALL_PROXY')
        os.environ['ALL_PROXY'] = 'socks5://test-only'
        try:
            adapter.install_local_client_proxy_guard(FakeLLM)
            instance = FakeLLM()
            self.assertFalse(instance.saw_proxy)
            self.assertEqual(os.environ['ALL_PROXY'], 'socks5://test-only')
        finally:
            if previous is None: os.environ.pop('ALL_PROXY', None)
            else: os.environ['ALL_PROXY'] = previous


if __name__ == '__main__': unittest.main()
