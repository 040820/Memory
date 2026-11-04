"""Launch the original main.py with a process-local remote embedding adapter.

This does not modify the original project files. Pass main.py's normal arguments.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path
from urllib.parse import urlparse

from remote_embedding import RemoteEmbeddingClient, install_manager_adapter
from qwen_adapter import install_non_thinking_adapter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-opt', required=True)
args, _ = parser.parse_known_args()
with open(args.opt, encoding='utf8') as stream:
    configuration = yaml.safe_load(stream)
experiment = configuration.get('experiment', {})
if experiment.get('enable_thinking') is not False:
    raise ValueError('experiment.enable_thinking must explicitly be false')
embedding_url = os.environ.get('LICO_EMBEDDING_URL', experiment['embedding_base_url'])
embedding_model = os.environ.get('LICO_EMBEDDING_MODEL', experiment['embedding_model'])
llm_url = configuration['llm']['base_url']
query_url = configuration.get('query_llm', {}).get('base_url') or llm_url
if experiment['llm_base_url'] != llm_url or query_url != llm_url:
    raise ValueError('experiment, llm, and query_llm URLs must match')
target_model = configuration['llm']['model']
if (configuration.get('query_llm', {}).get('model') or target_model) != target_model:
    raise ValueError('llm and query_llm models must match')
hosts = [urlparse(url).hostname for url in (embedding_url, llm_url)]
existing = os.environ.get('NO_PROXY', '')
no_proxy = ','.join(filter(None, [existing, 'localhost', '127.0.0.1', *hosts]))
os.environ['NO_PROXY'] = no_proxy
os.environ['no_proxy'] = no_proxy

import torch
from base.embeddings import EmbeddingManager
from base.llm import LLMManager

client = RemoteEmbeddingClient(embedding_url, embedding_model, experiment['embedding_dimensions'],
                               api_key=os.environ.get('LICO_EMBEDDING_API_KEY'))
install_manager_adapter(EmbeddingManager, client, torch)
install_non_thinking_adapter(LLMManager, target_model)
sys.argv[0] = str(ROOT / 'main.py')
runpy.run_path(sys.argv[0], run_name='__main__')
