"""Run one synthetic input-only LiCoMemory smoke check; no gold is loaded."""
import asyncio
import datetime
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONUTF8'] = '1'
os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY', ''), '10.11.154.236', '127.0.0.1', 'localhost']))
os.environ['no_proxy'] = os.environ['NO_PROXY']

import torch
import yaml
from remote_embedding import RemoteEmbeddingClient, install_manager_adapter
from qwen_adapter import install_non_thinking_adapter, install_local_client_proxy_guard
from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from init.config import Config
from init.graph_rag import GraphRAG


async def run():
    config_path = HERE / 'baseline_qwen.yaml'
    with config_path.open(encoding='utf8') as f:
        settings = yaml.safe_load(f)
    exp = settings['experiment']
    if exp.get('enable_thinking') is not False:
        raise ValueError('non-thinking setting required')
    install_manager_adapter(EmbeddingManager, RemoteEmbeddingClient(exp['embedding_base_url'], exp['embedding_model'], exp['embedding_dimensions']), torch)
    install_non_thinking_adapter(LLMManager, settings['llm']['model'])
    install_local_client_proxy_guard(LLMManager)
    config = Config.parse(config_path)
    config.data_type = 'LongmemEval'
    config.index_name = 'smoke_graph'
    with (HERE / 'controlled_inputs.json').open(encoding='utf8') as f:
        case = next(x for x in json.load(f) if x['case_id'] == 'change_01')
    corpus = [{'session_id': m['id'], 'session_time': m['time'],
               'context': '"user": ' + json.dumps(m['text'], ensure_ascii=False)} for m in case['messages']]
    base = HERE / 'smoke_runs'
    base.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='controlled_', dir=base))
    result = {'case_id': case['case_id'], 'status': 'started', 'question': case['queries'][0]['question'],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'config': str(config_path.relative_to(ROOT)), 'gold_loaded': False, 'dataset': 'synthetic_controlled'}
    (run_dir / 'status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    started = time.monotonic()
    try:
        graph = GraphRAG(config, str(run_dir))
        await graph.insert(corpus)
        stats = graph.core.graph.get_graph_stats()
        response = await graph.query(case['queries'][0]['question'], question_time='2024/06/02')
        answer = response.get('answer', '') if isinstance(response, dict) else str(response)
        failure = not answer or answer.startswith(('Error processing query:', 'Unable to generate answer'))
        result.update(status='failed' if failure else 'completed_unscored', graph_stats=stats, response=response,
                      elapsed_seconds=round(time.monotonic() - started, 2))
    except Exception as e:
        result.update(status='error', error_type=type(e).__name__, error=str(e), elapsed_seconds=round(time.monotonic() - started, 2))
    (run_dir / 'status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    print('SMOKE_RUN_DIR', run_dir, 'STATUS', result['status'], flush=True)
    if result['status'] != 'completed_unscored':
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(run())
