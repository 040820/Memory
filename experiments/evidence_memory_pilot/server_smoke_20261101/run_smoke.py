import asyncio, datetime, hashlib, json, os, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/cwf/Code/LiCoMemory-main')
EXP=ROOT/'experiments/evidence_memory_pilot'
OUT=EXP/'server_smoke_20261101'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(EXP))
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(key,None)
os.environ['NO_PROXY']='127.0.0.1,localhost'
os.environ['no_proxy']=os.environ['NO_PROXY']
import torch
from remote_embedding import RemoteEmbeddingClient, install_manager_adapter
from qwen_adapter import install_non_thinking_adapter
from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from init.config import Config
from init.graph_rag import GraphRAG

async def main():
    result={'status':'started','case_id':'change_01','gold_loaded':False,'protocol':'synthetic controlled preflight, unscored','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUT/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    start=time.monotonic()
    try:
        config=Config.parse(EXP/'baseline_qwen.yaml')
        config.data_type='LongmemEval'; config.index_name='server_smoke_graph'
        config.llm.base_url='http://127.0.0.1:8011/v1'; config.query_llm.base_url=config.llm.base_url
        install_non_thinking_adapter(LLMManager,config.llm.model)
        install_manager_adapter(EmbeddingManager,RemoteEmbeddingClient('http://127.0.0.1:8015/v1','bge-m3',1024),torch)
        data=json.loads((EXP/'controlled_inputs.json').read_text(encoding='utf8'))
        case=next(x for x in data if x['case_id']=='change_01')
        corpus=[{'session_id':m['id'],'session_time':m['time'],'context':'"user": '+json.dumps(m['text'],ensure_ascii=False)} for m in case['messages']]
        graph=GraphRAG(config,str(OUT)); await graph.insert(corpus)
        stats=graph.core.graph.get_graph_stats()
        response=await graph.query(case['queries'][0]['question'],question_time='2024/06/02')
        answer=response.get('answer','') if isinstance(response,dict) else str(response)
        failed=not answer or answer.startswith(('Error processing query:','Unable to generate answer'))
        result.update(status='failed' if failed else 'completed_unscored',graph_stats=stats,response=response)
    except Exception as e:
        result.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    result['elapsed_seconds']=round(time.monotonic()-start,2)
    (OUT/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    print('SERVER_SMOKE_STATUS',result['status'],'ELAPSED',result['elapsed_seconds'],flush=True)
    if result['status']!='completed_unscored': raise SystemExit(1)
asyncio.run(main())
