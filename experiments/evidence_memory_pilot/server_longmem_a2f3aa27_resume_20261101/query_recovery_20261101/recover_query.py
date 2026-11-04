import asyncio,datetime,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent;EXP=PREVIOUS.parent;ROOT=Path('/home/cwf/Code/LiCoMemory-main');SOURCE=EXP/'server_longmem_a2f3aa27_20261101'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP))
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(k,None)
os.environ['NO_PROXY']='127.0.0.1,localhost';os.environ['no_proxy']=os.environ['NO_PROXY']
import torch
from remote_embedding import RemoteEmbeddingClient,install_manager_adapter
from qwen_adapter import install_non_thinking_adapter
from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from init.config import Config
from init.graph_rag import GraphRAG
from serialize_result import to_jsonable

def save(data):
 tmp=HERE/'result.json.tmp';out=HERE/'result.json'
 with tmp.open('w',encoding='utf8') as f:json.dump(to_jsonable(data),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,out)
async def main():
 start=time.monotonic();result={'status':'started','query_rerun':True,'graph_reused':True,'gold_loaded_into_model':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};save(result)
 try:
  ck=list((PREVIOUS/'checkpoints').glob('*.json'))
  if len(ck)!=227:raise ValueError('checkpoint count is not 227')
  visible=json.loads((SOURCE/'visible_input.json').read_text(encoding='utf8'))
  config=Config.parse(EXP/'baseline_qwen.yaml');config.data_type='LongmemEval';config.index_name='longmem_graph';config.llm.base_url='http://127.0.0.1:8011/v1';config.query_llm.base_url=config.llm.base_url
  install_non_thinking_adapter(LLMManager,config.llm.model);install_manager_adapter(EmbeddingManager,RemoteEmbeddingClient('http://127.0.0.1:8015/v1','bge-m3',1024),torch)
  graph=GraphRAG(config,str(PREVIOUS));graph.core.graph.load_graph(str(PREVIOUS/'longmem_graph.pkl'))
  result['graph_stats']=graph.core.graph.get_graph_stats();result['chunks_in_graph']=len(graph.core.graph.chunk_storage);result['checkpoint_count']=len(ck)
  if result['chunks_in_graph']!=227:raise ValueError('graph chunk count is not 227')
  result['status']='querying';save(result)
  response=await graph.query(visible['question'],question_time=visible['question_time']);answer=response.get('answer','') if isinstance(response,dict) else str(response)
  if not answer or answer.startswith(('Error processing query:','Unable to generate answer')):raise ValueError('query returned error or empty answer')
  result['response']=response;result['status']='completed_unscored'
 except Exception as e:result.update(status='error',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
 result['elapsed_seconds']=round(time.monotonic()-start,2)
 try:save(result)
 except Exception as e:
  result.pop('response',None);result.update(status='serialization_error',error_type=type(e).__name__,error=str(e));save(result)
 print('RECOVERY_STATUS',result['status'],'ELAPSED',result['elapsed_seconds'],flush=True)
 if result['status']!='completed_unscored':raise SystemExit(1)
asyncio.run(main())
