"""Gold-free chronological source preparation and embedding candidate retrieval."""
import hashlib,json
from collections import Counter
from datetime import datetime
import numpy as np


def parse_time(value):
 try:return datetime.strptime(value,'%Y/%m/%d (%a) %H:%M')
 except (TypeError,ValueError):return None


def extract_sources(sample):
 cutoff=parse_time(sample.get('question_time'))
 if cutoff is None:raise ValueError('question time is unknown')
 stats=Counter();rows=[];seen=set()
 for session in sample['sessions']:
  raw=session.get('session_time');ts=parse_time(raw)
  for index,msg in enumerate(session.get('messages',[])):
   if msg.get('role')!='user':continue
   if ts is None:stats['excluded_unknown_time']+=1;continue
   if ts>=cutoff:stats['excluded_after_or_equal_cutoff']+=1;continue
   text=msg.get('content');sid=session.get('session_id')
   if not isinstance(text,str) or not text.strip() or not isinstance(sid,str) or not sid:stats['excluded_missing_source']+=1;continue
   identity=[sample['question_id'],sid,str(index)];key=json.dumps(identity,ensure_ascii=False,separators=(',',':'))
   eid='ev_'+hashlib.sha256(key.encode()).hexdigest()
   if eid in seen:raise ValueError('duplicate source ID')
   seen.add(eid);rows.append({'evidence_id':eid,'session_id':sid,'turn_index':index,'speaker':'user','text':text,'session_time':raw,'time_precision':'minute','timezone':'unknown'});stats['included']+=1
 rows.sort(key=lambda x:(parse_time(x['session_time']),x['session_id'],x['turn_index']))
 return rows,dict(stats)


def _normalize(vectors):
 x=np.asarray(vectors,dtype=np.float64)
 if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('invalid embeddings')
 norms=np.linalg.norm(x,axis=1)
 if (norms==0).any():raise ValueError('zero embedding')
 return x/norms[:,None]


def select_pairs(sources,vectors,*,min_similarity):
 if len(sources)!=len(vectors):raise ValueError('embedding count mismatch')
 matrix=_normalize(vectors);scores=matrix@matrix.T;out=[]
 for new_index,new in enumerate(sources):
  valid=[i for i in range(new_index) if sources[i]['session_id']!=new['session_id'] and parse_time(sources[i]['session_time'])<parse_time(new['session_time'])]
  if not valid:continue
  best=max(valid,key=lambda i:(float(scores[new_index,i]),-i));score=float(scores[new_index,best])
  if score<min_similarity:continue
  out.append({'new_index':new_index,'old_index':best,'similarity':score,'old':sources[best],'new':new,'interpretation':'retrieval_candidate_not_proof'})
 return out


def query_seeds(vectors,query_vector,k):
 if k<1:raise ValueError('k must be positive')
 matrix=_normalize(vectors);query=_normalize([query_vector])[0];scores=matrix@query
 return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))[:k]


def choose_pairs(pairs,seeds,max_pairs):
 if max_pairs<1:raise ValueError('max_pairs must be positive')
 out=[];seen=set()
 for index in seeds:
  matches=[p for p in pairs if p['new_index']==index]
  for pair in matches:
   key=(pair['old_index'],pair['new_index'])
   if key not in seen:seen.add(key);out.append(pair)
   if len(out)>=max_pairs:return out
 return out
