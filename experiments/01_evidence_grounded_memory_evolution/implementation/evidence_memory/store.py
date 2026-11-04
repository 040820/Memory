"""Immutable source records and unverified fact candidates in one snapshot."""
import copy
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def stable_id(prefix, value):
 data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')
 return prefix+'_'+hashlib.sha256(data).hexdigest()


def known_time(value):
 if not isinstance(value,str) or not value.strip():raise ValueError('known_at required')
 try:return datetime.fromisoformat(value.replace('Z','+00:00'))
 except ValueError:raise ValueError('known_at must be an ISO timestamp') from None


class EvidenceStore:
 def __init__(self,path):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
  self.lock_path=self.path.with_suffix(self.path.suffix+'.lock')

 @contextmanager
 def _locked(self):
  with self.lock_path.open('a+b') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   try:yield
   finally:fcntl.flock(lock,fcntl.LOCK_UN)

 def _load(self):
  if not self.path.exists():return {'schema_version':1,'sources':{},'candidates':{}}
  with self.path.open(encoding='utf8') as f:d=json.load(f)
  if d.get('schema_version')!=1:raise ValueError('unsupported schema')
  return d

 def _write(self,data):
  tmp=None
  try:
   with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=self.path.parent,prefix='.evidence-',delete=False) as f:
    tmp=f.name;json.dump(data,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
   os.replace(tmp,self.path);tmp=None
   fd=os.open(str(self.path.parent),os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
  finally:
   if tmp and os.path.exists(tmp):os.unlink(tmp)

 def put_source(self,source):
  fields=('dataset_id','conversation_id','session_id','turn_id','speaker','text','known_at')
  if not isinstance(source,dict) or any(not isinstance(source.get(k),str) or not source[k].strip() for k in fields if k!='known_at') or 'known_at' not in source:raise ValueError('complete source required')
  if source['known_at'] is not None:known_time(source['known_at'])
  clean={k:source[k] for k in fields}
  for k in ('raw_session_time','time_precision','timezone','known_at_basis'):
   if k in source:clean[k]=source[k]
  sid=stable_id('ev',[clean[k] for k in ('dataset_id','conversation_id','session_id','turn_id')])
  with self._locked():
   d=self._load();prev=d['sources'].get(sid)
   if prev is not None and prev!=clean:raise ValueError('source identity conflict')
   if prev is None:d['sources'][sid]=clean;self._write(d)
  return sid

 def put_candidate(self,fact):
  fields=('subject','predicate','value','quote','evidence_id')
  if not isinstance(fact,dict) or any(not isinstance(fact.get(k),str) or not fact[k].strip() for k in fields):raise ValueError('complete candidate required')
  clean={k:fact[k] for k in fields}
  for k in ('scope','modality','cardinality','valid_from','extractor_id'):
   if k in fact:clean[k]=fact[k]
  clean['status']='unverified';fid=stable_id('fact',clean)
  with self._locked():
   d=self._load();source=d['sources'].get(clean['evidence_id'])
   if not source or clean['quote'] not in source['text']:raise ValueError('source quote mismatch')
   if fid not in d['candidates']:d['candidates'][fid]=clean;self._write(d)
  return fid

 def locate_quote(self,evidence_id,quote):
  """Return all exact Python character offsets, with an exclusive end."""
  if not isinstance(quote,str) or not quote:raise ValueError('nonempty quote required')
  with self._locked():text=self._load()['sources'][evidence_id]['text']
  spans=[];start=0
  while True:
   index=text.find(quote,start)
   if index<0:return spans
   spans.append((index,index+len(quote)));start=index+1

 def sources(self):
  with self._locked():return copy.deepcopy(self._load()['sources'])

 def get_candidate(self,fid):
  with self._locked():return copy.deepcopy(self._load()['candidates'][fid])


class StateStore(EvidenceStore):
 """Versioned snapshot; human/model assessments are proposals, not proofs."""
 CHECKS=('source_supported','identity_supported','predicate_supported','value_supported','scope_supported','modality_supported','time_supported','relation_supported')

 def record_assessment(self,fid,review):
  if not isinstance(review,dict) or any(review.get(k) is not True for k in self.CHECKS):raise ValueError('all explicit support checks required')
  if not isinstance(review.get('assessor_id'),str) or not review['assessor_id'].strip() or not isinstance(review.get('reason'),str) or not review['reason'].strip():raise ValueError('assessment identity and reason required')
  if review.get('relation') not in ('add','change','correction'):raise ValueError('unsupported relation')
  with self._locked():
   d=self._load();fact=d['candidates'].get(fid)
   if not fact or not fact.get('extractor_id') or fact['extractor_id']==review['assessor_id']:raise ValueError('distinct assessor identity required; semantic independence is not established')
   source=d['sources'].get(fact['evidence_id'])
   if not source or fact['quote'] not in source['text']:raise ValueError('source quote mismatch')
   old_id=review.get('old_state_id')
   if review['relation'] in ('change','correction'):
    matches=[x for x in d.get('states',[]) if x['state_id']==old_id and x['status']=='current']
    if len(matches)!=1:raise ValueError('assessment must bind one current target')
    old=matches[0]
    if (old['subject'],old['predicate'],old['scope'])!=(fact['subject'],fact['predicate'],fact.get('scope')):raise ValueError('assessment target mismatch')
    old_evidence=list(old['evidence_ids'])
   else:
    if old_id is not None:raise ValueError('ADD assessment cannot target a state')
    old_evidence=[]
   clean={k:review[k] for k in (*self.CHECKS,'assessor_id','reason','relation')};clean.update(fact_id=fid,evidence_id=fact['evidence_id'],old_state_id=old_id,input_evidence_ids=list(dict.fromkeys(old_evidence+[fact['evidence_id']])),review_revision=d.get('revision',0))
   aid=stable_id('assessment',clean);d.setdefault('assessments',{})[aid]=clean;self._write(d)
  return aid

 def commit(self,decision,*,expected_revision,idempotency_key):
  required=('decision_id','action','fact_id','assessment_id','old_state_id','known_at')
  if not isinstance(decision,dict) or any(k not in decision for k in required) or not isinstance(idempotency_key,str) or not idempotency_key.strip():raise ValueError('complete decision and idempotency key required')
  if not isinstance(decision['decision_id'],str) or not decision['decision_id'].strip():raise ValueError('decision_id required')
  known=known_time(decision['known_at']);decision_hash=stable_id('decision',decision)
  with self._locked():
   d=self._load();commits=d.setdefault('commits',{});previous=commits.get(idempotency_key)
   if previous:
    if previous['decision_hash']!=decision_hash:raise ValueError('idempotency conflict')
    return copy.deepcopy(previous['result'])
   revision=d.get('revision',0)
   if revision!=expected_revision:raise ValueError('revision conflict')
   if any(x.get('decision_id')==decision['decision_id'] for x in d.get('decisions',[])):raise ValueError('decision_id already used')
   fact=d['candidates'].get(decision['fact_id']);review=d.get('assessments',{}).get(decision['assessment_id'])
   if not fact or not review or review.get('fact_id')!=decision['fact_id']:raise ValueError('assessment is not bound to fact')
   if review.get('old_state_id')!=decision['old_state_id'] or review.get('review_revision')!=revision:raise ValueError('assessment target or revision mismatch')
   for evidence_id in review.get('input_evidence_ids',[]):
    dependency=d['sources'].get(evidence_id)
    if not dependency:raise ValueError('missing dependency source')
    try:
     if known_time(dependency['known_at'])>known:raise ValueError('future dependency')
    except TypeError:raise ValueError('incomparable dependency time') from None
   if any(review.get(k) is not True for k in self.CHECKS):raise ValueError('support incomplete')
   src=d['sources'].get(fact['evidence_id'])
   if not src or fact['quote'] not in src['text']:raise ValueError('source mismatch')
   try:
    if known_time(src['known_at'])>known:raise ValueError('future source')
    effective=known_time(fact['valid_from'])
    if effective>known:raise ValueError('future effective time')
   except (KeyError,TypeError):raise ValueError('unknown or incomparable time') from None
   if fact.get('modality') not in ('asserted','approximate') or fact.get('cardinality') not in ('single','multi') or not fact.get('scope'):raise ValueError('unsafe modality, scope or cardinality')
   action=decision['action'];relations={'ADD':'add','REPLACE':'change','CORRECT':'correction'}
   if action not in relations or review['relation']!=relations[action]:raise ValueError('action not supported by review')
   if action in ('REPLACE','CORRECT') and fact['modality']!='asserted':raise ValueError('uncertain statement cannot replace a state')
   states=d.setdefault('states',[]);old=None;key=(fact['subject'],fact['predicate'],fact['scope'])
   if action in ('REPLACE','CORRECT'):
    found=[x for x in states if x['state_id']==decision['old_state_id'] and x['status']=='current']
    if len(found)!=1:raise ValueError('exactly one current target required')
    old=found[0]
    if (old['subject'],old['predicate'],old['scope'])!=key or fact['cardinality']!='single':raise ValueError('identity scope or cardinality mismatch')
    try:
     if known_time(old['known_at'])>known or (action=='REPLACE' and effective<known_time(old['valid_from'])):raise ValueError('inconsistent time')
    except TypeError:raise ValueError('incomparable time') from None
   elif decision['old_state_id'] is not None or (fact['cardinality']=='single' and any(x['status']=='current' and (x['subject'],x['predicate'],x['scope'])==key for x in states)):
    raise ValueError('ADD conflicts with current single-valued state')
   state_id=stable_id('state',[decision['decision_id'],decision['fact_id']]);state={'state_id':state_id,'subject':fact['subject'],'predicate':fact['predicate'],'scope':fact['scope'],'value':fact['value'],'modality':fact['modality'],'cardinality':fact['cardinality'],'valid_from':fact['valid_from'],'valid_to':None,'known_at':decision['known_at'],'status':'current','fact_id':decision['fact_id'],'evidence_ids':[fact['evidence_id']],'decision_id':decision['decision_id']}
   if old:
    old['status']='historical' if action=='REPLACE' else 'corrected';old['superseded_by']=state_id;old['superseded_known_at']=decision['known_at']
    if action=='REPLACE':old['valid_to']=fact['valid_from']
    state['dependency_state_id']=old['state_id'];state['evidence_ids']=list(dict.fromkeys(old['evidence_ids']+state['evidence_ids']))
   states.append(state);d.setdefault('decisions',[]).append(dict(decision,applied=True,revision=revision+1));result={'state_id':state_id,'revision':revision+1};commits[idempotency_key]={'decision_hash':decision_hash,'result':result};d['revision']=revision+1;self._write(d)
   return copy.deepcopy(result)

 def list_states(self):
  with self._locked():return copy.deepcopy(self._load().get('states',[]))

 def states_known_at(self, cutoff):
  """Return a decision-time snapshot, not a claim of validity at a time."""
  when=known_time(cutoff)
  with self._locked():
   d=self._load();result=[]
   for state in d.get('states',[]):
    try:
     if known_time(state['known_at'])>when:continue
     if any(known_time(d['sources'][eid]['known_at'])>when for eid in state['evidence_ids']):
      raise ValueError('state contains future evidence')
     item=copy.deepcopy(state)
     if item.get('superseded_known_at') and known_time(item['superseded_known_at'])>when:
      item['status']='current';item['valid_to']=None
      item.pop('superseded_by',None);item.pop('superseded_known_at',None)
     result.append(item)
    except (TypeError,KeyError):raise ValueError('missing or incomparable time dependency') from None
   return result
