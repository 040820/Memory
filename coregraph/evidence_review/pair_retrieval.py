"""Fact-level candidate retrieval; matching is a proposal, never proof."""
from collections import defaultdict
from datetime import datetime


def timestamp(raw):
 try:return datetime.strptime(raw,'%Y/%m/%d (%a) %H:%M')
 except (ValueError,TypeError):return None


def known(value):return isinstance(value,str) and bool(value.strip()) and value.strip().upper()!='UNKNOWN'

def match(a,b):
 if not known(a) or not known(b):return 'UNKNOWN'
 return 'MATCH' if a==b else 'MISMATCH'


def retrieve_pairs(facts,*,max_candidates):
 if max_candidates<1:raise ValueError('max_candidates must be positive')
 ids=[f.get('fact_id') for f in facts]
 if any(not known(x) for x in ids) or len(set(ids))!=len(ids):raise ValueError('unique fact ids required')
 eligible=[];unindexed=[]
 for f in facts:
  if not all(known(f.get(k)) for k in ('evidence_id','subject_key','attribute_key','session_id')) or f['subject_key']!='user' or timestamp(f.get('session_time')) is None or not isinstance(f.get('turn_index'),int):
   unindexed.append(f['fact_id']);continue
  eligible.append(f)
 eligible.sort(key=lambda x:(timestamp(x['session_time']),x['session_id'],x['turn_index'],x['fact_id']))
 by_key=defaultdict(list);pairs=[];truncated={};excluded=defaultdict(int)
 for new in eligible:
  key=(new['subject_key'],new.get('search_attribute_key',new['attribute_key']));ranked=[]
  for old in by_key[key]:
   object_match=match(old.get('object_key'),new.get('object_key'));scope_match=match(old.get('scope_key'),new.get('scope_key'))
   if object_match=='MISMATCH' or scope_match=='MISMATCH':excluded['object_or_scope_mismatch']+=1;continue
   ot,nt=timestamp(old['session_time']),timestamp(new['session_time'])
   order='BEFORE' if ot<nt else ('BEFORE_SAME_SESSION' if old['session_id']==new['session_id'] and old['turn_index']<new['turn_index'] else 'ORDER_UNKNOWN')
   ranked.append({'old_fact_id':old['fact_id'],'new_fact_id':new['fact_id'],'old_evidence_id':old['evidence_id'],'new_evidence_id':new['evidence_id'],'subject_match':'KEY_MATCH_UNVERIFIED','attribute_match':'KEY_MATCH_UNVERIFIED','object_match':object_match,'scope_match':scope_match,'time_order':order,'candidate_path':'exact_proposed_subject_attribute_keys' if old['attribute_key']==new['attribute_key'] else 'proposed_normalized_attribute_key','candidate_only':True})
  ranked.reverse()
  if len(ranked)>max_candidates:truncated[new['fact_id']]=len(ranked)-max_candidates
  pairs.extend(ranked[:max_candidates]);by_key[key].append(new)
 return {'pairs':pairs,'unindexed_fact_ids':unindexed,'truncated_by_new_fact':truncated,'excluded_counts':dict(excluded)}
