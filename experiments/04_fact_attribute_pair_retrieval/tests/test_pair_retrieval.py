import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pair_retrieval import retrieve_pairs

def fact(id,attribute='follower_count',subject='user',obj='instagram',scope='UNKNOWN',session='s1',time='2023/05/25 (Thu) 05:26',turn=0):
 return {'fact_id':id,'evidence_id':'ev_'+id,'subject_key':subject,'attribute_key':attribute,'object_key':obj,'scope_key':scope,'session_id':session,'session_time':time,'turn_index':turn,'status':'unverified'}

class PairRetrievalTests(unittest.TestCase):
 def test_same_attribute_finds_all_earlier_candidates(self):
  facts=[fact('a'),fact('b',session='s2',time='2023/05/25 (Thu) 07:00'),fact('c',session='s3',time='2023/05/25 (Thu) 09:00')]
  pairs=retrieve_pairs(facts,max_candidates=10)
  self.assertEqual({(p['old_fact_id'],p['new_fact_id']) for p in pairs['pairs']},{('a','b'),('a','c'),('b','c')})
  self.assertTrue(all(p['candidate_only'] for p in pairs['pairs']))
 def test_other_attributes_platforms_and_subjects_not_paired(self):
  facts=[fact('a'),fact('b',attribute='content_strategy',session='s2',time='2023/05/25 (Thu) 06:00'),fact('c',obj='twitter',session='s3',time='2023/05/25 (Thu) 07:00'),fact('d',subject='sister',session='s4',time='2023/05/25 (Thu) 08:00')]
  self.assertEqual(retrieve_pairs(facts,max_candidates=10)['pairs'],[])
 def test_unknown_scope_is_candidate_not_match(self):
  p=retrieve_pairs([fact('a',scope='UNKNOWN'),fact('b',scope='public',session='s2',time='2023/05/25 (Thu) 06:00')],max_candidates=10)['pairs'][0]
  self.assertEqual(p['scope_match'],'UNKNOWN')
 def test_same_session_turn_order_is_used(self):
  p=retrieve_pairs([fact('a',turn=0),fact('b',turn=2)],max_candidates=10)['pairs'][0]
  self.assertEqual(p['time_order'],'BEFORE_SAME_SESSION')
 def test_equal_time_different_sessions_is_unknown_not_dropped(self):
  p=retrieve_pairs([fact('a'),fact('b',session='s2')],max_candidates=10)['pairs'][0]
  self.assertEqual(p['time_order'],'ORDER_UNKNOWN')
 def test_missing_attribute_not_guessed(self):
  result=retrieve_pairs([fact('a',attribute='UNKNOWN'),fact('b',session='s2',time='2023/05/25 (Thu) 06:00')],max_candidates=10)
  self.assertEqual(result['pairs'],[]);self.assertIn('a',result['unindexed_fact_ids'])
 def test_truncation_is_recorded(self):
  facts=[fact('a'),fact('b',session='s2',time='2023/05/25 (Thu) 06:00'),fact('c',session='s3',time='2023/05/25 (Thu) 07:00')]
  r=retrieve_pairs(facts,max_candidates=1);self.assertEqual(len([p for p in r['pairs'] if p['new_fact_id']=='c']),1);self.assertEqual(r['truncated_by_new_fact']['c'],1)

class DerivedKeyTests(unittest.TestCase):
 def test_derived_key_proposes_candidate_without_overwriting_raw(self):
  a=fact('a',attribute='social_media_followers_count');b=fact('b',attribute='follower_count_estimate',session='s2',time='2023/05/25 (Thu) 06:00')
  a['search_attribute_key']=b['search_attribute_key']='follower_count'
  p=retrieve_pairs([a,b],max_candidates=10)['pairs'][0]
  self.assertEqual(p['candidate_path'],'proposed_normalized_attribute_key')
  self.assertEqual(a['attribute_key'],'social_media_followers_count')
