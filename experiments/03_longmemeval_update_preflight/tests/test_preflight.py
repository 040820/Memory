import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preflight import extract_sources,select_pairs,query_seeds

class PreflightTests(unittest.TestCase):
 def setUp(self):
  self.sample={'question_id':'q','question_time':'2023/05/20 (Sat) 10:00','question':'SECRET_QUESTION','answer':'SECRET_GOLD','sessions':[{'session_id':'s1','session_time':'2023/05/19 (Fri) 10:00','messages':[{'role':'user','content':'I live in Beijing.'},{'role':'assistant','content':'OK'}]},{'session_id':'s2','session_time':'2023/05/19 (Fri) 11:00','messages':[{'role':'user','content':'I moved to Shanghai.'}]},{'session_id':'future','session_time':'2023/05/20 (Sat) 11:00','messages':[{'role':'user','content':'Future.'}]}]}
 def test_only_prior_user_turns_and_no_gold(self):
  rows,stats=extract_sources(self.sample);self.assertEqual(len(rows),2);self.assertEqual(stats['excluded_after_or_equal_cutoff'],1);self.assertNotIn('SECRET',str(rows));self.assertEqual(rows[0]['turn_index'],0)
 def test_tied_time_and_same_session_do_not_create_pair(self):
  rows,_=extract_sources(self.sample);rows[1]['session_time']=rows[0]['session_time'];self.assertEqual(select_pairs(rows,[[1,0],[1,0]],min_similarity=0),[])
  rows[1]['session_time']='2023/05/19 (Fri) 11:00';rows[1]['session_id']=rows[0]['session_id'];self.assertEqual(select_pairs(rows,[[1,0],[1,0]],min_similarity=0),[])
 def test_cross_session_nearest_and_no_question(self):
  rows,_=extract_sources(self.sample);pairs=select_pairs(rows,[[1,0],[0.9,0.1]],min_similarity=0.5);self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]['old']['text'],'I live in Beijing.');self.assertNotIn('SECRET',str(pairs))
 def test_seed_order_and_limit(self):
  self.assertEqual(query_seeds([[1,0],[0,1]], [1,0],1),[0])

class PairSelectionTests(PreflightTests):
 def test_pairs_respect_seed_order_and_budget(self):
  from preflight import choose_pairs
  pairs=[{'old_index':0,'new_index':3},{'old_index':1,'new_index':4},{'old_index':3,'new_index':5}]
  result=choose_pairs(pairs,[4,3],2)
  self.assertEqual([(x['old_index'],x['new_index']) for x in result],[(1,4),(0,3)])
 def test_no_future_source_used_even_if_same_minute(self):
  self.sample['sessions'][-1]['session_time']='2023/05/20 (Sat) 10:00'
  rows,stats=extract_sources(self.sample)
  self.assertEqual(stats['excluded_after_or_equal_cutoff'],1)
  self.assertNotIn('Future.',str(rows))
