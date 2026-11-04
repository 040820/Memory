import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from answer import payload,validate
class AnswerTests(unittest.TestCase):
 def test_only_visible_source_fields_are_sent(self):
  sources=[{'evidence_id':'e','text':'hello','session_time':'2023/01/01','gold':'SECRET'}]
  p=payload('question','time',sources,None,model='x',max_tokens=100)
  self.assertNotIn('SECRET',str(p));self.assertNotIn('relation_suggestions',p['messages'][1]['content'])
  x=validate(json.dumps({'answer':'hello','citations':['e']}),sources);self.assertTrue(x['citations_exist']);self.assertFalse(x['semantic_support_verified'])
  self.assertFalse(validate(json.dumps({'answer':'hello','citations':['bad']}),sources)['citations_exist'])
