import tempfile, unittest
from pathlib import Path
from evidence_memory.api_config import request_settings

class ApiConfigTests(unittest.TestCase):
 def test_existing_vllm_remains_unauthenticated(self):
  payload={'chat_template_kwargs':{'enable_thinking':False}}
  headers, body=request_settings({},payload)
  self.assertEqual(headers,{})
  self.assertEqual(body,payload)
 def test_online_key_and_non_thinking_top_level(self):
  with tempfile.TemporaryDirectory() as d:
   key=Path(d)/'key';key.write_text('test-secret')
   headers,body=request_settings({'api_key_file':str(key),'thinking_parameter':'top_level'}, {'chat_template_kwargs':{'enable_thinking':False}})
   self.assertEqual(headers['Authorization'],'Bearer test-secret')
   self.assertFalse(body['enable_thinking'])
   self.assertNotIn('chat_template_kwargs',body)
 def test_missing_key_is_an_error(self):
  with self.assertRaises((OSError,ValueError)):
   request_settings({'api_key_file':'/this/path/does/not/exist'}, {})
