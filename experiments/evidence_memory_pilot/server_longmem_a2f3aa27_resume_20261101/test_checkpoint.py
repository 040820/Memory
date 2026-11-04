import tempfile,unittest
from pathlib import Path
from checkpoint import save_checkpoint,load_checkpoint
class TestCheckpoint(unittest.TestCase):
 def test_roundtrip_and_mismatch(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'3.json';save_checkpoint(p,'abc','response')
   self.assertEqual(load_checkpoint(p,'abc'),'response')
   with self.assertRaises(ValueError):load_checkpoint(p,'changed')
 def test_empty_response_is_not_saved(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'3.json'
   with self.assertRaises(ValueError):save_checkpoint(p,'abc','')
   self.assertFalse(p.exists())
if __name__=='__main__':unittest.main()
