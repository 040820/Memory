import json,unittest,torch
from serialize_result import to_jsonable
class TestSerialization(unittest.TestCase):
 def test_tensors_are_numbers_and_arrays(self):
  data={'score':torch.tensor(0.5),'vectors':[torch.tensor([1,2])]}
  result=to_jsonable(data)
  self.assertEqual(result,{'score':0.5,'vectors':[[1,2]]})
  json.dumps(result,allow_nan=False)
if __name__=='__main__':unittest.main()
