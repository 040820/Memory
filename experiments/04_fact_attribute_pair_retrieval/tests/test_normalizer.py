import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from normalization import normalize_attribute
class NormalizationTests(unittest.TestCase):
 def test_generic_social_follower_name_and_qualifier_normalized(self):
  self.assertEqual(normalize_attribute('social_media_followers_count','instagram')['key'],'follower_count')
  self.assertEqual(normalize_attribute('follower_count_estimate','instagram')['key'],'follower_count')
 def test_following_is_not_follower(self):
  self.assertNotEqual(normalize_attribute('following_count','instagram')['key'],normalize_attribute('follower_count','instagram')['key'])
 def test_unqualified_social_prefix_not_stripped_without_platform(self):
  self.assertEqual(normalize_attribute('social_media_followers_count','UNKNOWN')['key'],'social_media_follower_count')
 def test_unknown_remains_unknown(self):
  self.assertEqual(normalize_attribute('UNKNOWN','instagram')['key'],'UNKNOWN')
 def test_raw_and_rules_preserved(self):
  v=normalize_attribute('follower_count_estimate','instagram');self.assertEqual(v['raw'],'follower_count_estimate');self.assertIn('remove_estimate_qualifier',v['rules']);self.assertEqual(v['status'],'candidate_only')
