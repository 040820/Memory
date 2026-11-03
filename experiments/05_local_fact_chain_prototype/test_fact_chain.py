import unittest
from fact_chain import FactChain

class FactChainTests(unittest.TestCase):
    def test_distinct_occurrences_and_idempotency(self):
        store = FactChain()
        store.add_chunk('s1/c1', '上海')
        store.add_chunk('s2/c1', '北京')
        a = store.add_fact('用户', '居住地', '上海', 's1/c1')
        b = store.add_fact('用户', '居住地', '北京', 's2/c1')
        self.assertNotEqual(a, b)
        self.assertEqual(a, store.add_fact('用户', '居住地', '上海', 's1/c1'))
        self.assertEqual({a, b}, {f['id'] for f in store.find_facts('用户', '居住地')})

    def test_relationships_between_same_entities_are_distinct(self):
        store = FactChain(); store.add_chunk('s/c', '工作和旅游')
        a = store.add_fact('用户', '工作于', '北京', 's/c')
        b = store.add_fact('用户', '旅游于', '北京', 's/c')
        self.assertNotEqual(a, b)
        self.assertEqual(len(store.find_facts('用户')), 2)

    def test_bidirectional_chain_and_roundtrip(self):
        store = FactChain(); store.add_chunk('s/c', '搬到了北京')
        a = store.add_fact('用户', '居住地', '上海', 's/c')
        b = store.add_fact('用户', '居住地', '北京', 's/c')
        store.add_change(a, b, 'change', 's/c', '用户明确搬家')
        restored = FactChain.from_dict(store.to_dict())
        self.assertEqual(restored.changes_for(a), restored.changes_for(b))
        self.assertEqual(restored.changes_for(a)[0]['evidence_id'], 's/c')

    def test_missing_evidence_and_collisions_rejected(self):
        store = FactChain(); store.add_chunk('s/c', '原文')
        with self.assertRaises(ValueError): store.add_chunk('s/c', '另一段')
        with self.assertRaises(ValueError): store.add_fact('用户', '居住地', '北京', 'missing')
        a = store.add_fact('用户', '居住地', '北京', 's/c')
        with self.assertRaises(ValueError): store.add_change(a, 'missing', 'change', 's/c', '理由')

if __name__ == '__main__': unittest.main()
