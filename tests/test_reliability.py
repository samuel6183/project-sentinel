import unittest
import numpy as np
from sentinel.generate import generate
from sentinel.reliability import blocks,shifted_test,interval,SEEDS

class ReliabilityTests(unittest.TestCase):
    def test_partition_integrity(self):
        b=blocks(generate(10000,101))
        self.assertEqual(sum(map(len,b)),10000)
        for a,c in zip(b,b[1:]):
            self.assertLess(a.timestamp.max(),c.timestamp.min())
    def test_stress_preserves_features(self):
        df=generate(10000,101)
        shifted=shifted_test(df,101)
        self.assertTrue(df.drop(columns="synthetic_fraud_label").equals(shifted.drop(columns="synthetic_fraud_label")))
        self.assertFalse(df.synthetic_fraud_label.equals(shifted.synthetic_fraud_label))
    def test_seed_and_interval(self):
        self.assertNotIn(42,SEEDS)
        self.assertEqual(len(set(SEEDS)),10)
        mean,lo,hi=interval([1,2,3])
        self.assertEqual(mean,2)
        self.assertLess(lo,mean)
        self.assertGreater(hi,mean)
