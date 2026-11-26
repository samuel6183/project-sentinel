import unittest
import numpy as np
from sentinel.confirmation import calibration_bins,psi,SEEDS
from sentinel.reliability import SEEDS as V3
class ConfirmationTests(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(len(set(SEEDS)),20)
        self.assertFalse(set(SEEDS)&set(V3))
        self.assertNotIn(42,SEEDS)
    def test_bins(self):
        rows=calibration_bins(np.array([0,1,1]),np.array([0,.02,1]),np.array([0,.01,.05,1.0001]))
        self.assertEqual(sum(x["count"] for x in rows),3)
        self.assertEqual(sum(x["observed_sum"] for x in rows),2)
    def test_psi(self):
        a=np.arange(100,dtype=float)
        self.assertAlmostEqual(psi(a,a),0)
        self.assertGreater(psi(a,a+30),0)
