import unittest
import pandas as pd
from sentinel.generate import generate
from sentinel.analysis import validate,providers,reconciliation
from sentinel.model import evaluate

class SentinelTests(unittest.TestCase):
    def test_reproducibility(self):
        a=generate(1000,42); b=generate(1000,42)
        pd.testing.assert_frame_equal(a,b)
        validate(a)
        self.assertEqual(len(a),1000)
        self.assertEqual(a.transaction_id.nunique(),1000)
    def test_metrics(self):
        a=generate(5000)
        p=providers(a)
        self.assertEqual(p.attempts.sum(),len(a))
        self.assertTrue(((p.approval_rate+p.failure_rate+p.pending_rate-1).abs()<1e-10).all())
        self.assertAlmostEqual(p.absolute_discrepancy_kes.sum(),reconciliation(a).difference_kes.abs().sum(),places=5)
    def test_duplicate_rejected(self):
        a=generate(10)
        with self.assertRaises(ValueError):
            validate(pd.concat([a,a]))
    def test_holdout(self):
        m=evaluate(generate(10000))
        self.assertEqual(m["train_rows"],8000)
        self.assertEqual(m["test_rows"],2000)
        self.assertTrue(0<=m["rf_average_precision"]<=1)

if __name__=="__main__":
    unittest.main()
