import unittest
import numpy as np
from sentinel.generate import generate
from sentinel.advanced import split,choose_threshold,compare,features

class AdvancedTests(unittest.TestCase):
    def test_temporal_boundaries(self):
        parts=split(generate(20000))
        for a,b in zip(parts,parts[1:]):
            self.assertLess(a.timestamp.max(),b.timestamp.min())
        self.assertEqual(sum(map(len,parts)),20000)
        self.assertNotIn("synthetic_fraud_label",features(parts[0]).columns)
        self.assertNotIn("actual_settlement",features(parts[0]).columns)
    def test_threshold_budget_and_ties(self):
        y=np.array([1,0,1,0,0])
        p=np.array([.9,.9,.8,.2,.1])
        t=choose_threshold(y,p,max_alert_rate=.2)
        self.assertEqual(t["validation_alerts"],0)
        self.assertEqual(int((p>=t["threshold"]).sum()),0)
        t=choose_threshold(y,p,max_alert_rate=.6)
        self.assertEqual(t["validation_alerts"],3)
    def test_comparison(self):
        table,info=compare(generate(20000))
        self.assertEqual(len(table),5)
        self.assertIn(info["champion_selected_on_validation"],table.model.tolist())
        self.assertTrue((table.average_precision.between(0,1)).all())
        self.assertTrue((table.validation_alerts<=int(info["partitions"][2]["rows"]*.05)).all())
        self.assertTrue(((table.tn+table.fp+table.fn+table.tp)==info["partitions"][3]["rows"]).all())
