import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.metrics import aggregate_studies, evaluate_predictions
from cxr.submission import export_submission


class StudyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"Subject_id": ["0001", "0001", "0002"],
                                   "Study_id": ["00001", "00001", "00002"]})
        self.targets = np.array([[1]*10, [1]*10, [0]*10], dtype=np.float32)
        self.probs = np.array([[0.8]*10, [0.6]*10, [0.2]*10])

    def test_pools_views_before_scoring(self):
        result = evaluate_predictions(self.frame, self.probs, self.targets)
        self.assertEqual(result["study_count"], 2)
        self.assertEqual(result["macro_auc"], 1.0)
        np.testing.assert_allclose(result["study_probs"][:, 0], [0.7, 0.2])

    def test_conflicting_labels_fail(self):
        self.targets[1, 0] = 0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            aggregate_studies(self.frame, self.probs, self.targets)

    def test_nonfinite_predictions_fail(self):
        self.probs[0, 0] = np.nan
        with self.assertRaises(ValueError):
            aggregate_studies(self.frame, self.probs)

    def test_single_class_auc_is_explicitly_undefined(self):
        result = evaluate_predictions(self.frame, self.probs, np.ones((3, 10)))
        self.assertIsNone(result["macro_auc"])
        self.assertEqual(result["auc_classes_evaluated"], 0)

    def test_export_preserves_ids_and_complete_probabilities(self):
        study, probs, _ = aggregate_studies(self.frame, self.probs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "submission.csv"
            report = export_submission(study, probs, path)
            output = pd.read_csv(path, dtype={"Subject_id": str, "Study_id": str})
            self.assertEqual(report["rows"], 20)
            self.assertEqual(output.iloc[0].Subject_id, "0001")
            self.assertEqual(output.iloc[0].Study_id, "00001")
            self.assertEqual(output.groupby(["Subject_id", "Study_id"]).size().tolist(), [10, 10])
            self.assertFalse(report["official_scoring_contract_confirmed"])


if __name__ == "__main__":
    unittest.main()
