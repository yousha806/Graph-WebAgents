import json
import tempfile
import unittest
from pathlib import Path

from inference import eval_next_action


class EvalNextActionTests(unittest.TestCase):
    def test_action_accuracy_metrics(self):
        records = [
            {"gt_action_index": 1, "pred_action_index": 1, "gt_action": "CLICK", "pred_action": "CLICK"},
            {"gt_action_index": 2, "pred_action_index": 0, "gt_action": "TYPE", "pred_action": "CLICK"},
            {"gt_action_index": 3, "pred_action_index": None, "gt_action": "SELECT", "pred_action": None},
        ]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.5)
        self.assertAlmostEqual(eval_next_action.action_label_accuracy(records), 0.5)
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 1.0 / 3.0)

    def test_evaluate_file_parses_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            with path.open("w", encoding="utf8") as f:
                f.write(json.dumps({
                    "split": "test_task",
                    "baseline": "intern_image_allinputs_axtree",
                    "gt_action_index": 2,
                    "pred_action_index": 2,
                    "gt_action": "CLICK",
                    "pred_action": "CLICK",
                }) + "\n")
                f.write("{bad json}\n")

            metrics = eval_next_action.evaluate_file(path)
            self.assertEqual(metrics["num_lines"], 2)
            self.assertEqual(metrics["num_parsed"], 1)
            self.assertIn("intern_image_allinputs_axtree", metrics["baselines"])
            self.assertIn("test_task", metrics["splits"])


if __name__ == "__main__":
    unittest.main()
