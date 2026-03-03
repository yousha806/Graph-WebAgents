"""Tests for inference/eval_next_action.py.

Covers all metrics with:
  - The new pred_action_indices (list) field (primary format)
  - Backwards-compat with the legacy scalar pred_action_index field
  - Edge cases: parse failures, empty lists, missing keys
"""
import json
import tempfile
import unittest
from pathlib import Path

from inference import eval_next_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(gt_idx, pred_indices, gt_action=None, pred_action=None,
         annotation_id="task1", action_uid=None):
    """Build a minimal prediction record using the new list field."""
    r = {
        "gt_action_index": gt_idx,
        "pred_action_indices": pred_indices,
        "annotation_id": annotation_id,
    }
    if gt_action is not None:
        r["gt_action"] = gt_action
    if pred_action is not None:
        r["pred_action"] = pred_action
    if action_uid is not None:
        r["action_uid"] = action_uid
    return r


def _scalar_row(gt_idx, pred_idx, gt_action=None, pred_action=None,
                annotation_id="task1"):
    """Old-format record with scalar pred_action_index (backwards-compat)."""
    r = {"gt_action_index": gt_idx, "pred_action_index": pred_idx,
         "annotation_id": annotation_id}
    if gt_action is not None:
        r["gt_action"] = gt_action
    if pred_action is not None:
        r["pred_action"] = pred_action
    return r


# ---------------------------------------------------------------------------
# action_index_accuracy (ElemAcc / Top-1)
# ---------------------------------------------------------------------------

class TestActionIndexAccuracy(unittest.TestCase):
    def test_all_correct(self):
        records = [_row(1, [1, 2, 3]), _row(4, [4, 0, 1])]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 1.0)

    def test_none_correct(self):
        records = [_row(1, [2, 3, 4]), _row(0, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.0)

    def test_partial_correct(self):
        records = [_row(1, [1, 2, 3]), _row(0, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.5)

    def test_empty_prediction_list(self):
        records = [_row(1, [])]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.0)

    def test_missing_gt(self):
        records = [{"pred_action_indices": [1, 2, 3]}]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.0)

    def test_no_records(self):
        self.assertAlmostEqual(eval_next_action.action_index_accuracy([]), 0.0)

    def test_scalar_fallback_correct(self):
        records = [_scalar_row(2, 2)]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 1.0)

    def test_scalar_fallback_wrong(self):
        records = [_scalar_row(2, 3)]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.0)

    def test_scalar_fallback_none(self):
        records = [_scalar_row(2, None)]
        self.assertAlmostEqual(eval_next_action.action_index_accuracy(records), 0.0)


# ---------------------------------------------------------------------------
# top3_element_accuracy
# ---------------------------------------------------------------------------

class TestTop3ElementAccuracy(unittest.TestCase):
    def test_gt_at_rank1(self):
        records = [_row(1, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 1.0)

    def test_gt_at_rank2(self):
        records = [_row(2, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 1.0)

    def test_gt_at_rank3(self):
        records = [_row(3, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 1.0)

    def test_gt_not_in_list(self):
        records = [_row(4, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 0.0)

    def test_partial(self):
        records = [_row(1, [1, 2, 3]), _row(2, [1, 2, 3]), _row(9, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 2 / 3)

    def test_scalar_fallback(self):
        records = [_scalar_row(1, 1), _scalar_row(2, 3)]
        self.assertAlmostEqual(eval_next_action.top3_element_accuracy(records), 0.5)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank(unittest.TestCase):
    def test_gt_at_rank1(self):
        records = [_row(1, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 1.0)

    def test_gt_at_rank2(self):
        records = [_row(2, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 0.5)

    def test_gt_at_rank3(self):
        records = [_row(3, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 1 / 3)

    def test_gt_not_in_list_is_zero_not_1_over_n(self):
        # Miss → rr=0, NOT 1/N
        records = [_row(9, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 0.0)

    def test_mixed(self):
        # rank-1 (1.0) + rank-2 (0.5) + miss (0) → mean = 0.5
        records = [_row(1, [1, 2, 3]), _row(2, [1, 2, 3]), _row(9, [1, 2, 3])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 0.5)

    def test_empty_prediction(self):
        records = [_row(1, [])]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 0.0)

    def test_scalar_fallback_hit(self):
        records = [_scalar_row(1, 1)]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 1.0)

    def test_scalar_fallback_miss(self):
        records = [_scalar_row(2, 1)]
        self.assertAlmostEqual(eval_next_action.mean_reciprocal_rank(records), 0.0)


# ---------------------------------------------------------------------------
# action_label_accuracy (ActionAcc)
# ---------------------------------------------------------------------------

class TestActionLabelAccuracy(unittest.TestCase):
    def test_correct_case_insensitive(self):
        records = [_row(0, [0], gt_action="CLICK", pred_action="click")]
        self.assertAlmostEqual(eval_next_action.action_label_accuracy(records), 1.0)

    def test_wrong(self):
        records = [_row(0, [0], gt_action="CLICK", pred_action="TYPE")]
        self.assertAlmostEqual(eval_next_action.action_label_accuracy(records), 0.0)

    def test_parse_failure_counts_as_wrong(self):
        records = [
            _row(0, [0], gt_action="CLICK", pred_action="CLICK"),
            _row(1, [], gt_action="TYPE", pred_action=None),
        ]
        self.assertAlmostEqual(eval_next_action.action_label_accuracy(records), 0.5)

    def test_missing_gt_action_skipped(self):
        records = [
            _row(0, [1]),  # no gt_action → skipped, not in denominator
            _row(1, [1], gt_action="CLICK", pred_action="CLICK"),
        ]
        self.assertAlmostEqual(eval_next_action.action_label_accuracy(records), 1.0)

    def test_no_records(self):
        self.assertAlmostEqual(eval_next_action.action_label_accuracy([]), 0.0)


# ---------------------------------------------------------------------------
# parse_failure_rate
# ---------------------------------------------------------------------------

class TestParseFailureRate(unittest.TestCase):
    def test_no_failures(self):
        records = [_row(0, [1, 2]), _row(1, [0])]
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 0.0)

    def test_all_failures(self):
        records = [_row(0, []), _row(1, [])]
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 1.0)

    def test_partial(self):
        records = [_row(0, [1]), _row(1, []), _row(2, [])]
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 2 / 3)

    def test_scalar_none_is_failure(self):
        records = [_scalar_row(0, None)]
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 1.0)

    def test_scalar_non_none_is_success(self):
        records = [_scalar_row(0, 0)]
        self.assertAlmostEqual(eval_next_action.parse_failure_rate(records), 0.0)

    def test_empty(self):
        self.assertAlmostEqual(eval_next_action.parse_failure_rate([]), 0.0)


# ---------------------------------------------------------------------------
# task_success_rate
# ---------------------------------------------------------------------------

class TestTaskSuccessRate(unittest.TestCase):
    def test_single_step_correct(self):
        records = [_row(1, [1, 2, 3], annotation_id="t1")]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 1.0)

    def test_single_step_wrong(self):
        records = [_row(1, [2, 3, 4], annotation_id="t1")]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 0.0)

    def test_multi_step_all_correct(self):
        records = [
            _row(1, [1, 2, 3], annotation_id="t1"),
            _row(2, [2, 1, 0], annotation_id="t1"),
        ]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 1.0)

    def test_multi_step_one_wrong_fails_task(self):
        records = [
            _row(1, [1, 2, 3], annotation_id="t1"),
            _row(2, [0, 1, 3], annotation_id="t1"),  # top-1 wrong
        ]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 0.0)

    def test_multiple_tasks_partial(self):
        # t1 succeeds (1 step correct), t2 fails (second step wrong)
        records = [
            _row(1, [1, 2], annotation_id="t1"),
            _row(0, [0], annotation_id="t2"),
            _row(3, [9], annotation_id="t2"),
        ]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 0.5)

    def test_parse_failure_fails_task(self):
        records = [
            _row(1, [1], annotation_id="t1"),
            _row(2, [], annotation_id="t1"),  # parse failure → top-1 missing
        ]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 0.0)

    def test_scalar_fallback(self):
        records = [
            _scalar_row(1, 1, annotation_id="t1"),
            _scalar_row(2, 2, annotation_id="t1"),
        ]
        self.assertAlmostEqual(eval_next_action.task_success_rate(records), 1.0)

    def test_no_records(self):
        self.assertAlmostEqual(eval_next_action.task_success_rate([]), 0.0)


# ---------------------------------------------------------------------------
# step_accuracy (ElemAcc AND ActionAcc both correct)
# ---------------------------------------------------------------------------

class TestStepAccuracy(unittest.TestCase):
    def test_both_correct(self):
        records = [_row(1, [1, 2], gt_action="CLICK", pred_action="CLICK")]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 1.0)

    def test_elem_wrong(self):
        records = [_row(1, [2, 3], gt_action="CLICK", pred_action="CLICK")]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 0.0)

    def test_action_wrong(self):
        records = [_row(1, [1, 2], gt_action="CLICK", pred_action="TYPE")]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 0.0)

    def test_both_wrong(self):
        records = [_row(1, [2, 3], gt_action="CLICK", pred_action="TYPE")]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 0.0)

    def test_partial(self):
        records = [
            _row(1, [1, 2], gt_action="CLICK", pred_action="CLICK"),  # correct
            _row(2, [0, 1], gt_action="TYPE", pred_action="TYPE"),     # wrong elem
        ]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 0.5)

    def test_scalar_fallback(self):
        records = [_scalar_row(1, 1, gt_action="CLICK", pred_action="CLICK")]
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 1.0)

    def test_missing_gt_skipped(self):
        records = [_row(1, [2])]  # no gt_action → denominator stays 0
        self.assertAlmostEqual(eval_next_action.step_accuracy(records), 0.0)


# ---------------------------------------------------------------------------
# per_action_prf
# ---------------------------------------------------------------------------

class TestPerActionPrf(unittest.TestCase):
    def test_perfect_click(self):
        records = [
            _row(0, [0], gt_action="CLICK", pred_action="CLICK"),
            _row(1, [1], gt_action="CLICK", pred_action="CLICK"),
        ]
        prf = eval_next_action.per_action_prf(records)
        self.assertAlmostEqual(prf["CLICK"]["precision"], 1.0)
        self.assertAlmostEqual(prf["CLICK"]["recall"], 1.0)
        self.assertAlmostEqual(prf["CLICK"]["f1"], 1.0)
        self.assertEqual(prf["CLICK"]["support"], 2)

    def test_false_positive(self):
        records = [_row(0, [0], gt_action="TYPE", pred_action="CLICK")]
        prf = eval_next_action.per_action_prf(records)
        self.assertAlmostEqual(prf["CLICK"]["precision"], 0.0)
        self.assertAlmostEqual(prf["CLICK"]["recall"], 0.0)

    def test_false_negative(self):
        records = [_row(0, [0], gt_action="CLICK", pred_action="TYPE")]
        prf = eval_next_action.per_action_prf(records)
        self.assertAlmostEqual(prf["CLICK"]["recall"], 0.0)
        self.assertEqual(prf["CLICK"]["support"], 1)


# ---------------------------------------------------------------------------
# evaluate_file (integration)
# ---------------------------------------------------------------------------

class TestEvaluateFile(unittest.TestCase):
    def _write_jsonl(self, tmp: str, lines: list) -> Path:
        path = Path(tmp) / "preds.jsonl"
        with path.open("w", encoding="utf8") as f:
            for obj in lines:
                f.write((json.dumps(obj) if isinstance(obj, dict) else obj) + "\n")
        return path

    def test_basic_metrics_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "split": "test_task",
                "baseline": "intern_image_allinputs_axtree",
                "annotation_id": "ann1",
                "gt_action_index": 2,
                "pred_action_indices": [2, 1, 0],
                "gt_action": "CLICK",
                "pred_action": "CLICK",
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertEqual(m["num_lines"], 1)
            self.assertEqual(m["num_parsed"], 1)
            self.assertAlmostEqual(m["ElemAcc"], 1.0)
            self.assertAlmostEqual(m["Top3Elem"], 1.0)
            self.assertAlmostEqual(m["MRR"], 1.0)
            self.assertAlmostEqual(m["ActionAcc"], 1.0)
            self.assertAlmostEqual(m["TaskSuccess"], 1.0)
            self.assertAlmostEqual(m["StepAcc"], 1.0)

    def test_bad_json_lines_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [
                {"gt_action_index": 0, "pred_action_indices": [0], "annotation_id": "a"},
                "{bad json}",
            ])
            m = eval_next_action.evaluate_file(path)
            self.assertEqual(m["num_lines"], 2)
            self.assertEqual(m["num_parsed"], 1)
            self.assertGreater(m["json_parse_failure_rate"], 0)

    def test_baselines_and_splits_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "split": "test_task",
                "baseline": "intern_image_allinputs_axtree",
                "gt_action_index": 0,
                "pred_action_indices": [0],
                "annotation_id": "a",
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertIn("intern_image_allinputs_axtree", m["baselines"])
            self.assertIn("test_task", m["splits"])

    def test_wrong_prediction_scores_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "annotation_id": "a",
                "gt_action_index": 1,
                "pred_action_indices": [9, 8, 7],
                "gt_action": "CLICK",
                "pred_action": "TYPE",
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertAlmostEqual(m["ElemAcc"], 0.0)
            self.assertAlmostEqual(m["Top3Elem"], 0.0)
            self.assertAlmostEqual(m["MRR"], 0.0)
            self.assertAlmostEqual(m["ActionAcc"], 0.0)

    def test_top3_hit_not_top1(self):
        """gt at rank-2: Top3Elem=1, ElemAcc=0, MRR=0.5."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "annotation_id": "a",
                "gt_action_index": 2,
                "pred_action_indices": [1, 2, 3],
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertAlmostEqual(m["ElemAcc"], 0.0)
            self.assertAlmostEqual(m["Top3Elem"], 1.0)
            self.assertAlmostEqual(m["MRR"], 0.5)

    def test_backwards_compat_scalar_field(self):
        """evaluate_file must work with old scalar pred_action_index records."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "annotation_id": "a",
                "gt_action_index": 3,
                "pred_action_index": 3,
                "gt_action": "CLICK",
                "pred_action": "CLICK",
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertAlmostEqual(m["ElemAcc"], 1.0)
            self.assertAlmostEqual(m["Top3Elem"], 1.0)

    def test_gt_action_as_mind2web_json_string(self):
        """gt_action stored as the raw Mind2Web JSON string is correctly resolved."""
        # This matches the actual output before normalize_gt_operation was fixed —
        # HuggingFace serialises operation dicts as JSON strings, so gt_action ends
        # up as '{"ORIGINAL_OP": "CLICK", "VALUE": "", "OP": "CLICK"}' instead of
        # 'CLICK'. ActionAcc and StepAcc must still be > 0 in this case.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_jsonl(tmp, [{
                "annotation_id": "a",
                "gt_action_index": 2,
                "pred_action_indices": [2, 1, 0],
                "gt_action": '{"ORIGINAL_OP": "CLICK", "VALUE": "", "OP": "CLICK"}',
                "pred_action": "CLICK",
            }])
            m = eval_next_action.evaluate_file(path)
            self.assertAlmostEqual(m["ActionAcc"], 1.0)
            self.assertAlmostEqual(m["StepAcc"], 1.0)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate(unittest.TestCase):
    def test_empty(self):
        agg = eval_next_action.aggregate([])
        self.assertEqual(agg["num_files"], 0)
        self.assertAlmostEqual(agg["macro_ElemAcc"], 0.0)

    def test_single_file(self):
        m = {"ElemAcc": 0.8, "ActionAcc": 0.7, "Top3Elem": 0.9,
             "MRR": 0.75, "TaskSuccess": 0.5, "StepAcc": 0.6,
             "prediction_parse_failure_rate": 0.1}
        agg = eval_next_action.aggregate([m])
        self.assertAlmostEqual(agg["macro_ElemAcc"], 0.8)
        self.assertAlmostEqual(agg["macro_MRR"], 0.75)

    def test_macro_average(self):
        m1 = {"ElemAcc": 0.6, "ActionAcc": 0.6, "Top3Elem": 0.8,
              "MRR": 0.6, "TaskSuccess": 0.4, "StepAcc": 0.5,
              "prediction_parse_failure_rate": 0.0}
        m2 = {"ElemAcc": 0.8, "ActionAcc": 0.8, "Top3Elem": 1.0,
              "MRR": 0.8, "TaskSuccess": 0.6, "StepAcc": 0.7,
              "prediction_parse_failure_rate": 0.2}
        agg = eval_next_action.aggregate([m1, m2])
        self.assertAlmostEqual(agg["macro_ElemAcc"], 0.7)
        self.assertEqual(agg["num_files"], 2)


if __name__ == "__main__":
    unittest.main()
