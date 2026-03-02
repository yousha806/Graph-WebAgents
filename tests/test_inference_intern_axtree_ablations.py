"""Tests for inference/intern_axtree_ablations.py.

Covers all pure-Python/CPU functions — model loading and GPU ops are excluded.
"""
import io
import unittest

import torch
from PIL import Image

from inference import intern_axtree_ablations as intern


# ---------------------------------------------------------------------------
# resolve_dtype
# ---------------------------------------------------------------------------

class TestResolveDtype(unittest.TestCase):
    def test_bf16(self):
        self.assertEqual(intern.resolve_dtype("bf16"), torch.bfloat16)

    def test_fp16(self):
        self.assertEqual(intern.resolve_dtype("fp16"), torch.float16)

    def test_fp32(self):
        self.assertEqual(intern.resolve_dtype("fp32"), torch.float32)


# ---------------------------------------------------------------------------
# build_quantization_config
# ---------------------------------------------------------------------------

class TestBuildQuantizationConfig(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(intern.build_quantization_config("none", torch.float16))

    def test_4bit(self):
        cfg = intern.build_quantization_config("4bit", torch.float16)
        self.assertTrue(cfg.load_in_4bit)
        self.assertEqual(cfg.bnb_4bit_compute_dtype, torch.float16)

    def test_8bit(self):
        cfg = intern.build_quantization_config("8bit", torch.float16)
        self.assertTrue(cfg.load_in_8bit)

    def test_invalid_raises(self):
        with self.assertRaises((ValueError, KeyError, Exception)):
            intern.build_quantization_config("2bit", torch.float16)


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------

class TestTruncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(intern.truncate("hello", 100), "hello")

    def test_long_text_clipped(self):
        result = intern.truncate("a" * 200, 50)
        self.assertEqual(len(result), 50)

    def test_empty_string(self):
        self.assertEqual(intern.truncate("", 100), "")

    def test_none_treated_as_empty(self):
        # truncate does `text = text or ""`
        self.assertEqual(intern.truncate(None, 100), "")


# ---------------------------------------------------------------------------
# candidate_lines
# ---------------------------------------------------------------------------

class TestCandidateLines(unittest.TestCase):
    def test_indices_prefix(self):
        lines = intern.candidate_lines(["CLICK button", "TYPE input"])
        self.assertIn("0: CLICK button", lines)
        self.assertIn("1: TYPE input", lines)

    def test_empty_list(self):
        self.assertEqual(intern.candidate_lines([]), "")

    def test_single_candidate(self):
        lines = intern.candidate_lines(["SELECT dropdown"])
        self.assertEqual(lines, "0: SELECT dropdown")


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):
    def _row(self, task="Click Submit", axtree="[button] \"Submit\"",
             actions=None):
        return {
            "confirmed_task": task,
            "axtree_text": axtree,
            "action_reprs": actions or ["CLICK button"],
        }

    def test_contains_image_token(self):
        prompt = intern.build_prompt(self._row(), use_cot=False, max_axtree_chars=10000)
        self.assertIn("<image>", prompt)

    def test_contains_task(self):
        prompt = intern.build_prompt(self._row(task="Find login"), use_cot=False, max_axtree_chars=10000)
        self.assertIn("Find login", prompt)

    def test_contains_axtree(self):
        prompt = intern.build_prompt(self._row(axtree="[button] \"Go\""), use_cot=False, max_axtree_chars=10000)
        self.assertIn('[button] "Go"', prompt)

    def test_contains_candidates(self):
        prompt = intern.build_prompt(self._row(actions=["CLICK a", "TYPE b"]),
                                     use_cot=False, max_axtree_chars=10000)
        self.assertIn("0: CLICK a", prompt)
        self.assertIn("1: TYPE b", prompt)

    def test_cot_schema_included(self):
        prompt = intern.build_prompt(self._row(), use_cot=True, max_axtree_chars=10000)
        self.assertIn("reasoning", prompt)

    def test_non_cot_schema_no_reasoning_key(self):
        prompt = intern.build_prompt(self._row(), use_cot=False, max_axtree_chars=10000)
        # The non-CoT schema should not ask for "reasoning"
        self.assertNotIn('"reasoning"', prompt)

    def test_axtree_truncated_to_max_chars(self):
        long_axtree = "x" * 5000
        prompt = intern.build_prompt(self._row(axtree=long_axtree),
                                     use_cot=False, max_axtree_chars=100)
        # The axtree section must not exceed max_chars
        self.assertIn("x" * 100, prompt)
        self.assertNotIn("x" * 101, prompt)


# ---------------------------------------------------------------------------
# parse_model_output — JSON path
# ---------------------------------------------------------------------------

class TestParseModelOutputJson(unittest.TestCase):
    def test_valid_top3_json(self):
        raw = '{"top3_action_indices": [2, 0, 1], "action_type": "CLICK", "target_element": "btn"}'
        result = intern.parse_model_output(raw, num_candidates=5)
        self.assertEqual(result["pred_action_indices"], [2, 0, 1])
        self.assertEqual(result["action_type"], "CLICK")
        self.assertEqual(result["target_element"], "btn")
        self.assertIsNone(result["parse_error"])

    def test_valid_cot_json(self):
        raw = ('{"reasoning": "It is the submit button", '
               '"top3_action_indices": [1], "action_type": "CLICK", "target_element": "Submit"}')
        result = intern.parse_model_output(raw, num_candidates=3)
        self.assertEqual(result["pred_action_indices"], [1])
        self.assertEqual(result["reasoning"], "It is the submit button")

    def test_out_of_range_indices_filtered(self):
        raw = '{"top3_action_indices": [0, 99, 2], "action_type": "CLICK", "target_element": "x"}'
        result = intern.parse_model_output(raw, num_candidates=3)
        # 99 is out of range for 3 candidates
        self.assertNotIn(99, result["pred_action_indices"])
        self.assertIn(0, result["pred_action_indices"])
        self.assertIn(2, result["pred_action_indices"])

    def test_markdown_fences_stripped(self):
        raw = '```json\n{"top3_action_indices": [1, 0], "action_type": "TYPE", "target_element": "box"}\n```'
        result = intern.parse_model_output(raw, num_candidates=5)
        self.assertEqual(result["pred_action_indices"], [1, 0])

    def test_legacy_scalar_index_accepted(self):
        raw = '{"top3_action_indices": 3, "action_type": "CLICK", "target_element": "btn"}'
        result = intern.parse_model_output(raw, num_candidates=5)
        self.assertEqual(result["pred_action_indices"], [3])

    def test_json_capped_at_3_indices(self):
        raw = '{"top3_action_indices": [0, 1, 2, 3, 4], "action_type": "CLICK", "target_element": "x"}'
        result = intern.parse_model_output(raw, num_candidates=10)
        self.assertLessEqual(len(result["pred_action_indices"]), 3)

    def test_empty_output_returns_empty_indices(self):
        result = intern.parse_model_output("", num_candidates=5)
        self.assertEqual(result["pred_action_indices"], [])
        self.assertIsNotNone(result["parse_error"])

    def test_all_out_of_range_sets_parse_error(self):
        raw = '{"top3_action_indices": [99, 100], "action_type": "CLICK", "target_element": "x"}'
        result = intern.parse_model_output(raw, num_candidates=3)
        self.assertEqual(result["pred_action_indices"], [])
        self.assertIsNotNone(result["parse_error"])

    def test_action_type_uppercased(self):
        raw = '{"top3_action_indices": [0], "action_type": "click", "target_element": "btn"}'
        result = intern.parse_model_output(raw, num_candidates=3)
        self.assertEqual(result["action_type"], "CLICK")


# ---------------------------------------------------------------------------
# parse_model_output — regex fallback
# ---------------------------------------------------------------------------

class TestParseModelOutputRegexFallback(unittest.TestCase):
    def test_plain_integer_in_range(self):
        result = intern.parse_model_output("The answer is 2", num_candidates=5)
        self.assertIn(2, result["pred_action_indices"])
        self.assertIn("regex fallback", result["parse_error"])

    def test_multiple_integers_deduplicated_and_ordered(self):
        result = intern.parse_model_output("Candidates 1 and 1 and 3", num_candidates=5)
        indices = result["pred_action_indices"]
        # 1 should appear only once
        self.assertEqual(indices.count(1), 1)

    def test_no_valid_integer_returns_empty(self):
        result = intern.parse_model_output("no digits here at all", num_candidates=5)
        self.assertEqual(result["pred_action_indices"], [])
        self.assertIsNotNone(result["parse_error"])

    def test_negative_integer_ignored(self):
        result = intern.parse_model_output("index -1", num_candidates=5)
        self.assertNotIn(-1, result["pred_action_indices"])

    def test_regex_capped_at_3(self):
        result = intern.parse_model_output("0 1 2 3 4", num_candidates=10)
        self.assertLessEqual(len(result["pred_action_indices"]), 3)


# ---------------------------------------------------------------------------
# extract_operation / normalize_gt_operation
# ---------------------------------------------------------------------------

class TestExtractOperation(unittest.TestCase):
    def test_click(self):
        self.assertEqual(intern.extract_operation("CLICK [id=1]"), "CLICK")

    def test_type(self):
        self.assertEqual(intern.extract_operation("type input-box with text"), "TYPE")

    def test_leading_whitespace(self):
        self.assertEqual(intern.extract_operation("  select option"), "SELECT")

    def test_empty_string(self):
        self.assertIsNone(intern.extract_operation(""))

    def test_none(self):
        self.assertIsNone(intern.extract_operation(None))


class TestNormalizeGtOperation(unittest.TestCase):
    def test_string(self):
        self.assertEqual(intern.normalize_gt_operation("click"), "CLICK")
        self.assertEqual(intern.normalize_gt_operation(" scroll "), "SCROLL")

    def test_dict_op_key(self):
        self.assertEqual(intern.normalize_gt_operation({"op": "type"}), "TYPE")

    def test_dict_action_key(self):
        self.assertEqual(intern.normalize_gt_operation({"action": "select"}), "SELECT")

    def test_dict_no_known_key(self):
        self.assertIsNone(intern.normalize_gt_operation({"unknown": "x"}))

    def test_none(self):
        self.assertIsNone(intern.normalize_gt_operation(None))

    def test_empty_string(self):
        self.assertIsNone(intern.normalize_gt_operation(""))


# ---------------------------------------------------------------------------
# to_pil_image
# ---------------------------------------------------------------------------

class TestToPilImage(unittest.TestCase):
    def _make_png_bytes(self) -> bytes:
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_pil_image_returned_as_rgb(self):
        img = Image.new("RGBA", (5, 5))
        result = intern.to_pil_image(img)
        self.assertEqual(result.mode, "RGB")

    def test_dict_with_bytes(self):
        png = self._make_png_bytes()
        result = intern.to_pil_image({"bytes": png})
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")

    def test_unsupported_raises(self):
        with self.assertRaises((ValueError, Exception)):
            intern.to_pil_image(12345)


if __name__ == "__main__":
    unittest.main()
