import unittest

import torch

from inference import intern_axtree_ablations as intern


class InternAXTreeAblationsTests(unittest.TestCase):
    def test_resolve_dtype(self):
        self.assertEqual(intern.resolve_dtype("bf16"), torch.bfloat16)
        self.assertEqual(intern.resolve_dtype("fp16"), torch.float16)
        self.assertEqual(intern.resolve_dtype("fp32"), torch.float32)

    def test_build_quantization_config_none(self):
        config = intern.build_quantization_config("none", torch.float16)
        self.assertIsNone(config)

    def test_build_quantization_config_4bit(self):
        config = intern.build_quantization_config("4bit", torch.float16)
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)

    def test_build_quantization_config_8bit(self):
        config = intern.build_quantization_config("8bit", torch.float16)
        self.assertTrue(config.load_in_8bit)

    def test_parse_predicted_index(self):
        self.assertEqual(intern.parse_predicted_index("FINAL_ANSWER: 2", 5), 2)
        self.assertEqual(intern.parse_predicted_index("Reasoning... then 3", 5), 3)
        self.assertIsNone(intern.parse_predicted_index("FINAL_ANSWER: 10", 5))
        self.assertIsNone(intern.parse_predicted_index("no digits", 5))

    def test_extract_and_normalize_operations(self):
        self.assertEqual(intern.extract_operation("CLICK [id=1]"), "CLICK")
        self.assertEqual(intern.normalize_gt_operation({"op": "type"}), "TYPE")
        self.assertEqual(intern.normalize_gt_operation(" scroll "), "SCROLL")


if __name__ == "__main__":
    unittest.main()
