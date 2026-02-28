import unittest

from baselines.models import multimodal_qwen


class MultimodalQwenTests(unittest.TestCase):
    def test_build_quantization_kwargs_none(self):
        self.assertIsNone(multimodal_qwen.build_quantization_kwargs("none"))

    def test_build_quantization_kwargs_4bit(self):
        kwargs = multimodal_qwen.build_quantization_kwargs("4bit")
        self.assertTrue(kwargs["load_in_4bit"])
        self.assertEqual(kwargs["bnb_4bit_quant_type"], "nf4")

    def test_build_quantization_kwargs_8bit(self):
        kwargs = multimodal_qwen.build_quantization_kwargs("8bit")
        self.assertTrue(kwargs["load_in_8bit"])


if __name__ == "__main__":
    unittest.main()
