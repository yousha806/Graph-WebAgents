import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_mind2web


class _DummyDataset:
    def __init__(self, length=5):
        self._length = length
        self.saved_path = None

    def __len__(self):
        return self._length

    def save_to_disk(self, path):
        self.saved_path = path


class DownloadMind2WebTests(unittest.TestCase):
    def test_build_split_selector_none(self):
        self.assertEqual(download_mind2web._build_split_selector("train", None), "train")

    def test_build_split_selector_fraction(self):
        self.assertEqual(download_mind2web._build_split_selector("train", 0.01), "train[:1%]")
        self.assertEqual(download_mind2web._build_split_selector("train", 0.125), "train[:12.5%]")

    @patch("download_mind2web.load_dataset")
    def test_download_and_save_split_calls_dataset_api(self, mock_load_dataset):
        dummy = _DummyDataset(length=7)
        mock_load_dataset.return_value = dummy

        with tempfile.TemporaryDirectory() as tmp:
            split, count, out_path, selector = download_mind2web._download_and_save_split(
                split="test_website",
                subset_fraction=0.1,
                cache_dir="C:/tmp/cache",
                output_dir=tmp,
            )

            self.assertEqual(split, "test_website")
            self.assertEqual(count, 7)
            self.assertEqual(selector, "test_website[:10%]")
            self.assertEqual(out_path, str(Path(tmp) / "test_website"))
            self.assertEqual(dummy.saved_path, out_path)

            mock_load_dataset.assert_called_once_with(
                download_mind2web.DATASET_NAME,
                split="test_website[:10%]",
                cache_dir="C:/tmp/cache",
            )


if __name__ == "__main__":
    unittest.main()
