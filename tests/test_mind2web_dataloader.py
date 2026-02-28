import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mind2web_dataloader


class _DummyHFDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class Mind2WebDataloaderTests(unittest.TestCase):
    def setUp(self):
        self.sample_row = {
            "annotation_id": "ann_1",
            "action_uid": "act_1",
            "website": "example",
            "domain": "example.com",
            "confirmed_task": "click submit",
            "cleaned_html": "<button>Submit</button>",
            "raw_html": "<html><button>Submit</button></html>",
            "screenshot": "img_obj_or_path",
            "operation": {"op": "CLICK", "value": ""},
            "action_reprs": ["0: click submit"],
            "target_action_index": "0",
            "target_action_reprs": "click submit",
            "pos_candidates": [{"backend_node_id": "1"}],
            "neg_candidates": [{"backend_node_id": "2"}],
        }

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            mind2web_dataloader.MultimodalMind2WebDataset(split="val")

    def test_missing_local_dir_raises(self):
        with self.assertRaises(ValueError):
            mind2web_dataloader.MultimodalMind2WebDataset(split="test_website")

    @patch("mind2web_dataloader.load_from_disk")
    def test_getitem_maps_expected_fields(self, mock_load_from_disk):
        mock_load_from_disk.return_value = _DummyHFDataset([self.sample_row])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test_website").mkdir()
            ds = mind2web_dataloader.MultimodalMind2WebDataset(
                split="test_website", local_dir=tmp
            )
        sample = ds[0]

        self.assertEqual(sample["annotation_id"], "ann_1")
        self.assertEqual(sample["action_uid"], "act_1")
        self.assertEqual(sample["confirmed_task"], "click submit")
        self.assertEqual(sample["target_action_index"], "0")
        self.assertEqual(sample["html"], "<button>Submit</button>")

    @patch("mind2web_dataloader.load_from_disk")
    def test_max_html_chars_applied(self, mock_load_from_disk):
        mock_load_from_disk.return_value = _DummyHFDataset([self.sample_row])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test_website").mkdir()
            ds = mind2web_dataloader.MultimodalMind2WebDataset(
                split="test_website", local_dir=tmp, max_html_chars=10, html_field="raw_html"
            )
        self.assertEqual(ds[0]["html"], "<html><but")

    def test_local_dir_missing_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp).mkdir(exist_ok=True)
            with self.assertRaises(FileNotFoundError):
                mind2web_dataloader.MultimodalMind2WebDataset(split="test_task", local_dir=tmp)

    def test_collate_fn(self):
        batch = [
            {
                "annotation_id": "a1",
                "action_uid": "u1",
                "website": "w",
                "domain": "d",
                "confirmed_task": "t",
                "html": "h",
                "screenshot": "s",
                "operation": {"op": "CLICK"},
                "action_reprs": ["r"],
                "target_action_index": "0",
                "target_action_reprs": "r",
                "pos_candidates": [],
                "neg_candidates": [],
            },
            {
                "annotation_id": "a2",
                "action_uid": "u2",
                "website": "w2",
                "domain": "d2",
                "confirmed_task": "t2",
                "html": "h2",
                "screenshot": "s2",
                "operation": {"op": "TYPE"},
                "action_reprs": ["r2"],
                "target_action_index": "1",
                "target_action_reprs": "r2",
                "pos_candidates": [],
                "neg_candidates": [],
            },
        ]
        collated = mind2web_dataloader.mind2web_collate_fn(batch)
        self.assertEqual(collated["annotation_id"], ["a1", "a2"])
        self.assertEqual(collated["target_action_index"], ["0", "1"])

    @patch("mind2web_dataloader.create_mind2web_dataloader")
    def test_create_all_dataloaders_respects_split_list(self, mock_create):
        mock_create.side_effect = lambda split, **kwargs: f"loader_{split}"
        loaders = mind2web_dataloader.create_all_dataloaders(splits=["train", "test_task"])

        self.assertEqual(loaders["train"], "loader_train")
        self.assertEqual(loaders["test_task"], "loader_test_task")
        self.assertEqual(set(loaders.keys()), {"train", "test_task"})


if __name__ == "__main__":
    unittest.main()
