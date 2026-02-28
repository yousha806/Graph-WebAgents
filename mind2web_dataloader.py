"""General-purpose dataset and dataloader for Multimodal Mind2Web.

Dataset: https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web

Available splits: train, test_task, test_website, test_domain
(There is NO validation split.)

Columns per row (from HF):
    annotation_id, action_uid, website, domain, subdomain, confirmed_task,
    screenshot (Image), raw_html, cleaned_html, operation (dict),
    pos_candidates, neg_candidates, action_reprs, target_action_index (str),
    target_action_reprs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

DATASET_NAME = "osunlp/Multimodal-Mind2Web"
VALID_SPLITS = ["train", "test_task", "test_website", "test_domain"]


class MultimodalMind2WebDataset(Dataset):
    """PyTorch Dataset wrapper around Multimodal Mind2Web.

    Args:
        split: One of train, test_task, test_website, test_domain.
        local_dir: If provided, load from a local directory saved by
                   ``download_mind2web.py`` instead of streaming from HF.
        max_html_chars: Optional truncation limit for cleaned_html.
        html_field: Which HTML column to expose as ``html`` ("cleaned_html"
                    or "raw_html").
    """

    def __init__(
        self,
        split: str,
        local_dir: Optional[str] = None,
        max_html_chars: Optional[int] = None,
        html_field: str = "cleaned_html",
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        if local_dir is None:
            raise ValueError(
                "local_dir is required. Run download_mind2web.py first, then pass "
                "the output directory as local_dir."
            )
        split_path = Path(local_dir) / split
        if not split_path.exists():
            raise FileNotFoundError(f"Local split not found: {split_path}")
        self.dataset = load_from_disk(str(split_path))

        self.split = split
        self.max_html_chars = max_html_chars
        self.html_field = html_field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]

        html = row.get(self.html_field, "") or ""
        if self.max_html_chars is not None:
            html = html[: self.max_html_chars]

        return {
            # identifiers
            "annotation_id": row["annotation_id"],
            "action_uid": row["action_uid"],
            # task / site metadata
            "website": row.get("website", ""),
            "domain": row.get("domain", ""),
            "confirmed_task": row["confirmed_task"],
            # multimodal inputs
            "html": html,
            "screenshot": row["screenshot"],
            # action labels
            "operation": row["operation"],
            "action_reprs": row["action_reprs"],
            "target_action_index": row["target_action_index"],
            "target_action_reprs": row.get("target_action_reprs", ""),
            # candidates
            "pos_candidates": row.get("pos_candidates", []),
            "neg_candidates": row.get("neg_candidates", []),
        }


# Keys that the collate function aggregates into lists.
_COLLATE_KEYS = [
    "annotation_id",
    "action_uid",
    "website",
    "domain",
    "confirmed_task",
    "html",
    "screenshot",
    "operation",
    "action_reprs",
    "target_action_index",
    "target_action_reprs",
    "pos_candidates",
    "neg_candidates",
]


def mind2web_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Stack a list of sample dicts into a dict of lists."""
    return {key: [item[key] for item in batch] for key in _COLLATE_KEYS}


def create_mind2web_dataloader(
    split: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_html_chars: Optional[int] = None,
    html_field: str = "cleaned_html",
    local_dir: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for a single Mind2Web split."""
    dataset = MultimodalMind2WebDataset(
        split=split,
        local_dir=local_dir,
        max_html_chars=max_html_chars,
        html_field=html_field,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mind2web_collate_fn,
    )


def create_all_dataloaders(
    splits: Optional[List[str]] = None,
    batch_size: int = 1,
    train_shuffle: bool = True,
    eval_shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_html_chars: Optional[int] = None,
    html_field: str = "cleaned_html",
    local_dir: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for one or more Mind2Web splits.

    Args:
        splits: List of split names. Defaults to all four splits.

    Returns:
        Dict mapping split name to its DataLoader.
    """
    if splits is None:
        splits = list(VALID_SPLITS)

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        loaders[split] = create_mind2web_dataloader(
            split=split,
            batch_size=batch_size,
            shuffle=train_shuffle if split == "train" else eval_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_html_chars=max_html_chars,
            html_field=html_field,
            local_dir=local_dir,
        )
    return loaders
