"""Download Multimodal Mind2Web from Hugging Face and save it locally.

Available splits: train, test_task, test_website, test_domain
Dataset: https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web

Examples:
    python download_mind2web.py
    python download_mind2web.py --splits train test_task test_website test_domain
    python download_mind2web.py --subset_fraction 0.01 --output_dir data/mind2web
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import List

from datasets import load_dataset, load_from_disk

DATASET_NAME = "osunlp/Multimodal-Mind2Web"
VALID_SPLITS = ["train", "test_task", "test_website", "test_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Multimodal Mind2Web dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_website", "test_domain", "test_task"],
        choices=VALID_SPLITS,
        help="Dataset splits to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mind2web",
        help="Local directory to store downloaded splits via save_to_disk",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory (use a drive with free space)",
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=None,
        help="If set, keep this fraction (0-1] of each split",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for split download/save",
    )
    return parser.parse_args()


def _build_split_selector(split: str, subset_fraction: float | None) -> str:
    """Build a HF split selector string, e.g. 'train[:1%]'."""
    if subset_fraction is None or subset_fraction >= 1:
        return split
    percentage = f"{subset_fraction * 100:.6f}".rstrip("0").rstrip(".")
    return f"{split}[:{percentage}%]"


def _download_and_save_split(
    split: str,
    subset_fraction: float | None,
    cache_dir: str | None,
    output_dir: str,
) -> tuple[str, int, str, str]:
    """Worker function: load one split from HF and save to disk."""
    split_path = Path(output_dir) / split
    if split_path.exists():
        existing = load_from_disk(str(split_path))
        print(f"  {split} already on disk ({len(existing)} records) — skipping download.")
        return split, len(existing), str(split_path), split

    split_selector = _build_split_selector(split, subset_fraction)
    split_dataset = load_dataset(
        DATASET_NAME,
        split=split_selector,
        cache_dir=cache_dir,
    )
    split_dataset.save_to_disk(str(split_path))
    return split, len(split_dataset), str(split_path), split_selector


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.subset_fraction is not None and not (0 < args.subset_fraction <= 1):
        raise ValueError("--subset_fraction must be in the range (0, 1].")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1.")

    print(f"Downloading {DATASET_NAME} — splits: {args.splits}")

    failures: List[str] = []
    with ProcessPoolExecutor(max_workers=min(args.num_workers, len(args.splits))) as executor:
        future_to_split = {
            executor.submit(
                _download_and_save_split,
                split,
                args.subset_fraction,
                args.cache_dir,
                str(output_root),
            ): split
            for split in args.splits
        }

        for future in as_completed(future_to_split):
            split = future_to_split[future]
            try:
                saved_split, record_count, split_path, split_selector = future.result()
                print(f"  {split_selector} → {record_count} records → {split_path}")
            except Exception as exc:
                failures.append(f"{split}: {exc}")
                print(f"  FAILED '{split}': {exc}")

    if failures:
        raise RuntimeError("Some splits failed:\n" + "\n".join(failures))

    print("Download complete.")


if __name__ == "__main__":
    main()
