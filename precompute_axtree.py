"""Precompute accessibility-tree (AXTree) representations for Multimodal Mind2Web.

Renders each row's HTML in a headless Chromium browser, extracts the AXTree
snapshot, and saves the enriched dataset to disk.

Requires: playwright  (install browsers once with `playwright install chromium`)

Example:
    python precompute_axtree.py --split test_website --out_dir data/mind2web_axtree
"""

import argparse
import json
import re

from datasets import load_dataset
from playwright.sync_api import sync_playwright
from tqdm import tqdm

DATASET_NAME = "osunlp/Multimodal-Mind2Web"
VALID_SPLITS = ["train", "test_task", "test_website", "test_domain"]

# -----------------------------
# AXTree extraction utilities
# -----------------------------

def sanitize_html(html: str) -> str:
    """Make rendering more deterministic and safer (no external loads / scripts)."""
    if not html:
        return ""
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.I | re.S)
    html = re.sub(r"<link\b[^>]*>", "", html, flags=re.I)
    return html

def flatten_axtree(snapshot: dict, max_nodes: int = 800, max_chars: int = 12000) -> str:
    """Turn AXTree JSON into compact, model-friendly text."""
    if not snapshot:
        return ""

    lines = []
    stack = [(snapshot, 0)]
    while stack and len(lines) < max_nodes:
        node, depth = stack.pop()
        role = node.get("role", "")
        name = (node.get("name") or "").strip()
        value = node.get("value")
        value_str = "" if value is None else f" value={str(value).strip()}"

        line = ("  " * depth) + f"{role}: {name}{value_str}".strip()
        lines.append(line)

        children = node.get("children") or []
        for c in reversed(children):
            stack.append((c, depth + 1))

        if sum(len(x) + 1 for x in lines) > max_chars:
            break

    text = "\n".join(lines)
    return text[:max_chars]

def build_page(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(java_script_enabled=False)  # deterministic-ish
    # Block any network activity (HTML may reference remote assets)
    context.route("**/*", lambda route: route.abort())
    page = context.new_page()
    return browser, context, page

def html_to_axtree(page, html: str) -> dict:
    html = sanitize_html(html)
    page.set_content(html, wait_until="domcontentloaded")
    # Keep full tree (interesting_only=False)
    snap = page.accessibility.snapshot(interesting_only=False)
    return snap or {}

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Precompute AXTree for Mind2Web splits")
    ap.add_argument("--split", default="train", choices=VALID_SPLITS)
    ap.add_argument("--out_dir", default="data/mind2web_axtree")
    ap.add_argument("--html_field", default="cleaned_html", choices=["raw_html", "cleaned_html"])
    args = ap.parse_args()

    print(f"Loading {DATASET_NAME} split='{args.split}'...")
    ds = load_dataset(DATASET_NAME, split=args.split)

    axtree_json_col = []
    axtree_text_col = []

    with sync_playwright() as p:
        browser, context, page = build_page(p)
        try:
            for ex in tqdm(ds, desc=f"AXTree [{args.split}]"):
                snap = html_to_axtree(page, ex.get(args.html_field, ""))
                axtree_json_col.append(json.dumps(snap, ensure_ascii=False))
                axtree_text_col.append(flatten_axtree(snap))
        finally:
            context.close()
            browser.close()

    ds = ds.add_column("axtree_json", axtree_json_col)
    ds = ds.add_column("axtree_text", axtree_text_col)

    save_path = f"{args.out_dir}/{args.split}"
    ds.save_to_disk(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()