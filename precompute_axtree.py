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
from pathlib import Path

from datasets import load_from_disk
from playwright.sync_api import sync_playwright
from tqdm import tqdm

VALID_SPLITS = ["test_task", "test_website", "test_domain"]

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

# Resource types that are safe to block when rendering static HTML.
# We must keep "stylesheet" and "script" so Chrome can compute ARIA roles
# (CSS decides display/visibility; JS sets aria-* attributes).
_BLOCK_RESOURCE_TYPES = {
    "image", "media", "font", "websocket", "manifest", "other",
}


def _block_unnecessary_resources(route):
    if route.request.resource_type in _BLOCK_RESOURCE_TYPES:
        route.abort()
    else:
        route.abort()  # fallback: also abort fetch/xhr to real servers


def build_page(playwright):
    browser = playwright.chromium.launch(headless=True)
    # Keep JS enabled: Chrome's accessibility engine needs it to resolve
    # ARIA roles, aria-* attributes, and CSS-driven visibility.
    context = browser.new_context(java_script_enabled=True)
    # Block images/media/fonts for speed, but allow inline styles/scripts
    # that are already embedded in the HTML string.
    context.route("**/*", lambda route: route.abort())
    page = context.new_page()
    return browser, context, page

def _field_value(field) -> str:
    """Extract string value from a CDP AX property dict or plain string."""
    if isinstance(field, dict):
        return field.get("value", "")
    return str(field) if field else ""


def _cdp_nodes_to_tree(nodes: list) -> dict:
    """Convert the flat CDP Accessibility.getFullAXTree node list to a nested
    dict that matches the old page.accessibility.snapshot() format so that
    flatten_axtree() continues to work without modification.

    Chrome marks most non-semantic DOM nodes as ``ignored=True``; including
    them would flood the tree with hundreds of empty {role:'', name:''} entries.
    We skip them here, mirroring what page.accessibility.snapshot() used to do
    internally and what BrowserGym's extract_merged_axtree does.
    """
    if not nodes:
        return {}

    id_map = {n["nodeId"]: n for n in nodes if "nodeId" in n}

    def _build(node) -> dict | None:
        # Skip nodes Chrome has marked as not contributing to accessibility.
        if node.get("ignored", False):
            return None
        role = _field_value(node.get("role", {}))
        name = _field_value(node.get("name", {}))
        value = _field_value(node.get("value", {})) or None
        # Recurse into children, dropping ignored subtrees.
        child_ids = node.get("childIds", [])
        children = [
            child
            for cid in child_ids
            if cid in id_map
            for child in [_build(id_map[cid])]
            if child is not None
        ]
        result: dict = {"role": role, "name": name}
        if value:
            result["value"] = value
        if children:
            result["children"] = children
        return result

    # Prefer the WebArea root (the document root in the AX tree); otherwise
    # fall back to the first non-ignored node.
    root = next(
        (n for n in nodes if _field_value(n.get("role", {})) == "WebArea"),
        next((n for n in nodes if not n.get("ignored", False)), None),
    )
    if root is None:
        return {}
    return _build(root) or {}


def html_to_axtree(page, html: str) -> dict:
    """Render *cleaned* HTML in a headless browser and extract the AXTree.

    Input should be ``cleaned_html`` (the OSU-preprocessed field), not
    ``raw_html``.  Reasons:
    * ``pos_candidates`` backend_node_ids are only guaranteed to survive into
      ``cleaned_html`` after OSU's preprocessing pipeline.
    * ``raw_html`` contains tracking pixels, hidden advertisement divs, and
      large inlined resources that bloat the tree with useless nodes.
    * All Multimodal-Mind2Web baselines (SeeAct, WebAgent, BrowserGym
      WebArena) feed the AX tree from the cleaned/simplified DOM.

    ``sanitize_html`` is still applied to strip any residual <script> tags
    and <link rel> references that would trigger blocked-network errors.
    """
    html = sanitize_html(html)
    # Use "load" (not "domcontentloaded") so that inline scripts finish
    # updating ARIA attributes before we snapshot the accessibility tree.
    page.set_content(html, wait_until="load")

    # page.accessibility was removed in Playwright ≥ 1.40.
    # Use a CDP session to call Accessibility.getFullAXTree instead.
    # This is the same approach used by BrowserGym's extract_merged_axtree.
    try:
        cdp = page.context.new_cdp_session(page)
        result = cdp.send("Accessibility.getFullAXTree", {})
        cdp.detach()
        return _cdp_nodes_to_tree(result.get("nodes", []))
    except Exception:
        pass

    # Legacy fallback for older Playwright versions that still expose the API.
    if hasattr(page, "accessibility") and page.accessibility is not None:
        snap = page.accessibility.snapshot(interesting_only=False)
        return snap or {}

    return {}

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Precompute AXTree for Mind2Web splits")
    ap.add_argument("--split", default="test_website", choices=VALID_SPLITS)
    ap.add_argument("--data_dir", default="data/mind2web",
                    help="Directory produced by download_mind2web.py (contains one sub-dir per split)")
    ap.add_argument("--out_dir", default="data/mind2web_axtree")
    ap.add_argument(
        "--html_field", default="cleaned_html", choices=["raw_html", "cleaned_html"],
        help="HTML field to render. Default is 'cleaned_html' (OSU-preprocessed) — "
             "pos_candidates backend_node_ids are only guaranteed to exist there. "
             "Use 'raw_html' only for debugging.",
    )
    args = ap.parse_args()

    split_path = Path(args.data_dir) / args.split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split not found: {split_path}\n"
            "Run download_mind2web.py first."
        )

    save_path = Path(args.out_dir) / args.split
    if save_path.exists():
        print(f"AXTree split '{args.split}' already on disk at {save_path} — skipping.")
        return

    print(f"Loading split '{args.split}' from disk: {split_path}")
    ds = load_from_disk(str(split_path))

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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path))
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()