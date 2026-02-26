import sys
import types
import unittest


if "playwright" not in sys.modules:
    playwright_module = types.ModuleType("playwright")
    playwright_sync_module = types.ModuleType("playwright.sync_api")

    def _dummy_sync_playwright():
        raise RuntimeError("dummy sync_playwright should not be called in these unit tests")

    playwright_sync_module.sync_playwright = _dummy_sync_playwright
    sys.modules["playwright"] = playwright_module
    sys.modules["playwright.sync_api"] = playwright_sync_module

import precompute_axtree


class _FakeAccessibility:
    def __init__(self, snapshot_value):
        self.snapshot_value = snapshot_value

    def snapshot(self, interesting_only=False):
        return self.snapshot_value


class _FakePage:
    def __init__(self, snapshot_value):
        self.last_html = None
        self.last_wait_until = None
        self.accessibility = _FakeAccessibility(snapshot_value)

    def set_content(self, html, wait_until):
        self.last_html = html
        self.last_wait_until = wait_until


class PrecomputeAXTreeTests(unittest.TestCase):
    def test_sanitize_html_removes_script_and_link(self):
        html = "<html><head><link href='x.css'></head><body><script>alert(1)</script><div>ok</div></body></html>"
        cleaned = precompute_axtree.sanitize_html(html)
        self.assertNotIn("<script", cleaned.lower())
        self.assertNotIn("<link", cleaned.lower())
        self.assertIn("<div>ok</div>", cleaned)

    def test_flatten_axtree_contains_roles(self):
        snapshot = {
            "role": "WebArea",
            "name": "Root",
            "children": [
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Query", "value": "abc"},
            ],
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn("WebArea: Root", text)
        self.assertIn("button: Submit", text)
        self.assertIn("textbox: Query value=abc", text)

    def test_html_to_axtree_sanitizes_before_snapshot(self):
        page = _FakePage(snapshot_value={"role": "WebArea"})
        out = precompute_axtree.html_to_axtree(page, "<script>x</script><div>hi</div>")

        self.assertEqual(out, {"role": "WebArea"})
        self.assertEqual(page.last_wait_until, "domcontentloaded")
        self.assertNotIn("<script", page.last_html.lower())
        self.assertIn("<div>hi</div>", page.last_html)


if __name__ == "__main__":
    unittest.main()
