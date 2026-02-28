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


class _FakeCdpSession:
    def __init__(self, nodes):
        self._nodes = nodes
        self.detached = False

    def send(self, method, params):
        if method == "Accessibility.getFullAXTree":
            return {"nodes": self._nodes}
        raise ValueError(f"Unknown CDP method: {method}")

    def detach(self):
        self.detached = True


class _FakeContext:
    def __init__(self, cdp_session):
        self._cdp_session = cdp_session

    def new_cdp_session(self, page):
        return self._cdp_session


class _FakePage:
    def __init__(self, nodes):
        self.last_html = None
        self.last_wait_until = None
        _session = _FakeCdpSession(nodes)
        self.context = _FakeContext(_session)

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
        nodes = [
            {
                "nodeId": "1",
                "role": {"value": "WebArea"},
                "name": {"value": "Root"},
                "childIds": [],
            }
        ]
        page = _FakePage(nodes)
        out = precompute_axtree.html_to_axtree(page, "<script>x</script><div>hi</div>")

        self.assertEqual(out, {"role": "WebArea", "name": "Root"})
        # Must use 'load' (not 'domcontentloaded') so inline scripts finish
        # updating ARIA attributes before we snapshot the AX tree.
        self.assertEqual(page.last_wait_until, "load")
        self.assertNotIn("<script", page.last_html.lower())
        self.assertIn("<div>hi</div>", page.last_html)

    def test_cdp_nodes_to_tree_filters_ignored_nodes(self):
        """Chrome marks non-semantic nodes ignored=True; they must be dropped."""
        nodes = [
            {
                "nodeId": "1",
                "role": {"value": "WebArea"},
                "name": {"value": "Root"},
                "ignored": False,
                "childIds": ["2", "3"],
            },
            {
                "nodeId": "2",
                "role": {"value": ""},
                "name": {"value": ""},
                "ignored": True,  # Should be dropped
                "childIds": [],
            },
            {
                "nodeId": "3",
                "role": {"value": "button"},
                "name": {"value": "Submit"},
                "ignored": False,
                "childIds": [],
            },
        ]
        result = precompute_axtree._cdp_nodes_to_tree(nodes)
        self.assertEqual(result["role"], "WebArea")
        # The ignored node (nodeId 2) must not appear; only the button survives.
        children = result.get("children", [])
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["role"], "button")
        self.assertEqual(children[0]["name"], "Submit")


if __name__ == "__main__":
    unittest.main()
