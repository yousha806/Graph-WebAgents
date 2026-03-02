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

    # ------------------------------------------------------------------
    # flatten_axtree: format / structure
    # ------------------------------------------------------------------

    def test_flatten_axtree_basic_format(self):
        """Nodes use [role] "name" bracket format, not "role: name"."""
        snapshot = {
            "role": "WebArea",
            "name": "Root",
            "children": [
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Query", "value": "abc"},
            ],
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn('[WebArea] "Root"', text)
        self.assertIn('[button] "Submit"', text)
        self.assertIn('[textbox] "Query"', text)
        self.assertIn("value='abc'", text)
        # Old colon format must NOT appear
        self.assertNotIn("WebArea: Root", text)

    def test_flatten_axtree_pipe_indent(self):
        """Child nodes are indented with '| ' per depth level."""
        snapshot = {
            "role": "WebArea",
            "name": "Page",
            "children": [
                {"role": "button", "name": "Go"},
            ],
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        lines = text.splitlines()
        # Root at depth 0 — no indent
        self.assertFalse(lines[0].startswith("| "))
        # Child at depth 1 — one '| ' prefix
        self.assertTrue(lines[1].startswith("| "), repr(lines[1]))

    def test_flatten_axtree_aria_props(self):
        """ARIA state flags are shown in the text."""
        snapshot = {
            "role": "checkbox",
            "name": "Agree",
            "props": {"checked": True, "disabled": True},
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn("checked", text)
        self.assertIn("disabled", text)

    def test_flatten_axtree_backend_node_id(self):
        snapshot = {"role": "button", "name": "OK", "backend_node_id": 42}
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn("#id=42", text)

    def test_flatten_axtree_noise_role_filtered(self):
        """Noise roles (generic/none/etc.) with empty name are skipped;
        their children are promoted to the same depth."""
        snapshot = {
            "role": "WebArea",
            "name": "Root",
            "children": [
                {
                    "role": "generic",
                    "name": "",           # noise node — should be skipped
                    "children": [
                        {"role": "button", "name": "Click me"},
                    ],
                },
            ],
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn('[button] "Click me"', text)
        self.assertNotIn("[generic]", text)

    def test_flatten_axtree_noise_role_with_name_kept(self):
        """A noise-role node that has a name should NOT be filtered."""
        snapshot = {
            "role": "group",
            "name": "Settings panel",
        }
        text = precompute_axtree.flatten_axtree(snapshot)
        self.assertIn('[group] "Settings panel"', text)

    def test_flatten_axtree_empty_snapshot_returns_empty_string(self):
        self.assertEqual(precompute_axtree.flatten_axtree({}), "")
        self.assertEqual(precompute_axtree.flatten_axtree(None), "")

    def test_flatten_axtree_char_limit(self):
        """Output is capped at max_chars."""
        snapshot = {"role": "button", "name": "X" * 500}
        text = precompute_axtree.flatten_axtree(snapshot, max_chars=50)
        self.assertLessEqual(len(text), 50)

    def test_flatten_axtree_node_limit(self):
        """Tree is truncated after max_nodes lines."""
        root = {"role": "WebArea", "name": "r",
                "children": [{"role": "button", "name": f"b{i}"} for i in range(20)]}
        text = precompute_axtree.flatten_axtree(root, max_nodes=5, max_chars=100_000)
        self.assertLessEqual(len(text.splitlines()), 5)

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

    # ------------------------------------------------------------------
    # _cdp_nodes_to_tree
    # ------------------------------------------------------------------

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
        children = result.get("children", [])
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["role"], "button")
        self.assertEqual(children[0]["name"], "Submit")

    def test_cdp_nodes_to_tree_extracts_aria_props(self):
        """ARIA properties (disabled, checked, etc.) are captured into 'props'."""
        nodes = [
            {
                "nodeId": "1",
                "role": {"value": "WebArea"},
                "name": {"value": "Root"},
                "ignored": False,
                "childIds": ["2"],
            },
            {
                "nodeId": "2",
                "role": {"value": "checkbox"},
                "name": {"value": "Agree"},
                "ignored": False,
                "childIds": [],
                "properties": [
                    {"name": "checked", "value": {"type": "boolean", "value": True}},
                    {"name": "disabled", "value": {"type": "boolean", "value": True}},
                ],
            },
        ]
        result = precompute_axtree._cdp_nodes_to_tree(nodes)
        checkbox = result["children"][0]
        self.assertEqual(checkbox["role"], "checkbox")
        props = checkbox.get("props", {})
        self.assertTrue(props.get("checked"), "checked prop missing")
        self.assertTrue(props.get("disabled"), "disabled prop missing")

    def test_cdp_nodes_to_tree_backend_node_id(self):
        """backendDOMNodeId is preserved as backend_node_id."""
        nodes = [
            {
                "nodeId": "1",
                "role": {"value": "WebArea"},
                "name": {"value": "Root"},
                "backendDOMNodeId": 42,
                "ignored": False,
                "childIds": [],
            },
        ]
        result = precompute_axtree._cdp_nodes_to_tree(nodes)
        self.assertEqual(result.get("backend_node_id"), 42)

    def test_cdp_nodes_to_tree_empty_returns_empty(self):
        self.assertEqual(precompute_axtree._cdp_nodes_to_tree([]), {})
        self.assertEqual(precompute_axtree._cdp_nodes_to_tree(None), {})

    def test_cdp_nodes_to_tree_hierarchy_preserved(self):
        """Parent → child relationships via childIds are correctly nested."""
        nodes = [
            {
                "nodeId": "1",
                "role": {"value": "WebArea"},
                "name": {"value": "Root"},
                "ignored": False,
                "childIds": ["2"],
            },
            {
                "nodeId": "2",
                "role": {"value": "navigation"},
                "name": {"value": "Nav"},
                "ignored": False,
                "childIds": ["3"],
            },
            {
                "nodeId": "3",
                "role": {"value": "link"},
                "name": {"value": "Home"},
                "ignored": False,
                "childIds": [],
            },
        ]
        result = precompute_axtree._cdp_nodes_to_tree(nodes)
        nav = result["children"][0]
        self.assertEqual(nav["role"], "navigation")
        link = nav["children"][0]
        self.assertEqual(link["role"], "link")
        self.assertEqual(link["name"], "Home")

    # ------------------------------------------------------------------
    # sanitize_html
    # ------------------------------------------------------------------

    def test_sanitize_html_strips_scripts_and_links(self):
        raw = (
            "<html><head><link href='x.css'><link rel='icon' href='f.ico'></head>"
            "<body><script>alert(1)</script><p>ok</p></body></html>"
        )
        cleaned = precompute_axtree.sanitize_html(raw)
        self.assertNotIn("<script", cleaned.lower())
        self.assertNotIn("<link", cleaned.lower())
        self.assertIn("<p>ok</p>", cleaned)

    def test_sanitize_html_preserves_content(self):
        raw = "<div id='main'>Hello</div>"
        self.assertEqual(precompute_axtree.sanitize_html(raw), raw)

    def test_sanitize_html_empty(self):
        self.assertEqual(precompute_axtree.sanitize_html(""), "")
        self.assertEqual(precompute_axtree.sanitize_html(None), "")


if __name__ == "__main__":
    unittest.main()
