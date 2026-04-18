"""
Tests for baselines/models/multimodal_GNN_Cross.py

Run from project root:
    pytest tests/test_multimodal_GNN_Cross.py -v

All tests are CPU-only. Heavy dependencies (Qwen, SBERT, peft, bitsandbytes)
are stubbed at the sys.modules level before the module under test is imported,
so no model weights are downloaded or loaded.
"""

import io
import os
import re
import sys
import json
import math
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

# ─── Stub heavy deps BEFORE importing the module under test ──────────────────
# Plain MagicMock satisfies `from X import Y` (attr access returns MagicMock).
# We must set __spec__ = None on each stub so importlib.util.find_spec doesn't
# raise ValueError("X.__spec__ is not set") when transformers probes them.
import types as _types

for _stub_name in [
    "qwen_vl_utils",
    "peft",
    "datasets",
    "sentence_transformers",
    "bitsandbytes",
    "accelerate",
    "tqdm",
    "transformers",   # stub entirely to avoid accelerate.__spec__ probe
]:
    if _stub_name not in sys.modules:
        _m = MagicMock()
        _m.__spec__  = None
        _m.__name__  = _stub_name
        _m.__path__  = []
        _m.__package__ = _stub_name
        sys.modules[_stub_name] = _m

# tqdm: make it a transparent passthrough so iteration-based code works
sys.modules["tqdm"].tqdm = lambda x, **kw: x

# peft: give named attrs that look like real objects
_peft = sys.modules["peft"]
_peft.LoraConfig     = MagicMock()
_peft.get_peft_model = MagicMock(side_effect=lambda m, cfg: m)
_peft.TaskType       = MagicMock()
_peft.PeftModel      = MagicMock()

# transformers: the module does `from transformers import X as Y` —
# all attributes on the stub return MagicMock, so no further setup needed.

# ─── Add baselines/models to path and import module under test ────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "baselines" / "models"))

import multimodal_GNN_Cross as M  # noqa: E402

# ─── Shared constants / helpers ───────────────────────────────────────────────
SAMPLE_HTML = """
<html><body>
  <div id="main">
    <button backend_node_id="42">Login</button>
    <input type="text" backend_node_id="99" />
    <a href="#" backend_node_id="7">Home</a>
    <p>Some paragraph text</p>
    <span>Inline element</span>
  </div>
</body></html>
"""

_N       = 12
_E       = 20
_D_GRAPH = M.GRAPH_OUT_DIM
_D_QWEN  = M.QWEN_HIDDEN_DIM

def _nf(n=_N):  return torch.randn(n, M.NODE_FEAT_DIM)
def _ei(n=_N, e=_E): return torch.randint(0, n, (2, e), dtype=torch.long)


class _FakeSBERT:
    """Lightweight SBERT stand-in: returns random 384-dim embeddings."""
    def encode(self, texts, convert_to_tensor=True, show_progress_bar=False, device="cpu"):
        return torch.randn(len(texts), 384)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractActionFromText:
    def test_keyword_click(self):
        assert M.extract_action_from_text("CLICK button") == "CLICK"

    def test_keyword_type(self):
        assert M.extract_action_from_text("TYPE 'hello'") == "TYPE"

    def test_keyword_scroll_lowercase(self):
        assert M.extract_action_from_text("scroll down the page") == "SCROLL"

    def test_keyword_hover(self):
        assert M.extract_action_from_text("HOVER over element") == "HOVER"

    def test_first_word_fallback(self):
        # "Submit" is not in the hard-coded keyword list; should fall back to
        # first token capitalised (≤ 10 chars, all alpha)
        result = M.extract_action_from_text("Submit the form")
        assert result == "SUBMIT"

    def test_none_input(self):
        assert M.extract_action_from_text(None) is None

    def test_empty_string(self):
        assert M.extract_action_from_text("") is None

    def test_non_string_input(self):
        assert M.extract_action_from_text(42) is None


class TestScreenshotToPil:
    def test_none_returns_white_rgb(self):
        img = M._screenshot_to_pil(None)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_pil_grayscale_converted_to_rgb(self):
        src = Image.new("L", (100, 100), 128)
        out = M._screenshot_to_pil(src)
        assert out.mode == "RGB"

    def test_pil_rgba_converted_to_rgb(self):
        src = Image.new("RGBA", (64, 64))
        out = M._screenshot_to_pil(src)
        assert out.mode == "RGB"

    def test_bytes_input_roundtrip(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, format="PNG")
        out = M._screenshot_to_pil(buf.getvalue())
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == (64, 48)

    def test_invalid_object_returns_blank(self):
        out = M._screenshot_to_pil(object())
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"


class TestSaveJsonl:
    def test_round_trip(self):
        records = [{"a": 1, "b": "hello"}, {"c": [True, None, 3.14]}]
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            path = f.name
        try:
            M.save_jsonl(path, records)
            with open(path, encoding="utf-8") as f:
                loaded = [json.loads(line) for line in f]
            assert loaded == records
        finally:
            os.unlink(path)

    def test_empty_records(self):
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            M.save_jsonl(path, [])
            assert os.path.getsize(path) == 0
        finally:
            os.unlink(path)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  dom_to_graph  (SBERT mocked via module-level _TEXT_ENCODER)
# ═════════════════════════════════════════════════════════════════════════════

class TestDomToGraph:
    def setup_method(self):
        M._TEXT_ENCODER = _FakeSBERT()

    def teardown_method(self):
        M._TEXT_ENCODER = None

    def test_normal_html_node_feat_shape(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        assert nf.ndim == 2
        assert nf.shape[1] == M.NODE_FEAT_DIM, (
            f"expected feat dim {M.NODE_FEAT_DIM}, got {nf.shape[1]}"
        )

    def test_normal_html_edge_index_shape(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        assert ei.shape[0] == 2, "edge_index first dim must be 2"
        assert ei.dtype == torch.long

    def test_node_count_bounded_by_max_nodes(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        assert nf.shape[0] <= M.MAX_NODES

    def test_node_feat_dtype_float32(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        assert nf.dtype == torch.float32

    def test_empty_html_fallback_single_node(self):
        nf, ei = M.dom_to_graph("")
        assert nf.shape == (1, M.NODE_FEAT_DIM), (
            f"fallback node wrong shape: {nf.shape}"
        )
        assert ei.shape == (2, 1)

    def test_none_html_fallback(self):
        nf, ei = M.dom_to_graph(None)
        assert nf.shape[0] == 1

    def test_edges_within_node_range(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        n = nf.shape[0]
        assert ei.min().item() >= 0
        assert ei.max().item() < n

    def test_interactive_nodes_prioritised(self):
        # 5 buttons + 100 paragraphs — all 5 buttons should fit inside MAX_NODES
        html = (
            "<html><body>"
            + "<button>B</button>" * 5
            + "<p>P</p>" * 100
            + "</body></html>"
        )
        nf, ei = M.dom_to_graph(html)
        # At least as many nodes as there are interactive elements
        assert nf.shape[0] >= 5

    def test_sbert_feature_range(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        # Tag one-hot block (indices 384:518) should be 0 or 1
        tag_block = nf[:, 384: 384 + M.TAG_VOCAB_SIZE]
        assert ((tag_block == 0) | (tag_block == 1)).all()

    def test_structural_depth_normalized(self):
        nf, ei = M.dom_to_graph(SAMPLE_HTML)
        # Structural features start at index 384+134=518; depth is index 518
        depth_col = nf[:, 518]
        assert (depth_col >= 0).all() and (depth_col <= 1).all()


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GATLayer
# ═════════════════════════════════════════════════════════════════════════════

class TestGATLayer:
    def test_output_shape(self):
        layer = M.GATLayer(in_dim=16, out_dim=16, heads=4)
        x  = torch.randn(8, 16)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                           [1, 2, 3, 4, 5, 6, 7, 0]])
        out = layer(x, ei)
        assert out.shape == (8, 16), f"expected (8,16) got {out.shape}"

    def test_output_dtype_fp32(self):
        layer = M.GATLayer(in_dim=8, out_dim=8, heads=2)
        x  = torch.randn(4, 8)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        out = layer(x, ei)
        assert out.dtype == torch.float32

    def test_head_dim_calculation(self):
        layer = M.GATLayer(in_dim=8, out_dim=8, heads=2)
        assert layer.head_dim == 4

    def test_single_node_self_loop(self):
        layer = M.GATLayer(in_dim=4, out_dim=4, heads=2)
        x  = torch.randn(1, 4)
        ei = torch.tensor([[0], [0]])
        out = layer(x, ei)
        assert out.shape == (1, 4)

    def test_gradients_flow_through_x(self):
        layer = M.GATLayer(in_dim=8, out_dim=8, heads=2)
        x  = torch.randn(5, 8, requires_grad=True)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        layer(x, ei).sum().backward()
        assert x.grad is not None

    def test_gradients_flow_through_weights(self):
        layer = M.GATLayer(in_dim=8, out_dim=8, heads=2)
        x  = torch.randn(5, 8)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        layer(x, ei).sum().backward()
        assert layer.W.weight.grad is not None

    def test_varying_node_count(self):
        layer = M.GATLayer(in_dim=16, out_dim=16, heads=4)
        for n in [1, 5, 20, 64]:
            ei = torch.randint(0, n, (2, n * 2))
            out = layer(torch.randn(n, 16), ei)
            assert out.shape == (n, 16), f"failed for n={n}"


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DOMGraphTransformer
# ═════════════════════════════════════════════════════════════════════════════

class TestDOMGraphTransformer:
    def test_output_shape_default_config(self):
        gat = M.DOMGraphTransformer()
        out = gat(_nf(), _ei())
        assert out.shape == (_N, _D_GRAPH), (
            f"expected ({_N},{_D_GRAPH}) got {out.shape}"
        )

    def test_output_dtype_float32(self):
        gat = M.DOMGraphTransformer()
        out = gat(_nf(), _ei())
        assert out.dtype == torch.float32

    def test_single_node_graph(self):
        gat = M.DOMGraphTransformer()
        x  = torch.randn(1, M.NODE_FEAT_DIM)
        ei = torch.tensor([[0], [0]], dtype=torch.long)
        out = gat(x, ei)
        assert out.shape == (1, _D_GRAPH)

    def test_layer_and_norm_count_match_gat_layers(self):
        gat = M.DOMGraphTransformer()
        assert len(gat.layers)      == M.GAT_LAYERS
        assert len(gat.norm_layers) == M.GAT_LAYERS

    def test_custom_dims(self):
        gat = M.DOMGraphTransformer(
            in_dim=32, hidden_dim=64, out_dim=128, heads=4, num_layers=2
        )
        x  = torch.randn(5, 32)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        out = gat(x, ei)
        assert out.shape == (5, 128)

    def test_backward_pass(self):
        gat = M.DOMGraphTransformer()
        x  = _nf().requires_grad_(True)
        ei = _ei()
        gat(x, ei).sum().backward()
        assert x.grad is not None


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GraphCrossAttentionAdapter
# ═════════════════════════════════════════════════════════════════════════════

class TestGraphCrossAttentionAdapter:
    @staticmethod
    def _make(graph_dim=64, qwen_dim=128, heads=4):
        return M.GraphCrossAttentionAdapter(
            graph_dim=graph_dim, qwen_dim=qwen_dim, attn_heads=heads
        )

    def test_output_shape_matches_hidden_states(self):
        adapter = self._make()
        B, S, D_Q, N_nodes, D_G = 2, 10, 128, 8, 64
        hs    = torch.randn(B, S, D_Q)
        nodes = torch.randn(N_nodes, D_G)
        out   = adapter(hs, nodes)
        assert out.shape == (B, S, D_Q), f"expected {(B,S,D_Q)} got {out.shape}"

    def test_gate_initialised_to_zero(self):
        adapter = self._make()
        assert adapter.gate.item() == 0.0, (
            "gate must start at 0 so tanh(gate)=0 → adapter is identity at init"
        )

    def test_identity_at_init(self):
        """tanh(0) = 0, so output must exactly equal hidden_states at init."""
        adapter = self._make()
        hs    = torch.randn(1, 6, 128)
        nodes = torch.randn(4, 64)
        out   = adapter(hs, nodes)
        assert torch.allclose(out, hs, atol=1e-6), (
            "adapter should be identity when gate=0"
        )

    def test_non_identity_after_gate_update(self):
        adapter = self._make()
        with torch.no_grad():
            adapter.gate.fill_(3.0)   # tanh(3) ≈ 0.995
        hs    = torch.randn(1, 6, 128)
        nodes = torch.randn(4, 64)
        out   = adapter(hs, nodes)
        assert not torch.allclose(out, hs, atol=1e-4), (
            "adapter should modify hidden states when gate != 0"
        )

    def test_gradients_flow_to_hidden_states(self):
        adapter = self._make()
        with torch.no_grad():
            adapter.gate.fill_(1.0)
        hs    = torch.randn(1, 4, 128, requires_grad=True)
        nodes = torch.randn(3, 64)
        adapter(hs, nodes).sum().backward()
        assert hs.grad is not None

    def test_gate_receives_gradient(self):
        adapter = self._make()
        with torch.no_grad():
            adapter.gate.fill_(1.0)
        hs    = torch.randn(1, 4, 128)
        nodes = torch.randn(3, 64)
        adapter(hs, nodes).sum().backward()
        assert adapter.gate.grad is not None

    def test_node_proj_dimensions(self):
        adapter = self._make(graph_dim=64, qwen_dim=256)
        assert adapter.node_proj.in_features  == 64
        assert adapter.node_proj.out_features == 256

    def test_batch_size_one_vs_two(self):
        adapter = self._make()
        nodes = torch.randn(5, 64)
        for B in [1, 2, 4]:
            hs  = torch.randn(B, 8, 128)
            out = adapter(hs, nodes)
            assert out.shape == (B, 8, 128)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  QwenWithDOMGraph inject hook
# ═════════════════════════════════════════════════════════════════════════════

class TestQwenWithDOMGraphHook:
    @staticmethod
    def _make_combined(graph_dim=16, qwen_dim=32):
        graph_enc = M.DOMGraphTransformer(
            in_dim=32, hidden_dim=16, out_dim=graph_dim, heads=2, num_layers=1
        )
        adapter = M.GraphCrossAttentionAdapter(
            graph_dim=graph_dim, qwen_dim=qwen_dim, attn_heads=4
        )
        qwen_mock = MagicMock()
        return M.QwenWithDOMGraph(qwen_mock, graph_enc, adapter, inject_layer=20)

    def test_hook_output_shape_unchanged(self):
        combined  = self._make_combined()
        node_embs = torch.randn(5, 16)
        combined._pending_node_embs = node_embs
        hook_fn   = combined._build_inject_hook()
        hs        = torch.randn(1, 10, 32)
        new_args, _ = hook_fn(None, (hs,), {})
        assert new_args[0].shape == hs.shape

    def test_hook_is_identity_at_gate_zero(self):
        combined  = self._make_combined()
        node_embs = torch.randn(5, 16)
        combined._pending_node_embs = node_embs
        hook_fn   = combined._build_inject_hook()
        hs        = torch.randn(1, 10, 32)
        new_args, _ = hook_fn(None, (hs,), {})
        assert torch.allclose(new_args[0], hs, atol=1e-6)

    def test_hook_clears_pending_node_embs(self):
        combined  = self._make_combined()
        combined._pending_node_embs = torch.randn(5, 16)
        hook_fn   = combined._build_inject_hook()
        hook_fn(None, (torch.randn(1, 4, 32),), {})
        assert combined._pending_node_embs is None

    def test_hook_passthrough_when_no_pending_embs(self):
        """If _pending_node_embs is None, hook must return args unchanged."""
        combined  = self._make_combined()
        combined._pending_node_embs = None
        hook_fn   = combined._build_inject_hook()
        hs        = torch.randn(1, 6, 32)
        result    = hook_fn(None, (hs,), {"key": "val"})
        # Should return original args, kwargs unchanged
        new_args, new_kwargs = result
        assert new_args[0] is hs
        assert new_kwargs["key"] == "val"

    def test_extra_args_preserved(self):
        combined  = self._make_combined()
        combined._pending_node_embs = torch.randn(3, 16)
        hook_fn   = combined._build_inject_hook()
        hs        = torch.randn(1, 4, 32)
        extra     = torch.ones(2)
        new_args, _ = hook_fn(None, (hs, extra), {})
        # args beyond index 0 must be passed through unchanged
        assert new_args[1] is extra


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Predict output parsing  (logic extracted from predict())
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictParsing:
    """Tests the [ID]-extraction and fallback logic inside predict()."""

    @staticmethod
    def _parse(gen_text: str, num_candidates: int, top_k: int = 3):
        matches = re.findall(r'\[(\d+)\]', gen_text)
        ranked = []
        for m in matches:
            idx = int(m)
            if 0 <= idx < num_candidates and idx not in ranked:
                ranked.append(idx)
        if not ranked:
            ranked = list(range(min(max(top_k, 3), num_candidates)))
        return ranked[:top_k] if top_k > 1 else (ranked[0] if ranked else 0)

    def test_valid_three_results_in_order(self):
        text = "1. [2] click search\n2. [0] type query\n3. [5] scroll down"
        assert self._parse(text, 10, 3) == [2, 0, 5]

    def test_deduplicates_repeated_ids(self):
        text = "1. [1] action\n2. [1] repeated\n3. [2] another"
        result = self._parse(text, 5, 3)
        assert result == [1, 2]   # [1] deduplicated; only 2 unique

    def test_filters_out_of_range_id(self):
        text = "1. [99] invalid\n2. [0] valid\n3. [1] also valid"
        result = self._parse(text, 5, 3)
        assert 99 not in result
        assert 0 in result

    def test_fallback_on_unparseable_output(self):
        result = self._parse("I cannot determine the best action.", 5, 3)
        assert isinstance(result, list) and len(result) == 3
        assert all(0 <= i < 5 for i in result)

    def test_top_k_1_returns_int(self):
        text = "1. [3] some action"
        result = self._parse(text, 10, top_k=1)
        assert isinstance(result, int)
        assert result == 3

    def test_top_k_truncation(self):
        text = "1. [0] a\n2. [1] b\n3. [2] c\n4. [3] d"
        assert self._parse(text, 10, top_k=2) == [0, 1]

    def test_fallback_respects_num_candidates(self):
        result = self._parse("garbage text", num_candidates=2, top_k=3)
        assert all(0 <= i < 2 for i in result)

    def test_empty_generation_fallback(self):
        result = self._parse("", 4, 3)
        assert isinstance(result, list)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Training target string construction
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainingTargetFormat:
    """Tests the top-3 ranked-list target logic from Mind2WebDataset.__getitem__."""

    @staticmethod
    def _build(candidates, target_idx):
        neg_cands = [(i, c) for i, c in enumerate(candidates) if i != int(target_idx)]
        random.shuffle(neg_cands)
        top3 = [(int(target_idx), candidates[int(target_idx)])] + neg_cands[:2]
        return "\n".join([f"{r+1}. [{i}] {c}" for r, (i, c) in enumerate(top3)])

    def test_ground_truth_is_rank_1(self):
        cands  = ["click A", "type B", "scroll C", "hover D"]
        target = self._build(cands, 2)
        assert target.startswith("1. [2]"), (
            f"GT not at rank 1:\n{target}"
        )

    def test_exactly_three_lines_when_enough_candidates(self):
        cands  = ["A", "B", "C", "D", "E"]
        target = self._build(cands, 0)
        lines  = [l for l in target.strip().split("\n") if l]
        assert len(lines) == 3

    def test_format_uses_bracket_ids(self):
        cands  = ["X", "Y", "Z", "W"]
        target = self._build(cands, 1)
        ids    = re.findall(r'\[(\d+)\]', target)
        assert len(ids) == 3

    def test_no_duplicate_candidate_ids(self):
        cands  = ["a", "b", "c", "d", "e"]
        target = self._build(cands, 0)
        ids    = [int(x) for x in re.findall(r'\[(\d+)\]', target)]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in target: {ids}"

    def test_two_candidates_gives_two_lines(self):
        # Only GT + 1 negative available → 2-line target
        cands  = ["only_a", "only_b"]
        target = self._build(cands, 0)
        lines  = [l for l in target.strip().split("\n") if l]
        assert len(lines) == 2

    def test_one_candidate_gives_one_line(self):
        cands  = ["solo"]
        target = self._build(cands, 0)
        assert "[0]" in target
        lines  = [l for l in target.strip().split("\n") if l]
        assert len(lines) == 1

    def test_rank_numbers_are_sequential(self):
        cands  = ["a", "b", "c", "d"]
        target = self._build(cands, 1)
        rank_nums = [int(m) for m in re.findall(r'^(\d+)\.', target, re.MULTILINE)]
        assert rank_nums == list(range(1, len(rank_nums) + 1))


# ═════════════════════════════════════════════════════════════════════════════
# 9.  GAT weight key remapping  (checkpoint loading logic)
# ═════════════════════════════════════════════════════════════════════════════

class TestGATWeightKeyRemapping:
    """Tests the key-prefix stripping used when loading a GAT_Pretrain checkpoint."""

    def test_gat_prefix_stripped(self):
        raw = {
            "gat.layers.0.W.weight": torch.zeros(1),
            "gat.layers.1.W.weight": torch.zeros(1),
            "node_head.weight":       torch.zeros(1),  # should be excluded
        }
        gat_state = {k[len("gat."):]: v for k, v in raw.items() if k.startswith("gat.")}
        assert "layers.0.W.weight" in gat_state
        assert "layers.1.W.weight" in gat_state
        assert "node_head.weight"  not in gat_state

    def test_non_gat_keys_excluded(self):
        raw = {
            "gat.norm_layers.0.weight": torch.ones(1),
            "task_proj.weight":          torch.ones(1),
            "action_head.bias":          torch.ones(1),
        }
        gat_state = {k[len("gat."):]: v for k, v in raw.items() if k.startswith("gat.")}
        assert list(gat_state.keys()) == ["norm_layers.0.weight"]

    def test_checkpoint_dict_wrapper_unwrapped(self):
        # ckpt = {"model_state_dict": {...}} format from GAT_Pretrain.py
        raw_state = {"gat.layers.0.W.weight": torch.zeros(2, 2)}
        ckpt      = {"model_state_dict": raw_state}
        extracted = ckpt.get("model_state_dict", ckpt)
        gat_state = {k[len("gat."):]: v for k, v in extracted.items() if k.startswith("gat.")}
        assert "layers.0.W.weight" in gat_state

    def test_bare_checkpoint_also_works(self):
        # ckpt stored without "model_state_dict" key
        raw_state = {"gat.layers.0.W.weight": torch.zeros(2, 2)}
        extracted = raw_state.get("model_state_dict", raw_state)
        gat_state = {k[len("gat."):]: v for k, v in extracted.items() if k.startswith("gat.")}
        assert "layers.0.W.weight" in gat_state

    def test_load_state_dict_compatible_weights(self):
        """Verify that remapped keys can actually be loaded into DOMGraphTransformer."""
        gat = M.DOMGraphTransformer(
            in_dim=8, hidden_dim=4, out_dim=4, heads=2, num_layers=1
        )
        # Build a fake checkpoint from gat's own state dict (prefixed with "gat.")
        fake_ckpt = {f"gat.{k}": v.clone() for k, v in gat.state_dict().items()}
        gat_state = {
            k[len("gat."):]: v for k, v in fake_ckpt.items() if k.startswith("gat.")
        }
        missing, unexpected = gat.load_state_dict(gat_state, strict=False)
        assert len(missing) == 0, f"Missing keys after remapping: {missing}"


# ═════════════════════════════════════════════════════════════════════════════
# 10.  Config sanity checks
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigSanity:
    def test_node_feat_dim_is_523(self):
        # 384 (SBERT) + 134 (tag one-hot) + 5 (structural) = 523
        assert M.NODE_FEAT_DIM == 523

    def test_tag_vocab_size_is_134(self):
        assert M.TAG_VOCAB_SIZE == 134
        assert len(M.TAG_VOCAB) == 134

    def test_graph_inject_layer_in_valid_range(self):
        # Qwen2.5-VL-8B has 28 transformer layers (0-indexed: 0..27)
        assert 0 < M.GRAPH_INJECT_LAYER < 28

    def test_lora_alpha_at_least_rank(self):
        # Standard convention: alpha >= rank
        assert M.LORA_ALPHA >= M.LORA_RANK

    def test_max_target_len_fits_top3(self):
        # "1. [49] a fairly long candidate action text " × 3 ≈ 60 tokens minimum
        assert M.MAX_TARGET_LEN >= 60

    def test_gat_heads_divides_hidden_dim(self):
        assert M.GRAPH_HIDDEN_DIM % M.GAT_HEADS == 0, (
            f"GAT_HEADS={M.GAT_HEADS} must divide GRAPH_HIDDEN_DIM={M.GRAPH_HIDDEN_DIM}"
        )

    def test_gat_heads_divides_out_dim(self):
        assert M.GRAPH_OUT_DIM % M.GAT_HEADS == 0, (
            f"GAT_HEADS={M.GAT_HEADS} must divide GRAPH_OUT_DIM={M.GRAPH_OUT_DIM}"
        )

    def test_proj_attn_heads_divides_qwen_dim(self):
        assert M.QWEN_HIDDEN_DIM % M.PROJ_ATTN_HEADS == 0

    def test_gat_frozen_default_true(self):
        assert M.GAT_FROZEN is True, "GAT should be frozen by default"

    def test_graph_out_dim_matches_gat_pretrain(self):
        # Must match GRAPH_OUT_DIM in GAT_Pretrain.py (512 for A100 default)
        assert M.GRAPH_OUT_DIM == 512
