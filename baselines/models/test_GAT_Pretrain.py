"""
Mini test suite for GAT_Pretrain.py components.
Paste each cell into Colab AFTER running all definition cells, then run this cell.

Tests:
  1.  encode_texts            – shape + dtype
  2.  dom_to_graph normal     – node feats / edges / backend_ids
  3.  dom_to_graph empty      – fallback dummy node
  4.  GATLayer fp32           – forward shape
  5.  GATLayer bfloat16       – dtype consistency under autocast
  6.  DOMGraphTransformer     – stacked layers, output shape
  7.  GATPretrainer fp32      – node_logits / action_logits shapes
  8.  GATPretrainer bfloat16  – autocast end-to-end
  9.  Loss + backward         – gradients flow
  10. _bid helper             – dict / JSON-str / plain-str / None
"""

import traceback, json
import torch
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────────────
_PASS = "  ✓ "
_FAIL = "  ✗ "

def run(name, fn):
    try:
        msg = fn()
        print(f"{_PASS} {name}" + (f"  [{msg}]" if msg else ""))
    except Exception:
        print(f"{_FAIL} {name}")
        traceback.print_exc()
        print()

# ── shared dummy graph ────────────────────────────────────────────────────────
_N, _E = 12, 30
_nf = lambda dev=DEVICE: torch.randn(_N, NODE_FEAT_DIM, device=dev)
_ei = lambda dev=DEVICE: torch.randint(0, _N, (2, _E), dtype=torch.long, device=dev)
_te = lambda dev=DEVICE: torch.randn(1, SBERT_DIM, device=dev)

SAMPLE_HTML = """
<html><body>
  <div id="main">
    <button backend_node_id="42">Login</button>
    <input type="text" backend_node_id="99" />
    <a href="#" backend_node_id="7">Home</a>
    <p>Some paragraph text</p>
  </div>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — encode_texts
# ─────────────────────────────────────────────────────────────────────────────
def test_encode_texts():
    out = encode_texts(["click the login button", "search for flights"])
    assert out.shape == (2, SBERT_DIM), f"shape {out.shape}"
    assert out.dtype == torch.float32, f"dtype {out.dtype}"
    return f"shape {tuple(out.shape)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — dom_to_graph: normal HTML
# ─────────────────────────────────────────────────────────────────────────────
def test_dom_to_graph_normal():
    nf, ei, bids = dom_to_graph(SAMPLE_HTML)
    assert nf.ndim == 2 and nf.shape[1] == NODE_FEAT_DIM, f"feat shape {nf.shape}"
    assert ei.shape[0] == 2, "edge_index must be (2, E)"
    assert len(bids) == nf.shape[0], "backend_ids length mismatch"
    assert "42" in bids or "99" in bids, "backend_node_id not extracted from attrs"
    return f"N={nf.shape[0]}, E={ei.shape[1]}, ids_sample={bids[:3]}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — dom_to_graph: empty / None HTML → fallback dummy node
# ─────────────────────────────────────────────────────────────────────────────
def test_dom_to_graph_empty():
    nf, ei, bids = dom_to_graph("")
    assert nf.shape == (1, NODE_FEAT_DIM), f"expected (1,{NODE_FEAT_DIM}) got {nf.shape}"
    assert ei.shape == (2, 1), f"expected (2,1) got {ei.shape}"
    return "fallback node OK"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — GATLayer: float32 forward
# ─────────────────────────────────────────────────────────────────────────────
def test_gat_layer_fp32():
    layer = GATLayer(in_dim=16, out_dim=16, heads=4).to(DEVICE)
    x  = torch.randn(8, 16, device=DEVICE)
    ei = torch.tensor([[0,1,2,3,4,5,6,7,0,1],[1,2,3,4,5,6,7,0,2,3]], device=DEVICE)
    out = layer(x, ei)
    assert out.shape == (8, 16), f"shape {out.shape}"
    assert out.dtype == torch.float32
    return f"shape {tuple(out.shape)}, dtype {out.dtype}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — GATLayer: bfloat16 under autocast (A100 path)
# ─────────────────────────────────────────────────────────────────────────────
def test_gat_layer_bf16():
    if DEVICE != "cuda":
        return "skipped (no CUDA)"
    layer = GATLayer(in_dim=16, out_dim=16, heads=4).to(DEVICE)
    x  = torch.randn(8, 16, device=DEVICE)
    ei = torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]], device=DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x, ei)
    assert out.shape == (8, 16), f"shape {out.shape}"
    # output dtype may be bfloat16 or float32 depending on layer (both OK)
    return f"dtype={out.dtype}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — DOMGraphTransformer: stacked GAT layers
# ─────────────────────────────────────────────────────────────────────────────
def test_dom_graph_transformer():
    gat = DOMGraphTransformer().to(DEVICE)
    out = gat(_nf(), _ei())
    assert out.shape == (_N, GRAPH_OUT_DIM), f"shape {out.shape}"
    return f"shape {tuple(out.shape)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — GATPretrainer: fp32 output shapes
# ─────────────────────────────────────────────────────────────────────────────
def test_gat_pretrainer_fp32():
    m = GATPretrainer().to(DEVICE)
    nl, al = m(_nf(), _ei(), _te())
    assert nl.shape == (_N, 2),          f"node_logits {nl.shape}"
    assert al.shape == (NUM_ACTIONS,),   f"action_logits {al.shape}"
    return f"node_logits={tuple(nl.shape)}, action_logits={tuple(al.shape)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — GATPretrainer: bfloat16 autocast end-to-end
# ─────────────────────────────────────────────────────────────────────────────
def test_gat_pretrainer_bf16():
    if DEVICE != "cuda":
        return "skipped (no CUDA)"
    m = GATPretrainer().to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        nl, al = m(_nf(), _ei(), _te())
    assert nl.shape == (_N, 2)
    assert al.shape == (NUM_ACTIONS,)
    return f"node dtype={nl.dtype}, action dtype={al.dtype}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Loss computation + backward pass
# ─────────────────────────────────────────────────────────────────────────────
def test_loss_backward():
    m = GATPretrainer().to(DEVICE)
    labels   = torch.zeros(_N, dtype=torch.long, device=DEVICE)
    labels[2] = 1
    act      = torch.tensor(0, device=DEVICE)

    with torch.amp.autocast(_AMP_DEVICE, dtype=_AMP_DTYPE):
        nl, al   = m(_nf(), _ei(), _te())
        cls_w    = torch.tensor([1.0, float(_N - 1)], device=DEVICE)
        node_loss   = F.cross_entropy(nl, labels, weight=cls_w)
        action_loss = F.cross_entropy(al.unsqueeze(0), act.unsqueeze(0))
        loss = node_loss + 0.5 * action_loss

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "no gradients"
    return f"loss={loss.item():.4f}, params_with_grad={len(grad_norms)}"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — _bid helper: dict / JSON string / plain string / None
# ─────────────────────────────────────────────────────────────────────────────
def test_bid_helper():
    def _bid(c):
        if isinstance(c, dict):
            return str(c.get("backend_node_id", ""))
        if isinstance(c, str):
            try:
                d = json.loads(c)
                return str(d.get("backend_node_id", "")) if isinstance(d, dict) else c
            except Exception:
                return c
        return ""

    assert _bid({"backend_node_id": 42})         == "42",  "dict int"
    assert _bid({"backend_node_id": "99"})        == "99",  "dict str"
    assert _bid('{"backend_node_id": "7"}')       == "7",   "json str"
    assert _bid('{"backend_node_id": 123}')       == "123", "json int"
    assert _bid("plain-string")                   == "plain-string", "plain str"
    assert _bid(None)                             == "",    "None"
    assert _bid({})                               == "",    "empty dict"
    return "all 7 cases pass"

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print(" GAT_Pretrain — component tests")
print("=" * 55)
run("1.  encode_texts",                test_encode_texts)
run("2.  dom_to_graph normal HTML",    test_dom_to_graph_normal)
run("3.  dom_to_graph empty HTML",     test_dom_to_graph_empty)
run("4.  GATLayer fp32",               test_gat_layer_fp32)
run("5.  GATLayer bfloat16 autocast",  test_gat_layer_bf16)
run("6.  DOMGraphTransformer",         test_dom_graph_transformer)
run("7.  GATPretrainer fp32",          test_gat_pretrainer_fp32)
run("8.  GATPretrainer bfloat16",      test_gat_pretrainer_bf16)
run("9.  Loss + backward",             test_loss_backward)
run("10. _bid helper",                 test_bid_helper)
print("=" * 55)
