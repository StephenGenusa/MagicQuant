"""Provenance the model card depends on is actually persisted
(docs/audits/2026-11-01-model-card-review.md: C10, D-b1, D-b3, D-b4, D-b5, D-b6).
"""
import json

from magicquant.orchestrator import MagicQuantOrchestrator
from magicquant.utils.naming import config_key
from magicquant.utils.provenance import run_info

from tests.test_writer import StubSource, _f32_tensor, create_hybrid_gguf


def _build(tmp_path, tensors, groups, base="Q8_0"):
    src = StubSource(tensors)
    out = str(tmp_path / "out.gguf")
    import magicquant.gguf.source as source_mod

    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        create_hybrid_gguf(
            output_path=out, base_model_path="ignored",
            quant_config={"base": base, "groups": groups}, verbose=False,
        )
    finally:
        source_mod.open_model_source = orig
    from magicquant.gguf.reader import GGUFReader

    with GGUFReader(out) as reader:
        return reader.get_metadata()


def test_clean_artifact_records_no_double_quant_and_no_fallbacks(tmp_path):
    meta = _build(tmp_path, [
        _f32_tensor("token_embd.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (256, 256)),
    ], {"E": "Q8_0", "D": "Q8_0"})
    assert json.loads(meta["magicquant.source_precision"]) == {
        "dequantized": False, "source_types": {},
    }
    assert json.loads(meta["magicquant.fallbacks"]) == {}


def test_fallbacks_are_summarized_in_metadata(tmp_path):
    # A 96-wide row cannot take a 256-block K-quant: the writer falls back.
    meta = _build(tmp_path, [
        _f32_tensor("token_embd.weight", (256, 256)),
        _f32_tensor("blk.0.ffn_down.weight", (64, 96)),
    ], {"E": "Q8_0", "D": "Q6_K"}, base="Q6_K")
    fallbacks = json.loads(meta["magicquant.fallbacks"])
    assert fallbacks, "expected a recorded fallback for the 96-wide tensor"
    (reason, info), = fallbacks.items()
    assert info["count"] == 1
    assert info["example"].startswith("blk.0.ffn_down.weight: Q6_K->")


def test_record_generated_writes_sidecar(tmp_path):
    orch = MagicQuantOrchestrator.__new__(MagicQuantOrchestrator)
    orch.output_dir = tmp_path
    orch.baseline_ppl = 6.5
    orch.baseline_provenance = "measured"
    gguf = tmp_path / "Model-Q5_K_M.gguf"
    gguf.write_bytes(b"x" * 1234)
    cfg = {"E": "BF16", "D": "IQ4_NL"}

    orch._record_generated(gguf, cfg, "Q5", 6.7)
    orch._record_generated(tmp_path / "missing.gguf", cfg, "Q4", None)  # non-fatal

    data = json.loads((tmp_path / "generated.json").read_text())
    entry = data["files"]["Model-Q5_K_M.gguf"]
    assert entry == {
        "tier": "Q5", "config_key": config_key(cfg), "bytes": 1234,
        "verify_ppl": 6.7, "verify_baseline_ppl": 6.5,
        "baseline_provenance": "measured",
    }
    assert "missing.gguf" not in data["files"]


def test_run_info_shape():
    info = run_info()
    assert set(info) == {"magicquant_version", "git_commit", "argv"}
    assert isinstance(info["argv"], list)
