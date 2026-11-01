"""magicquant kl-score: one reference, every file, one instrument."""
import json

import pytest

from magicquant.kl_score import load_kl_scores, score_directory


class _FakeTools:
    def __init__(self, llamacpp_path=None, ctx_size=512):
        self.ctx_size = ctx_size
        self.ppl_chunks = None
        self.perplexity_tool = "/fake/llama-perplexity"
        self.saved = []

    def _resolve_data_file(self, _):
        return "/data/wiki.test.raw"

    def save_base_logits(self, base, corpus, out, ctx_size=512):
        self.saved.append(out)
        open(out, "wb").write(b"logits")
        return 14.0

    def calculate_kl_divergence(self, quant, logits, corpus, ctx_size=512):
        if quant.endswith("bad.gguf"):
            return None
        return {"mean_kl": 0.01, "ppl": 14.1, "ppl_err": 0.1}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _FakeTools)
    for name in ("a-Q4.gguf", "bad.gguf"):
        (tmp_path / name).write_bytes(b"x" * 10)
    ref = tmp_path / "ref.gguf"
    ref.write_bytes(b"r")
    return tmp_path, ref


def test_scores_every_file_and_removes_logits_it_created(run_dir):
    d, ref = run_dir
    result = score_directory(d, str(ref), verbose=False)
    assert set(result["files"]) == {"a-Q4.gguf", "bad.gguf", "ref.gguf"}
    assert result["files"]["a-Q4.gguf"]["mean_kl"] == 0.01
    assert result["files"]["bad.gguf"]["error"] == "KL measurement failed"
    assert result["reference"] == "ref.gguf" and result["corpus"] == "wiki.test.raw"
    assert not (d / "_kl_score_base_logits.kld").exists()
    assert load_kl_scores(d)["files"]["a-Q4.gguf"]["bytes"] == 10


def test_keep_base_logits(run_dir):
    d, ref = run_dir
    score_directory(d, str(ref), keep_base_logits=True, verbose=False)
    assert (d / "_kl_score_base_logits.kld").exists()


def test_reused_logits_are_never_deleted(run_dir, tmp_path_factory):
    d, ref = run_dir
    logits = tmp_path_factory.mktemp("keep") / "base.kld"
    logits.write_bytes(b"precious")
    score_directory(d, str(ref), base_logits=str(logits), verbose=False)
    assert logits.read_bytes() == b"precious"


def test_missing_inputs_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("magicquant.utils.llamacpp.LlamaCppTools", _FakeTools)
    with pytest.raises(FileNotFoundError):
        score_directory(tmp_path, str(tmp_path / "nope.gguf"), verbose=False)
    assert load_kl_scores(tmp_path) is None
    (tmp_path / "kl_scores.json").write_text(json.dumps({"files": "not a dict"}))
    assert load_kl_scores(tmp_path) is None
