"""`magicquant generate` must build with the imatrix the search measured with.

Without it every imatrix-weighted scheme (K-quants, IQ family) encodes
differently from the measured candidate, so the shipped file is not the file
the search's (and the card's) numbers describe.
"""
import json
import sys

import pytest

import magicquant.__main__ as cli


class _FakeOrch:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.imatrix_calls = []
        self.imatrix_ok = True
        self.generated_with_imatrix = None
        _FakeOrch.last = self

    def enable_imatrix(self, corpus_path=None, **kw):
        self.imatrix_calls.append(corpus_path)
        return _FakeOrch.imatrix_ok

    def generate_tiered_models(self, tiered, model_name_prefix, tiers, verify):
        self.generated_with_imatrix = bool(self.imatrix_calls)
        return []


def _run(monkeypatch, tmp_path, measurement, *extra):
    (tmp_path / "search_results.json").write_text(json.dumps({
        "measurement": measurement,
        "tiered": {"Q4": {"config": {"E": "Q4_K_M"}}},
    }))
    monkeypatch.setattr("magicquant.orchestrator.MagicQuantOrchestrator", _FakeOrch)
    monkeypatch.delenv("MAGICQUANT_USE_IMATRIX", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "magicquant", "generate", "/tmp/base.gguf",
        "--output-dir", str(tmp_path), "--tiers", "Q4", *extra,
    ])
    cli.main()
    return _FakeOrch.last


def test_generate_follows_search_imatrix(monkeypatch, tmp_path):
    _FakeOrch.imatrix_ok = True
    orch = _run(monkeypatch, tmp_path, {"imatrix_active": True})
    assert orch.imatrix_calls == [None] and orch.generated_with_imatrix


def test_generate_without_search_imatrix_stays_unweighted(monkeypatch, tmp_path):
    _FakeOrch.imatrix_ok = True
    orch = _run(monkeypatch, tmp_path, {"imatrix_active": False})
    assert orch.imatrix_calls == [] and orch.generated_with_imatrix is False


def test_generate_refuses_when_search_imatrix_unavailable(monkeypatch, tmp_path):
    _FakeOrch.imatrix_ok = False
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, {"imatrix_active": True})
    assert _FakeOrch.last.generated_with_imatrix is None  # never generated


def test_no_imatrix_opt_out_warns(monkeypatch, tmp_path, capsys):
    _FakeOrch.imatrix_ok = True
    orch = _run(monkeypatch, tmp_path, {"imatrix_active": True}, "--no-imatrix")
    assert orch.imatrix_calls == []
    assert "will differ from the measured ones" in capsys.readouterr().out
