"""compare must attribute PPL by file identity and label quants correctly.

Regression: _enrich_ppl matched tiers by filename substring, so a reference
"ref-bartowski-Q8_0.gguf" inherited MagicQuant's Q8 tier PPL; the quant
label took the FIRST filename match, so "Model-f16-Q4_K_M.gguf" read F16.
"""
import json
from pathlib import Path

from magicquant.compare.cmd_compare import _enrich_ppl
from magicquant.compare.output import ModelInfo, _quant_from_filename
from magicquant.utils.naming import config_key


def _model(tmp_path, name, bpw=5.0):
    path = tmp_path / name
    if not path.exists():
        path.write_bytes(b"not a gguf")
    return ModelInfo(path=path, filename=name, bpw=bpw, quant_type="",
                     param_count=1, architecture="llama", size_gb=1.0)


def test_quant_label_takes_last_match():
    assert _quant_from_filename("Llama-3.2-1B-Instruct-f16-Q4_K_M.gguf") == "Q4_K_M"
    assert _quant_from_filename("Model-Q8_0.gguf") == "Q8_0"
    assert _quant_from_filename("ref-F16.gguf") == "F16"
    assert _quant_from_filename("weights.gguf") == "—"


def test_outside_file_does_not_inherit_tier_ppl(tmp_path):
    cfg = {"E": "Q8_0", "D": "BF16"}
    (tmp_path / "search_results.json").write_text(json.dumps({
        "baseline_ppl": 14.0,
        "tiered": {"Q8": {"config": cfg, "ppl": 14.22}},
    }))
    (tmp_path / "generated.json").write_text(json.dumps({"files": {
        "Model-f16-Q8_0.gguf": {"tier": "Q8", "config_key": config_key(cfg),
                                "verify_ppl": 14.2219},
    }}))
    ours = _model(tmp_path, "Model-f16-Q8_0.gguf", bpw=8.6)
    theirs = _model(tmp_path, "ref-bartowski-Q8_0.gguf", bpw=8.5)
    ref = _model(tmp_path, "ref-F16.gguf", bpw=16.0)
    _enrich_ppl([ours, theirs, ref], tmp_path)
    assert ours.ppl == 14.2219          # the shipped file's own verified PPL
    assert theirs.ppl is None           # never borrowed from the "Q8" tier
    assert ref.ppl == 14.0              # source baseline for a >=14 bpw file


def test_tier_ppl_matched_by_group_schemes_metadata(tmp_path):
    from tests.test_model_card_sections import _write_tier_gguf

    groups = {"E": "Q8_0", "D": "Q8_0"}
    _write_tier_gguf(tmp_path, "renamed.gguf", groups)
    (tmp_path / "search_results.json").write_text(json.dumps({
        "tiered": {"Q8": {"config": groups, "ppl": 9.5}},
    }))
    m = _model(tmp_path, "renamed.gguf")
    _enrich_ppl([m], tmp_path)
    assert m.ppl == 9.5
    assert isinstance(m.path, Path)


def test_kl_scores_supply_ppl_and_kld_for_every_file(tmp_path):
    (tmp_path / "search_results.json").write_text(json.dumps({
        "baseline_ppl": 14.0, "tiered": {"Q8": {"config": {"E": "Q8_0"}, "ppl": 14.22}},
    }))
    (tmp_path / "kl_scores.json").write_text(json.dumps({"reference": "ref-F16.gguf", "files": {
        "Model-f16-Q8_0.gguf": {"ppl": 14.2234, "mean_kl": 0.0134},
        "ref-bartowski-Q8_0.gguf": {"ppl": 14.0137, "mean_kl": 0.0013},
        "broken.gguf": {"error": "KL measurement failed"},
    }}))
    ours = _model(tmp_path, "Model-f16-Q8_0.gguf")
    theirs = _model(tmp_path, "ref-bartowski-Q8_0.gguf")
    broken = _model(tmp_path, "broken.gguf")
    _enrich_ppl([ours, theirs, broken], tmp_path)
    assert (ours.ppl, ours.mean_kld) == (14.2234, 0.0134)
    assert (theirs.ppl, theirs.mean_kld) == (14.0137, 0.0013)
    assert broken.ppl is None and broken.mean_kld is None
