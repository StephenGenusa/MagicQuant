"""Model-card sections for the open findings of
docs/audits/2026-11-01-model-card-review.md (C5-C13, P1-P9, R1-R3)."""
import argparse
import json

import pytest

from magicquant.utils.model_card import (
    collect_card_inputs,
    generate_model_card,
    upload_blockers,
)
from magicquant.utils.naming import config_key

QWOPUS = "tests/fixtures/qwopus_search_results_2026-07-04/search_results.json"
GIB = 1024 ** 3


def _qwopus():
    with open(QWOPUS) as f:
        return json.load(f)


def _row(card, tier):
    return next(ln for ln in card.splitlines() if ln.startswith(f"| {tier} |"))


V2_BLOCK = {
    "config": {"E": "Q8_0", "U": "MXFP4_MOE"},
    "tensor_config": {}, "tensor_actual_types": {},
    "algo": "v2-budget", "budget_bytes": 24 * GIB,
    "predicted_bytes": int(23.5 * GIB), "actual_bytes": int(23.6 * GIB),
    "ppl": 6.70, "baseline_ppl": 6.50,
}


# ── C5: v2 budget builds ────────────────────────────────────────────────────

def test_fresh_v2_run_renders_budget_row_and_its_baseline():
    data = {"tier_scheme_version": 2, "tiered": {"BUDGET-24GiB": V2_BLOCK}}
    card = generate_model_card(data, base_model_name="X")
    row = _row(card, "BUDGET-24GiB")
    assert "measured" in row and "6.700" in row and "+3.08%" in row
    assert "6.5000 (measured by the v2 budget run)" in card
    assert "per-tensor scheme selection" in card


def test_v2_block_merged_into_v1_results_is_rendered():
    data = _qwopus()
    data["tiered"]["BUDGET-24GiB"] = V2_BLOCK
    card = generate_model_card(data, base_model_name="X")
    assert "| BUDGET-24GiB |" in card
    assert "per-tensor and per-tensor-group" in card


def test_v2_block_without_anchor_is_predicted():
    block = dict(V2_BLOCK, actual_bytes=None, ppl=None)
    card = generate_model_card({"tiered": {"BUDGET-24GiB": block}}, "X")
    row = _row(card, "BUDGET-24GiB")
    assert "predicted (not built/measured)" in row and "≈" in row


def test_empty_results_never_render_an_empty_table():
    card = generate_model_card({"tiered": {}}, base_model_name="X")
    assert "*no tiers recorded*" in card


# ── C6 / C7 / R2: KLD, ΔPPL, method ─────────────────────────────────────────

def test_kld_joined_via_config_key():
    card = generate_model_card(_qwopus(), base_model_name="X")
    assert "Mean KLD" in card and "p90 KLD" in card
    assert "0.0308" in _row(card, "Q5")


def test_ppl_error_and_tail_kl_are_shown_when_recorded():
    data = _qwopus()
    key = config_key(data["tiered"]["Q6"]["config"])
    data["measurements"][key]["kl"].update(
        ppl_err=0.041, p99_kl=0.2, same_top_p=95.5)
    card = generate_model_card(data, base_model_name="X")
    row = _row(card, "Q6")
    assert "6.649 ± 0.041" in row and "0.2000" in row and "95.50%" in row


def test_method_section_states_conditions_or_not_recorded():
    card = generate_model_card(_qwopus(), base_model_name="X")
    assert "were not recorded for this run" in card
    data = dict(_qwopus(), measurement={
        "chunks": 100, "ctx_size": 512, "corpus": "/data/wiki.test.raw",
        "imatrix_active": True, "kl_enabled": True,
    })
    card = generate_model_card(data, base_model_name="X")
    assert "`wiki.test.raw`" in card and "`512` tokens" in card
    assert "`100` (capped; not the full corpus)" in card
    assert "Importance matrix: active" in card


def test_predicted_loss_is_never_displayed():
    data = {"baseline_ppl": 5.0, "baseline_provenance": "measured",
            "tiered": {"Q4": {"config": {"E": "BF16"}, "ppl": None,
                              "predicted_loss": -0.987654, "size_gb": 2.0}}}
    card = generate_model_card(data, base_model_name="X")
    assert "98.77" not in card and "-0.98" not in card


# ── C8 / R1 / C12: what's inside ────────────────────────────────────────────

def test_group_legend_bf16_note_and_intro_from_groups():
    card = generate_model_card(_qwopus(), base_model_name="X")
    assert "| `S` | SSM / linear attention |" in card
    assert "stored on disk as **F16**" in card
    assert "MoE experts" not in card  # qwopus has no X/R group


# ── C9: legacy banner with BF16 size ────────────────────────────────────────

def test_legacy_banner_states_current_band_when_bf16_size_known():
    data = dict(_qwopus(), baseline_size_gb=50.6)
    card = generate_model_card(data, base_model_name="X")
    assert "Under the current bands" in card
    assert "Q6 → Q6" in card and "Q5 → Q6" in card
    assert "Ratio to BF16" in card
    assert "**BF16 size:** 50.60 GiB" in card


# ── C10 / P8: double quant and the upload gate ──────────────────────────────

def test_double_quant_callout_and_blockers():
    data = _qwopus()
    inputs = {"files": [{"filename": "M-Q5_K_M.gguf", "bytes": 10,
                         "source_precision": {"dequantized": True,
                                              "source_types": {"Q8_0": 12}}}]}
    card = generate_model_card(data, base_model_name="X", inputs=inputs)
    assert "**Double-quantized.**" in card and "12× Q8_0" in card
    unverified, dq = upload_blockers(data, inputs)
    assert unverified == [] and dq == ["M-Q5_K_M.gguf"]


def test_upload_blockers_for_unmeasured_run():
    data = {"baseline_ppl": 5.0, "baseline_provenance": "fabricated",
            "tiered": {"Q4": {"config": {"E": "BF16"}, "ppl": None,
                              "predicted_loss": 0.1, "size_gb": 2.0}}}
    unverified, dq = upload_blockers(data)
    assert any("baseline" in u for u in unverified)
    assert any("tier Q4" in u for u in unverified)
    assert dq == []


def _card_args(tmp_path, **kw):
    ns = dict(output_dir=str(tmp_path), model=None, base_model="org/M",
              upload=True, repo="org/M-GGUF", license="apache-2.0",
              quantized_by=None, pipeline_tag="text-generation", language=None,
              no_front_matter=False, no_hash=True, allow_unverified_card=False,
              acknowledge_double_quant=False)
    ns.update(kw)
    return argparse.Namespace(**ns)


def test_cmd_card_refuses_upload_of_unverified_card(tmp_path, capsys):
    from magicquant.__main__ import cmd_card

    (tmp_path / "search_results.json").write_text(json.dumps({
        "baseline_ppl": 5.0, "baseline_provenance": "prediction-only",
        "tiered": {"Q4": {"config": {"E": "BF16"}, "ppl": None,
                          "predicted_loss": 0.1, "size_gb": 2.0}},
    }))
    with pytest.raises(SystemExit):
        cmd_card(_card_args(tmp_path))
    out = capsys.readouterr().out
    assert "refusing to upload" in out and "--allow-unverified-card" in out
    assert (tmp_path / "README.md").is_file()  # the card is still written


# ── C11: caveats ────────────────────────────────────────────────────────────

def test_caveats_list_degraded_flags():
    data = _qwopus()
    key = config_key(data["tiered"]["Q4"]["config"])
    data["measurements"][key].update(measurement_timeout=True,
                                     timeout_leg="kl", kl_timeout=False)
    data["tiered"]["Q8"]["source"] = "incumbent"
    data["predictor_tracking"] = False
    data["requested_tiers_missing"] = [{"tier": "Q2", "band": "(0, 0.178]"}]
    card = generate_model_card(data, base_model_name="X")
    caveats = card.split("## Caveats")[1]
    assert "Q4: a measurement leg timed out (kl)" in caveats
    assert "Q8: the winner is a seeded stock" in caveats
    assert "did not track the measured ranking" in caveats
    assert "Requested tier Q2 has no file" in caveats


def test_caveats_distinguish_clean_from_not_recorded():
    old = generate_model_card(_qwopus(), base_model_name="X")
    assert "Not recorded (older results file" in old
    data = _qwopus()
    for m in data["measurements"].values():
        m.update(kl_timeout=False, measurement_timeout=False)
    data["predictor_tracking"] = True
    clean = generate_model_card(data, base_model_name="X")
    assert "## Caveats\n\nNone recorded." in clean


def test_v2_imputed_kappa_is_a_caveat():
    inputs = {"v2_results": {"kappa_provenance": {"U": "measured",
                                                  "D": "imputed-median"},
                             "failures": []}}
    card = generate_model_card({"tiered": {"BUDGET-24GiB": V2_BLOCK}}, "X",
                               inputs=inputs)
    assert "imputed, not measured, for: D" in card


# ── C13 / R3: precedence and ordering ───────────────────────────────────────

def test_tiered_takes_precedence_over_survivors():
    data = {"tiered": {"Q4": {"config": {"E": "Q8_0"}, "ppl": 5.0, "size_gb": 1.0}},
            "tiered_survivors": {"Q6": {"config": {"E": "BF16"}, "ppl": 5.0,
                                        "size_gb": 2.0}}}
    card = generate_model_card(data, base_model_name="X")
    assert "| Q4 |" in card and "| Q6 |" not in card


def test_rows_sorted_by_size_descending():
    data = {"tiered": {
        "Q4": {"config": {"E": "Q4_K_M"}, "ppl": 6.0, "size_gb": 9.0},
        "Q6": {"config": {"E": "Q6_K"}, "ppl": 5.5, "size_gb": 3.0},
    }}
    card = generate_model_card(data, base_model_name="X")
    assert card.index("| Q4 |") < card.index("| Q6 |")


# ── P1: front matter ────────────────────────────────────────────────────────

def test_front_matter_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    card = generate_model_card(_qwopus(), base_model_name="org/Model",
                               front_matter={"license": "apache-2.0",
                                             "quantized_by": "someone",
                                             "pipeline_tag": "text-generation",
                                             "languages": ["en"]})
    assert card.startswith("---\n")
    fm = yaml.safe_load(card.split("---\n")[1])
    assert fm["base_model"] == ["org/Model"]
    assert fm["base_model_relation"] == "quantized"
    assert fm["license"] == "apache-2.0" and fm["library_name"] == "gguf"
    assert "gguf" in fm["tags"] and fm["language"] == ["en"]


def test_front_matter_omits_base_model_for_free_text_name():
    card = generate_model_card(_qwopus(), base_model_name="My Model",
                               front_matter={})
    assert "base_model" not in card.split("---\n")[1]


def test_no_front_matter_by_default():
    assert not generate_model_card(_qwopus(), "org/M").startswith("---")


# ── P2: files, P9 usage (real GGUFs on disk) ────────────────────────────────

def _write_tier_gguf(tmp_path, name, groups):
    from tests.test_writer import StubSource, _f32_tensor, create_hybrid_gguf
    import magicquant.gguf.source as source_mod

    src = StubSource([_f32_tensor("token_embd.weight", (256, 256)),
                      _f32_tensor("blk.0.ffn_down.weight", (256, 256))])
    orig = source_mod.open_model_source
    source_mod.open_model_source = lambda *a, **k: src
    try:
        create_hybrid_gguf(output_path=str(tmp_path / name),
                           base_model_path="ignored",
                           quant_config={"base": "Q8_0", "groups": groups},
                           verbose=False)
    finally:
        source_mod.open_model_source = orig


def test_files_matched_to_tiers_with_sha_and_exact_size(tmp_path):
    groups = {"E": "Q8_0", "D": "Q8_0"}
    _write_tier_gguf(tmp_path, "M-Q8_0.gguf", groups)
    _write_tier_gguf(tmp_path, "stray.gguf", {"E": "Q8_0", "D": "Q4_0"})
    data = {"baseline_ppl": 5.0, "baseline_provenance": "measured",
            "tiered": {"Q8": {"config": groups, "ppl": 5.1, "size_gb": 99.0}}}
    inputs = collect_card_inputs(tmp_path)
    card = generate_model_card(data, base_model_name="X", inputs=inputs)
    f = next(f for f in inputs["files"] if f["filename"] == "M-Q8_0.gguf")
    assert len(f["sha256"]) == 64 and f["sha256"] in card
    assert "| `M-Q8_0.gguf` | Q8 |" in card
    assert "| `stray.gguf` | unmatched |" in card
    # Exact bytes from the file replace the results' size_gb (99.0).
    assert f"{f['bytes'] / GIB:.2f}" in _row(card, "Q8")
    assert "llama-cli -m M-Q8_0.gguf" in card
    assert "No tensor fell back from its requested scheme." in card


def test_generated_json_supplies_verified_ppl(tmp_path):
    groups = {"E": "Q8_0", "D": "Q8_0"}
    _write_tier_gguf(tmp_path, "M-Q8_0.gguf", groups)
    (tmp_path / "generated.json").write_text(json.dumps({"files": {
        "M-Q8_0.gguf": {"tier": "Q8", "config_key": config_key(groups),
                        "verify_ppl": 5.123}}}))
    data = {"tiered": {"Q8": {"config": groups, "ppl": 5.1, "size_gb": 1.0}}}
    card = generate_model_card(data, "X",
                               inputs=collect_card_inputs(tmp_path, hash_files=False))
    assert "| `M-Q8_0.gguf` | Q8 |" in card and "| 5.123 |" in card
    assert "SHA-256" not in card  # --no-hash


# ── P3 / P4 / P5 / P6 / P7 ──────────────────────────────────────────────────

def test_compatibility_notes_follow_schemes():
    card = generate_model_card(_qwopus(), base_model_name="X")
    compat = card.split("## Compatibility")[1].split("##")[0]
    assert "MXFP4_MOE" in compat and "I-quants" in compat
    assert "ROCmFPX" not in compat
    data = {"tiered": {"Q4": {"config": {"E": "ROCMFP4"}, "ppl": 5.0, "size_gb": 1.0}}}
    assert "**only** on the ciru-ai/ROCmFPX" in generate_model_card(data, "X")


def test_provenance_section():
    data = dict(_qwopus(), run_info={"magicquant_version": "0.4.0",
                                     "git_commit": "abc1234",
                                     "argv": ["magicquant", "search", "m.gguf"]},
                llamacpp_binary="/opt/llama/llama-perplexity")
    card = generate_model_card(data, base_model_name="X")
    prov = card.split("## Provenance")[1]
    assert "`0.4.0`, commit `abc1234`" in prov
    assert "none (not seeded" in prov
    assert "`magicquant search m.gguf`" in prov


def test_speed_column_when_bench_recorded():
    data = _qwopus()
    key = config_key(data["tiered"]["Q6"]["config"])
    data["measurements"][key]["bench"] = {"pp_ts": 812.4, "tg_ts": 41.27}
    card = generate_model_card(data, base_model_name="X")
    assert "Speed pp / tg (t/s)" in card and "812 / 41.3" in _row(card, "Q6")


def test_compare_run_is_referenced(tmp_path):
    run = tmp_path / "comparisons" / "2026-10-01_12-00-00"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({
        "question_count": 40, "git_commit": "abc1234",
        "timestamp": "2026-10-01", "questions_sha256": "f" * 64}))
    (run / "comparison.md").write_text("# report")
    card = generate_model_card(_qwopus(), "X", inputs=collect_card_inputs(tmp_path))
    section = card.split("## Task-level validation")[1]
    assert "`40` questions" in section
    assert "comparisons/2026-10-01_12-00-00/comparison.md" in section


def test_all_measured_candidates_listed_with_pareto():
    card = generate_model_card(_qwopus(), base_model_name="X")
    section = card.split("## All measured candidates")[1].split("</details>")[0]
    assert "7 configurations" in section
    assert "| 21.95 | 6.764 | +2.91% | 0.0264 | ✓ |  |" in section


def test_predictor_tracking_dict_shape_as_persisted():
    """The orchestrator persists {"is_tracking", "tau", "n_pairs"}."""
    data = dict(_qwopus(), predictor_tracking={"is_tracking": False,
                                               "tau": 0.05, "n_pairs": 15})
    card = generate_model_card(data, base_model_name="X")
    assert "**no** (Kendall τ 0.05 over 15 pairs)" in card
    assert "did not track the measured ranking" in card
    data["predictor_tracking"] = {"is_tracking": True, "tau": 0.43, "n_pairs": 15}
    card = generate_model_card(data, base_model_name="X")
    assert "yes (Kendall τ 0.43 over 15 pairs)" in card
    assert "did not track" not in card


# ── Real-run findings (Llama-3.2-1B test, 2026-11-01) ───────────────────────

def _two_tier_run():
    return {
        "baseline_ppl": 14.0, "baseline_provenance": "measured",
        "measurement": {"kl_enabled": True, "ctx_size": 512},
        "tiered": {
            "Q8": {"config": {"E": "Q8_0", "Q": "MXFP4_MOE"}, "ppl": 14.22, "size_gb": 1.24},
            "Q6": {"config": {"E": "Q6_K", "Q": "Q6_K"}, "ppl": 14.06, "size_gb": 0.95},
        },
        "measurements": {},
    }


def test_dominated_tier_is_flagged_and_not_used_in_usage():
    card = generate_model_card(_two_tier_run(), base_model_name="X")
    pick = card.split("## Which file should I pick?")[1].split("##")[0]
    assert "**Q8**" in pick and "not recommended**: Q6 is smaller" in pick
    assert "Q6 is smaller" not in pick.split("**Q6**")[1]
    assert "Q8: dominated by Q6" in card.split("## Caveats")[1]


def test_kl_method_distinguishes_probe_only_from_per_candidate():
    card = generate_model_card(_two_tier_run(), base_model_name="X")
    assert "sensitivity probes only; search candidates were not KL-scored" in card
    data = _two_tier_run()
    data["measurements"] = {"E:Q6_K|Q:Q6_K": {"kl": {"mean_kl": 0.01}}}
    assert "measured per candidate" in generate_model_card(data, "X")


def test_compare_scores_keep_unscored_out_of_denominator(tmp_path):
    run = tmp_path / "comparisons" / "20261101T000000"
    for model, statuses in {
        "M-Q6_K.gguf": ["pass", "pass", "fail", "unscored"],
        "ref-other-Q8_0.gguf": ["pass", "near_miss", "fail", "unscored"],
    }.items():
        d = run / "raw_responses" / model
        d.mkdir(parents=True)
        for i, st in enumerate(statuses):
            (d / f"Q{i:02d}.json").write_text(json.dumps({"primary_status": st}))
    (run / "meta.json").write_text(json.dumps({"question_count": 4}))
    card = generate_model_card(_two_tier_run(), "X",
                               inputs=collect_card_inputs(tmp_path, hash_files=False))
    section = card.split("## Task-level validation")[1]
    assert "| `M-Q6_K.gguf` | reference | — | 2 / 3 | 0 | 1 | 1 |" in section
    assert "| `ref-other-Q8_0.gguf` | reference | — | 1 / 3 | 1 | 1 | 1 |" in section



def test_kl_scores_listed_for_every_file_including_references(tmp_path):
    groups = {"E": "Q8_0", "D": "Q8_0"}
    _write_tier_gguf(tmp_path, "M-Q8_0.gguf", groups)
    _write_tier_gguf(tmp_path, "ref-other-Q8_0.gguf", {"E": "Q8_0", "D": "Q4_0"})
    (tmp_path / "kl_scores.json").write_text(json.dumps({
        "reference": "ref-F16.gguf", "corpus": "wiki.test.raw", "ctx_size": 512,
        "files": {
            "M-Q8_0.gguf": {"ppl": 14.2234, "ppl_err": 0.1057, "mean_kl": 0.013404,
                            "p99_kl": 0.0852, "same_top_p": 93.493},
            "ref-other-Q8_0.gguf": {"ppl": 14.0137, "mean_kl": 0.001348,
                                    "p99_kl": 0.0074, "same_top_p": 97.661},
        }}))
    data = {"baseline_ppl": 14.0, "baseline_provenance": "measured",
            "tiered": {"Q8": {"config": groups, "ppl": 14.22, "size_gb": 1.0}},
            "measurements": {}}
    card = generate_model_card(data, "X", inputs=collect_card_inputs(tmp_path, hash_files=False))
    files = card.split("## Files")[1].split("## Tiers")[0]
    assert "| Mean KLD | p99 KLD | Same top token |" in files
    assert "| `ref-other-Q8_0.gguf` | unmatched |" in files and "0.0013" in files
    assert "14.223 ± 0.106 | 0.0134 | 0.0852 | 93.49%" in files
    assert "measured by `magicquant kl-score` against `ref-F16.gguf`" in files
    # The tier table falls back to the shipped file's kl-score numbers.
    assert "0.0134" in _row(card, "Q8")
