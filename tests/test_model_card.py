"""Model-card generation tests (M9)."""
from magicquant.utils.model_card import generate_model_card


SAMPLE = {
    "baseline_ppl": 5.1234,
    "tiered": {
        "Q6": {
            "config": {"E": "BF16", "H": "BF16", "Q": "Q6_K", "D": "MXFP4_MOE"},
            "ppl": 5.20,
            "measured_loss": 0.015,
            "size_gb": 3.4,
        },
        "Q4": {
            "config": {"E": "BF16", "H": "BF16", "Q": "IQ4_NL", "D": "MXFP4_MOE"},
            "ppl": 5.55,
            "measured_loss": 0.083,
            "size_gb": 2.1,
        },
    },
}


def test_card_contains_each_tier_and_metrics():
    card = generate_model_card(SAMPLE, base_model_name="Qwen3-4B")
    assert "Qwen3-4B" in card
    # Both tiers present
    assert "| Q6 |" in card
    assert "| Q4 |" in card
    # Measured PPL shown
    assert "5.200" in card
    assert "5.550" in card
    # Size shown
    assert "3.40" in card and "2.10" in card
    # Per-group scheme maps shown
    assert "Q:Q6_K" in card
    assert "D:MXFP4_MOE" in card
    # Attribution
    assert "magiccodingman" in card


def test_card_handles_missing_metrics():
    minimal = {"tiered": {"Q5": {"config": {"E": "BF16"}}}}
    card = generate_model_card(minimal, base_model_name="X")
    assert "| Q5 |" in card
    assert "E:BF16" in card  # em-dash placeholders for missing ppl/size are fine


def test_card_uses_tiered_survivors_fallback():
    data = {
        "tiered_survivors": {
            "Q8": {"config": {"E": "BF16"}, "ppl": 5.0, "measured_loss": 0.0, "size_gb": 8.0}
        }
    }
    card = generate_model_card(data, base_model_name="Y")
    assert "| Q8 |" in card


def test_card_warns_on_legacy_tier_scheme_version():
    """A search_results.json with no tier_scheme_version (pre-2026-07 fix,
    like SAMPLE above) must surface a disclosure banner -- its tier labels
    follow the OLD, wider size-ratio boundaries."""
    card = generate_model_card(SAMPLE, base_model_name="Qwen3-4B")
    assert "tier_scheme_version=1" in card
    assert "OLDER, wider size-ratio boundaries" in card


def test_card_no_warning_on_current_tier_scheme_version():
    from magicquant.quant.tiers import CURRENT_TIER_SCHEME_VERSION

    current = dict(SAMPLE, tier_scheme_version=CURRENT_TIER_SCHEME_VERSION)
    card = generate_model_card(current, base_model_name="Qwen3-4B")
    assert "tier_scheme_version=" not in card
    assert "OLDER, wider size-ratio boundaries" not in card


# --- Honesty gates (docs/audits/2026-11-01-model-card-review.md, C1-C3) ---

def _measured_run(**extra):
    base = dict(
        SAMPLE, baseline_provenance="measured", probing_provenance="measured",
    )
    base.update(extra)
    return base


import pytest  # noqa: E402


@pytest.mark.parametrize("prov", ["fabricated", "prediction-only", "unknown"])
def test_unmeasured_baseline_is_not_presented_as_measured(prov):
    """C1: the orchestrator's 5.0 placeholder must never read as a BF16 baseline."""
    data = _measured_run(baseline_ppl=5.0, baseline_provenance=prov)
    card = generate_model_card(data, base_model_name="X")
    assert "**Baseline (BF16) perplexity:** 5.0" not in card
    assert "not measured" in card and prov in card
    # No loss figures derived from a fabricated baseline.
    assert "1.50%" not in card and "8.33%" not in card


def test_measured_baseline_and_loss_are_shown():
    card = generate_model_card(_measured_run(), base_model_name="X")
    assert "**Baseline (BF16) perplexity:** 5.1234\n" in card
    # R2: PPL at 3 decimals; ΔPPL % is computed from PPL and baseline.
    assert "+1.50%" in card and "+8.33%" in card
    assert "| Q6 | `Q6_K` | measured |" in card
    assert "chosen from measured sensitivity" in card


def test_baseline_without_provenance_is_labelled():
    card = generate_model_card(SAMPLE, base_model_name="X")
    assert "provenance not recorded" in card
    assert "1.50%" in card  # pre-provenance files keep their figures


def test_predicted_tier_is_marked_and_shows_no_quality():
    """C2: a prediction-only tier must not look like a measured result."""
    data = {
        "baseline_ppl": 5.0,
        "baseline_provenance": "prediction-only",
        "probing_provenance": "heuristic",
        "tiered": {
            "Q4": {
                "config": {"E": "BF16", "D": "MXFP4_MOE"},
                "ppl": None,
                "measured_loss": 0.04,
                "predicted_loss": 0.05,
                "size_gb": 2.0,
            }
        },
    }
    card = generate_model_card(data, base_model_name="X")
    row = next(ln for ln in card.splitlines() if ln.startswith("| Q4 |"))
    assert "predicted (not built/measured)" in row
    assert "≈2.00" in row
    assert "4.00%" not in row and "5.00%" not in row
    assert "measured sensitivity" not in card
    assert "`heuristic`" in card


def test_kl_timeout_and_invalid_measurement_statuses():
    from magicquant.utils.naming import config_key

    q6 = SAMPLE["tiered"]["Q6"]["config"]
    q4 = SAMPLE["tiered"]["Q4"]["config"]
    data = _measured_run(measurements={
        config_key(q6): {"kl_timeout": True},
        config_key(q4): {"measurement_invalid": True},
    })
    card = generate_model_card(data, base_model_name="X")
    assert "| Q6 | `Q6_K` | measured (KL timed out) |" in card
    q4_row = next(ln for ln in card.splitlines() if ln.startswith("| Q4 |"))
    assert "invalid measurement" in q4_row
    assert "5.550" not in q4_row and "8.33%" not in q4_row


def test_sizes_are_labelled_gib_with_decimal_gb():
    """C3: size_gb is bytes/1024**3; label it GiB and add decimal GB."""
    data = _measured_run(tiered={
        "Q6": {"config": {"E": "BF16"}, "ppl": 5.2, "measured_loss": 0.01,
               "size_gb": 23.149005323648453},
    })
    card = generate_model_card(data, base_model_name="X")
    assert "Size (GiB)" in card and "Size (GB)" in card
    row = next(ln for ln in card.splitlines() if ln.startswith("| Q6 |"))
    assert "| 23.15 | 24.86 |" in row


def test_filename_label_is_disclosed_as_size_band():
    """C4: filenames carry an HF quant label (Q5 -> Q5_K_M) that names a size
    band, not a recipe. The card must say so and show each tier's label."""
    from magicquant.utils.naming import generate_name, hf_label_for_tier

    data = _measured_run(tiered={
        "Q5": {"config": {"D": "IQ4_NL", "E": "BF16", "U": "MXFP4_MOE"},
               "ppl": 6.68, "measured_loss": 0.016, "size_gb": 22.29},
    })
    card = generate_model_card(data, base_model_name="X")
    assert "names a size band, not a recipe" in card
    row = next(ln for ln in card.splitlines() if ln.startswith("| Q5 |"))
    assert "`Q5_K_M`" in row
    # The card's label and the shipped filename come from the same table.
    assert generate_name("Model-Q5", "MXFP4_MOE") == (
        f"Model-{hf_label_for_tier('Q5')}.gguf"
    )


def test_size_band_callout_present_even_without_tiers():
    card = generate_model_card({"tiered": {}}, base_model_name="X")
    assert "names a size band, not a recipe" in card
