"""Degenerate-answer flags and their place in the compare reports."""
from pathlib import Path

from magicquant.compare.degeneration import (
    LOOP_MIN_REPEATS,
    detect_flags,
    max_repeat,
    target_script_share,
)
from magicquant.compare.metadata import ReproMetadata
from magicquant.compare.output import (
    InferenceResult,
    ModelInfo,
    generate_html,
    generate_markdown,
    populate_scores,
)
from magicquant.compare.scoring import ScoreResult

# The real loop from the Llama-3.2-1B run (bartowski Q8_0, Russian Q11).
_LOOP = "Время: " + "8120 часов = 337,5 * 24 = 8120 часов. " * 60
# A long but healthy explanation (the max real answer repeated a passage 4x).
_HEALTHY = " ".join(
    f"Step {i}: the attention weights for head {i} are computed from Q and K."
    for i in range(60)
)


def test_loop_is_flagged_and_healthy_long_answer_is_not():
    assert max_repeat(_LOOP) >= LOOP_MIN_REPEATS
    assert detect_flags(_LOOP, {}) == ["loop"]
    assert detect_flags(_HEALTHY, {}) == []


def test_repetition_inside_code_blocks_is_ignored():
    code = "```python\n" + "print(binary_search(arr, 5))\n" * 40 + "```"
    assert detect_flags("Here are tests:\n" + code, {}) == []


def test_wrong_language_needs_response_language():
    q_zh = {"response_language": "zh"}
    assert detect_flags("Beijing", q_zh) == ["wrong_language"]
    assert detect_flags("中国的首都是北京。", q_zh) == []
    assert detect_flags("Beijing", {}) == []  # no language requested
    assert target_script_share("پایتخت ایران تهران است", "fa") == 1.0


def test_empty_answer():
    assert detect_flags(None, {}) == ["empty"]
    assert detect_flags("(no response)", {}) == ["empty"]


def test_token_limit_alone_is_not_a_flag():
    # A long legitimate answer that simply stops: no repetition, right language.
    assert detect_flags(_HEALTHY[:5000], {}) == []


def _meta():
    return ReproMetadata(
        timestamp="2026-11-01T00:00:00", magicquant_version="0", git_commit="x",
        cli_args="", questions_file="q.yaml", questions_sha256="0" * 64,
        question_count=2, pool_size=2, difficulty_breakdown={},
        system_prompt="You are a helpful assistant.", temperature=0.0, top_p=1.0,
        top_k=0, max_tokens=100, context_size=512, n_samples=1, llamacpp_path=None,
    )


def _report(tmp_path):
    questions = [
        {"id": 1, "difficulty": "easy", "failure_mode": "multilingual",
         "scoring_type": "exact_numeric", "ground_truth": 1.6, "prompt": "?"},
        {"id": 2, "difficulty": "hard", "failure_mode": "proof",
         "scoring_type": "none", "prompt": "Prove it."},
    ]
    model = ModelInfo(path=Path("m.gguf"), filename="m.gguf", bpw=5.0,
                      quant_type="Q4_K_M", param_count=1, architecture="llama",
                      size_gb=1.0)
    res = {
        1: InferenceResult([_LOOP], [ScoreResult("fail", "x", None, 1.6)], 1.0,
                           "fail", _LOOP, flags=["loop"]),
        2: InferenceResult([_LOOP], [ScoreResult("unscored", "x", None, None)], 1.0,
                           "unscored", _LOOP, flags=["loop"]),
    }
    populate_scores(model, questions, res)
    html_path, md_path = tmp_path / "c.html", tmp_path / "c.md"
    generate_html([model], questions, {"m.gguf": res}, _meta(), html_path)
    generate_markdown([model], questions, {"m.gguf": res}, _meta(), md_path)
    return model, html_path.read_text(), md_path.read_text()


def test_flags_counted_including_unscored_answers(tmp_path):
    model, _, _ = _report(tmp_path)
    assert model.flagged_answers == 2          # the looping proof counts too
    assert (model.total_questions, model.unscored_questions) == (1, 1)


def test_reports_explain_and_show_flags(tmp_path):
    _, html, md = _report(tmp_path)
    assert "How to read this" in html and "<b>Flagged</b>" in html
    # The "why" paragraph, with this run's own example (first flagged answer).
    example = ("In this run, `m.gguf` produced a repetition loop on Q1 "
               "(multilingual), an answer scored fail. 1 more flagged answer(s)")
    assert "Why this comparison exists." in md and example in md
    assert "<code>m.gguf</code> produced a repetition loop on Q1" in html
    assert "**Numerical answers** are the most sensitive canary" in md
    assert html.count("⚠ repetition loop") == 2
    assert "<th>Flagged</th>" in html
    assert "**How to read this**" in md
    assert "| Mean KLD | Flagged | Score |" in md
    assert "### Flagged answers" in md
    assert "| m.gguf | Q2 | proof | unscored | repetition loop |" in md



def test_why_paragraph_omits_example_without_flags(tmp_path):
    from magicquant.compare.output import _why_paragraph

    text = _why_paragraph([], [], {})
    assert text.startswith("**Why this comparison exists.**")
    assert "In this run" not in text
    assert text.endswith("the kinds of questions you care about.")


def test_why_paragraph_unscored_example():
    from magicquant.compare.output import _why_paragraph

    model = ModelInfo(path=Path("q4.gguf"), filename="q4.gguf", bpw=5.0,
                      quant_type="Q4_K_M", param_count=1, architecture="llama",
                      size_gb=1.0)
    q = {"id": 16, "failure_mode": "proof"}
    r = InferenceResult([_LOOP], [], 1.0, "unscored", _LOOP, flags=["loop"])
    text = _why_paragraph([model], [q], {"q4.gguf": {16: r}})
    assert ("In this run, `q4.gguf` produced a repetition loop on Q16 (proof), "
            "an answer the score does not count.") in text
