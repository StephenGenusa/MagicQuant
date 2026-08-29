"""Tests for magicquant.compare.scoring"""

import pytest
from magicquant.compare.scoring import (
    ScoreResult,
    _strip_think,
    compute_consistency,
    extract_last_number,
    score_code_syntax,
    score_exact_numeric,
    score_keyphrase,
    score_none,
    score_quadratic_roots,
    score_response,
)


# ── _strip_think ───────────────────────────────────────────────────────────────

def test_strip_think_no_tags():
    text = "The answer is 42."
    result, truncated = _strip_think(text)
    assert result == text
    assert truncated is False

def test_strip_think_complete_block():
    text = "<think>I need to compute this.</think>\nThe answer is 42."
    result, truncated = _strip_think(text)
    assert result == "The answer is 42."
    assert truncated is False

def test_strip_think_truncated_inside():
    text = "<think>Let me work this out step by step..."
    result, truncated = _strip_think(text)
    assert result == ""
    assert truncated is True

def test_strip_think_multiline_block():
    text = "<think>\nLine 1\nLine 2\n</think>\nFinal answer: 7"
    result, truncated = _strip_think(text)
    assert result == "Final answer: 7"
    assert truncated is False

def test_strip_think_case_insensitive():
    text = "<THINK>working...</THINK>\nResult: 99"
    result, truncated = _strip_think(text)
    assert result == "Result: 99"
    assert truncated is False

def test_strip_think_uses_last_close_tag():
    # Multiple </think> tags — use the last one
    text = "<think>step1</think>mid<think>step2</think>final"
    result, truncated = _strip_think(text)
    assert result == "final"
    assert truncated is False


# ── extract_last_number ────────────────────────────────────────────────────────

def test_extract_last_number_plain():
    assert extract_last_number("the answer is 42") == 42.0

def test_extract_last_number_with_commas():
    assert extract_last_number("total: $1,234.56") == 1234.56

def test_extract_last_number_scientific():
    assert extract_last_number("2.99e8 m/s") == pytest.approx(2.99e8)

def test_extract_last_number_negative():
    assert extract_last_number("temperature is -15 degrees") == -15.0

def test_extract_last_number_last_wins():
    assert extract_last_number("start with 10, end with 20") == 20.0

def test_extract_last_number_none():
    assert extract_last_number("no numbers here at all") is None

def test_extract_last_number_skips_empty_trailing_line():
    # Last line with no number is skipped; second-to-last line's number is returned
    text = "The result is 42.\nThe answer is"
    assert extract_last_number(text) == 42.0

def test_extract_last_number_prefers_last_complete_line():
    # "The remainder is" (no number) → skip; previous line ends with 2
    text = "2^10 ≡ 2 (mod 7)\nThe remainder is"
    assert extract_last_number(text) == 7.0  # last number in previous line is 7


# ── score_exact_numeric ────────────────────────────────────────────────────────

def _q(ground_truth, tol=None):
    return {"scoring_type": "exact_numeric", "ground_truth": ground_truth,
            "tolerance": tol or {}}

def test_exact_numeric_pass_exact():
    r = score_exact_numeric("Answer: 360", _q(360, {"abs": 0}))
    assert r.status == "pass"

def test_exact_numeric_pass_with_tolerance():
    r = score_exact_numeric("About 1191 dollars", _q(1191.02, {"abs": 0.05}))
    assert r.status == "pass"

def test_exact_numeric_fail():
    r = score_exact_numeric("The answer is 400", _q(360, {"abs": 0}))
    assert r.status == "fail"

def test_exact_numeric_near_miss_5pct():
    # 5% of 360 = 18, so 378 is exactly at the boundary — near_miss
    r = score_exact_numeric("roughly 378", _q(360, {"abs": 0}))
    assert r.status == "near_miss"

def test_exact_numeric_near_miss_2x_tol():
    # abs_tol=0.05, near_miss band = 2×0.05 = 0.10
    # 0.14 fails primary (|0.14-0.05|=0.09 > 0.05) but passes 2× band (≤0.10)
    r = score_exact_numeric("0.14", _q(0.05, {"abs": 0.05}))
    assert r.status == "near_miss"

def test_exact_numeric_no_number():
    r = score_exact_numeric("I don't know the answer", _q(42))
    assert r.status == "fail"
    assert r.extracted is None

def test_exact_numeric_zero_ground_truth():
    r = score_exact_numeric("0", _q(0, {"abs": 0}))
    assert r.status == "pass"

def test_exact_numeric_strips_think_block():
    # Think block says 999 (wrong); post-think says 360 (right)
    response = "<think>I think it might be 999...</think>\nAnswer: 360"
    r = score_exact_numeric(response, _q(360, {"abs": 0}))
    assert r.status == "pass"
    assert r.extracted == 360.0

def test_exact_numeric_truncated_in_think():
    response = "<think>Let me compute 15 × 24 = "
    r = score_exact_numeric(response, _q(360))
    assert r.status == "fail"
    assert "truncated" in r.detail.lower()

def test_exact_numeric_think_block_does_not_contaminate():
    # Think block has the wrong number as its last value; post-think has correct
    response = "<think>Perhaps 1000? No, let me try 500...</think>\nThe result is 360."
    r = score_exact_numeric(response, _q(360, {"abs": 0}))
    assert r.status == "pass"


# ── score_quadratic_roots ──────────────────────────────────────────────────────

def _qr(roots):
    return {"scoring_type": "quadratic_roots", "ground_truth": roots,
            "tolerance": {"abs": 0.01}}

def test_quad_roots_both_found():
    # Clean response with just the roots
    r = score_quadratic_roots("The two solutions are x = 1/3 and x = 2.", _qr([0.3333, 2.0]))
    assert r.status == "pass"

def test_quad_roots_both_found_no_extras():
    r = score_quadratic_roots("x = 0.333 or x = 2", _qr([0.3333, 2.0]))
    assert r.status == "pass"

def test_quad_roots_both_found_with_working():
    # Model shows full algebraic working (3, 7, 2 as coefficients etc.) — still pass
    response = (
        "3x² - 7x + 2 = 0\n"
        "Using the quadratic formula: discriminant = 49 - 24 = 25\n"
        "x = (7 ± 5) / 6\n"
        "x = 2 or x = 1/3"
    )
    r = score_quadratic_roots(response, _qr([0.3333, 2.0]))
    assert r.status == "pass"

def test_quad_roots_one_missing():
    r = score_quadratic_roots("x = 2.0 only", _qr([0.3333, 2.0]))
    assert r.status == "fail"
    assert 0.3333 in r.expected

def test_quad_roots_no_numbers():
    r = score_quadratic_roots("I cannot solve this", _qr([0.3333, 2.0]))
    assert r.status == "fail"

def test_quad_roots_truncated_in_think():
    response = "<think>Discriminant = 49 - 24 = 25, so x = "
    r = score_quadratic_roots(response, _qr([0.3333, 2.0]))
    assert r.status == "fail"
    assert "truncated" in r.detail.lower()

def test_quad_roots_strips_think():
    # Wrong number in think; correct roots in post-think
    response = "<think>The roots might be 5 and 6.</think>\nx = 1/3 or x = 2"
    r = score_quadratic_roots(response, _qr([0.3333, 2.0]))
    assert r.status == "pass"


# ── score_keyphrase ────────────────────────────────────────────────────────────

def _qk(gt):
    return {"scoring_type": "keyphrase", "ground_truth": gt}

def test_keyphrase_single_pass():
    r = score_keyphrase("The capital of France is Paris, of course.", _qk("Paris"))
    assert r.status == "pass"

def test_keyphrase_single_fail():
    r = score_keyphrase("The capital is Lyon.", _qk("Paris"))
    assert r.status == "fail"

def test_keyphrase_case_insensitive():
    r = score_keyphrase("golden loaf bakery", _qk("Golden Loaf"))
    assert r.status == "pass"

def test_keyphrase_list_all_present():
    r = score_keyphrase("f'(x) = 9x² - 4x + 5 is the derivative", _qk(["9x", "4x", "5"]))
    assert r.status == "pass"

def test_keyphrase_list_missing_one():
    r = score_keyphrase("f'(x) = 9x² + 5", _qk(["9x", "4x", "5"]))
    assert r.status == "fail"
    assert "4x" in r.detail

def test_keyphrase_strips_think():
    # Keyphrase absent from think block but present in post-think
    response = "<think>I should mention Lyon...</think>\nThe capital is Paris."
    r = score_keyphrase(response, _qk("Paris"))
    assert r.status == "pass"

def test_keyphrase_truncated_in_think():
    response = "<think>Let me recall... the capital of Iran is Teh"
    r = score_keyphrase(response, _qk("تهران"))
    assert r.status == "fail"
    assert "truncated" in r.detail.lower()


# ── score_code_syntax ──────────────────────────────────────────────────────────

def test_code_syntax_valid_fenced():
    code = "```python\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n```"
    r = score_code_syntax(code, {})
    assert r.status == "pass"

def test_code_syntax_valid_unfenced():
    r = score_code_syntax("def foo(x: int) -> int:\n    return x * 2\n", {})
    assert r.status == "pass"

def test_code_syntax_invalid():
    r = score_code_syntax("```python\ndef bad(\n```", {})
    assert r.status == "fail"
    assert "SyntaxError" in r.detail

def test_code_syntax_empty():
    r = score_code_syntax("", {})
    # Empty string parses successfully in Python (empty module)
    assert r.status == "pass"

def test_code_syntax_strips_think():
    response = "<think>I'll write a Fibonacci function.</think>\n```python\ndef fib(n): return n\n```"
    r = score_code_syntax(response, {})
    assert r.status == "pass"


# ── score_none ─────────────────────────────────────────────────────────────────

def test_score_none():
    r = score_none("anything", {})
    assert r.status == "unscored"


# ── score_response dispatcher ──────────────────────────────────────────────────

def test_dispatcher_routes_exact_numeric():
    q = {"scoring_type": "exact_numeric", "ground_truth": 42, "tolerance": {}}
    r = score_response("42", q)
    assert r.status == "pass"

def test_dispatcher_routes_none():
    q = {"scoring_type": "none", "ground_truth": None}
    r = score_response("whatever", q)
    assert r.status == "unscored"

def test_dispatcher_unknown_type_defaults_to_none():
    q = {"scoring_type": "mystery_type"}
    r = score_response("anything", q)
    assert r.status == "unscored"


# ── compute_consistency ────────────────────────────────────────────────────────

def test_consistency_all_pass():
    scored = [ScoreResult("pass", "", None, None)] * 3
    assert compute_consistency(scored) == pytest.approx(1.0)

def test_consistency_all_fail():
    scored = [ScoreResult("fail", "", None, None)] * 3
    assert compute_consistency(scored) == pytest.approx(1.0)

def test_consistency_majority_pass():
    scored = [
        ScoreResult("pass", "", None, None),
        ScoreResult("pass", "", None, None),
        ScoreResult("fail", "", None, None),
    ]
    assert compute_consistency(scored) == pytest.approx(2 / 3)

def test_consistency_empty():
    assert compute_consistency([]) == pytest.approx(1.0)
