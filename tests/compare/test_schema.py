"""Tests for questions.yaml schema validity."""

import math
from pathlib import Path

import pytest

QUESTIONS_FILE = (
    Path(__file__).parent.parent.parent / "magicquant" / "data" / "questions.yaml"
)

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_FAILURE_MODES = {
    "arithmetic", "factual_recall", "multilingual", "long_context",
    "multi_hop", "code", "proof", "instruction_following",
}
VALID_SCORING_TYPES = {
    "exact_numeric", "quadratic_roots", "keyphrase", "code_syntax", "none",
}


@pytest.fixture(scope="module")
def questions():
    pytest.importorskip("yaml")
    import yaml
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["questions"]


def test_file_loads(questions):
    assert len(questions) > 0


def test_question_count_20(questions):
    assert len(questions) == 20


def test_all_have_required_fields(questions):
    required = {"id", "difficulty", "failure_mode", "scoring_type", "prompt"}
    for q in questions:
        missing = required - q.keys()
        assert not missing, f"Q{q.get('id')} missing fields: {missing}"


def test_ids_unique_and_sequential(questions):
    ids = [q["id"] for q in questions]
    assert ids == list(range(1, 21)), f"IDs not 1-20: {ids}"


def test_all_difficulties_valid(questions):
    for q in questions:
        assert q["difficulty"] in VALID_DIFFICULTIES, (
            f"Q{q['id']} has invalid difficulty: {q['difficulty']!r}"
        )


def test_difficulty_distribution(questions):
    easy   = sum(1 for q in questions if q["difficulty"] == "easy")
    medium = sum(1 for q in questions if q["difficulty"] == "medium")
    hard   = sum(1 for q in questions if q["difficulty"] == "hard")
    assert easy == 5,  f"Expected 5 easy, got {easy}"
    assert medium == 7, f"Expected 7 medium, got {medium}"
    assert hard == 8,  f"Expected 8 hard, got {hard}"


def test_all_failure_modes_valid(questions):
    for q in questions:
        assert q["failure_mode"] in VALID_FAILURE_MODES, (
            f"Q{q['id']} has invalid failure_mode: {q['failure_mode']!r}"
        )


def test_all_scoring_types_valid(questions):
    for q in questions:
        assert q["scoring_type"] in VALID_SCORING_TYPES, (
            f"Q{q['id']} has invalid scoring_type: {q['scoring_type']!r}"
        )


def test_scored_questions_have_ground_truth(questions):
    # code_syntax verifies syntax only — no ground_truth needed
    no_gt_needed = {"none", "code_syntax"}
    for q in questions:
        if q["scoring_type"] not in no_gt_needed:
            assert q.get("ground_truth") is not None, (
                f"Q{q['id']} scored as {q['scoring_type']} but has no ground_truth"
            )


def test_exact_numeric_ground_truth_is_numeric(questions):
    for q in questions:
        if q["scoring_type"] == "exact_numeric":
            gt = q.get("ground_truth")
            try:
                float(gt)
            except (TypeError, ValueError):
                pytest.fail(f"Q{q['id']} exact_numeric ground_truth is not a number: {gt!r}")


def test_passage_files_exist(questions):
    data_dir = QUESTIONS_FILE.parent
    for q in questions:
        pf = q.get("passage_file")
        if pf:
            full_path = data_dir / pf
            assert full_path.exists(), f"Q{q['id']} passage file missing: {full_path}"


def test_catalog_math_q20(questions):
    """Verify ground truth for Q20 (catalog purchase) is correct."""
    q20 = next(q for q in questions if q["id"] == 20)
    assert q20["ground_truth"] == 847, (
        "3 × $127 + 2 × $233 = $381 + $466 = $847"
    )


def test_train_problem_q6(questions):
    """Verify Q6 (train meeting) ground truth is 72 minutes."""
    q6 = next(q for q in questions if q["id"] == 6)
    assert q6["ground_truth"] == 72


def test_russian_train_q11(questions):
    """Verify Q11 (Russian train) ground truth is 1.6 hours."""
    q11 = next(q for q in questions if q["id"] == 11)
    assert math.isclose(float(q11["ground_truth"]), 1.6, abs_tol=0.01)


def test_bat_and_ball_q8(questions):
    """Verify Q8 (bat and ball) ground truth is $0.05."""
    q8 = next(q for q in questions if q["id"] == 8)
    assert math.isclose(float(q8["ground_truth"]), 0.05, abs_tol=0.001)


def test_quadratic_q13_roots(questions):
    """Verify Q13 (quadratic) roots sum to 7/3 and product to 2/3."""
    q13 = next(q for q in questions if q["id"] == 13)
    roots = [float(x) for x in q13["ground_truth"]]
    assert len(roots) == 2
    assert math.isclose(sum(roots), 7 / 3, abs_tol=0.01), "sum of roots should = 7/3"
    assert math.isclose(roots[0] * roots[1], 2 / 3, abs_tol=0.01), "product of roots = 2/3"
