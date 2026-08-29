"""
Automated scoring engine for magicquant compare.

Each scorer returns a ScoreResult with status: pass | near_miss | fail | unscored.
near_miss is informational (shown in output) but counted as fail in summaries.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ScoreResult:
    status: str       # "pass" | "near_miss" | "fail" | "unscored"
    detail: str       # human-readable explanation
    extracted: Any    # what was pulled from the response
    expected: Any     # ground truth for display


# ── Think-block stripping ──────────────────────────────────────────────────────

def _strip_think(text: str) -> tuple[str, bool]:
    """Return (scorable_text, was_truncated_in_think).

    Thinking models (e.g. Qwen3, DeepSeek-R1) wrap chain-of-thought in
    <think>...</think>. Scorers should operate on the post-think answer only.

    - If </think> is present: return text after the last </think>.
    - If <think> is present but </think> is absent: the response was truncated
      inside the think block (token budget exhausted). Return ("", True).
    - If no think tags at all: return (text, False) unchanged.
    """
    lower = text.lower()
    if "<think>" not in lower:
        return text, False
    if "</think>" not in lower:
        return "", True
    split_point = lower.rfind("</think>") + len("</think>")
    return text[split_point:].strip(), False


# ── Numeric extraction ─────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")
_FRACTION_RE = re.compile(r"-?\d+\s*/\s*\d+")


def extract_last_number(text: str) -> Optional[float]:
    """Return the last number in *text*, scanning backwards line by line.

    Scans from the bottom of the text upward so that an incomplete trailing
    line (e.g. "The answer is" with no number) is skipped in favour of the
    last complete line that contains a number.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    for line in reversed(lines):
        if not line:
            continue
        matches = _NUMBER_RE.findall(line)
        if matches:
            for raw in reversed(matches):
                try:
                    return float(raw.replace(",", ""))
                except ValueError:
                    continue
    return None


def _extract_all_numbers_and_fractions(text: str) -> list[float]:
    """Return all numbers and simple fractions (a/b) found in *text*."""
    results: list[float] = []
    # Replace fractions first so they don't get double-parsed as two numbers
    remainder = text
    for m in _FRACTION_RE.finditer(text):
        parts = m.group().split("/")
        try:
            results.append(float(parts[0].strip()) / float(parts[1].strip()))
        except (ValueError, ZeroDivisionError):
            pass
        remainder = remainder.replace(m.group(), " ")
    for m in _NUMBER_RE.finditer(remainder):
        try:
            results.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return results


# ── Individual scorers ─────────────────────────────────────────────────────────

def score_exact_numeric(response: str, question: dict) -> ScoreResult:
    """Extract last number, compare to ground_truth with tolerance."""
    ground_truth = float(question["ground_truth"])
    tol = question.get("tolerance") or {}
    abs_tol = float(tol.get("abs", 0))
    rel_tol = float(tol.get("rel", 0))

    scorable, truncated = _strip_think(response)
    if truncated:
        return ScoreResult("fail", "Response truncated inside <think> block",
                           None, ground_truth)

    extracted = extract_last_number(scorable)
    if extracted is None:
        return ScoreResult("fail", "No numeric answer found", None, ground_truth)

    if math.isclose(extracted, ground_truth, rel_tol=rel_tol, abs_tol=abs_tol):
        return ScoreResult("pass", f"Answer: {extracted}", extracted, ground_truth)

    # near_miss: use 5% relative band when tolerance is zero; otherwise 2× tolerance
    if abs_tol == 0 and rel_tol == 0:
        band = 0.05 * abs(ground_truth) if ground_truth != 0 else 0.05
        near = abs(extracted - ground_truth) <= band
    else:
        near = math.isclose(extracted, ground_truth,
                            rel_tol=rel_tol * 2, abs_tol=abs_tol * 2)

    diff = extracted - ground_truth
    detail = f"Answer: {extracted} (expected {ground_truth}, off by {diff:+.4g})"
    return ScoreResult("near_miss" if near else "fail", detail, extracted, ground_truth)


def score_quadratic_roots(response: str, question: dict) -> ScoreResult:
    """Check that both expected roots appear in the response within tolerance."""
    expected: list[float] = [float(x) for x in question["ground_truth"]]
    tol = question.get("tolerance") or {}
    abs_tol = float(tol.get("abs", 0.01))

    scorable, truncated = _strip_think(response)
    if truncated:
        return ScoreResult("fail", "Response truncated inside <think> block",
                           [], expected)

    candidates = _extract_all_numbers_and_fractions(scorable)
    if not candidates:
        return ScoreResult("fail", "No numbers found in response", [], expected)

    matched_expected = []
    for exp_root in expected:
        found = any(abs(c - exp_root) <= abs_tol for c in candidates)
        if found:
            matched_expected.append(exp_root)

    if len(matched_expected) == 2:
        # Both roots present — pass regardless of extra working numbers.
        # Showing algebraic steps is expected behaviour for this question type.
        return ScoreResult("pass", "Both roots found", candidates, expected)

    missing = [e for e in expected if e not in matched_expected]
    return ScoreResult(
        "fail",
        f"Missing root(s): {missing} — found {candidates}",
        candidates, expected,
    )


def score_keyphrase(response: str, question: dict) -> ScoreResult:
    """All required keyphrases must appear (case-insensitive substring)."""
    gt = question["ground_truth"]
    phrases = [gt] if isinstance(gt, str) else list(gt)

    scorable, truncated = _strip_think(response)
    if truncated:
        return ScoreResult("fail", "Response truncated inside <think> block",
                           [], phrases)

    lower = scorable.lower()
    missing = [p for p in phrases if p.lower() not in lower]
    if not missing:
        return ScoreResult("pass", f"All keyphrases found: {phrases}", phrases, phrases)
    return ScoreResult(
        "fail",
        f"Missing: {missing}",
        [p for p in phrases if p not in missing],
        phrases,
    )


def score_code_syntax(response: str, question: dict) -> ScoreResult:
    """Extract Python code and verify it parses with ast.parse."""
    # Strip think block; ignore truncation flag (partial think = no code to check)
    scorable, _ = _strip_think(response)

    # Try fenced ```python ... ``` blocks first
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", scorable, re.DOTALL | re.IGNORECASE)
    candidates = [b.strip() for b in fenced if b.strip()]

    # Fallback: try the entire post-think text
    if not candidates:
        candidates = [scorable.strip()]

    for code in candidates:
        try:
            ast.parse(code)
            return ScoreResult("pass", "Syntax valid", code[:80], None)
        except SyntaxError as exc:
            return ScoreResult(
                "fail",
                f"SyntaxError at line {exc.lineno}: {exc.msg}",
                code[:80], None,
            )

    return ScoreResult("fail", "No code block found", None, None)


def score_none(_response: str, _question: dict) -> ScoreResult:
    return ScoreResult("unscored", "Manual review required", None, None)


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_SCORERS = {
    "exact_numeric":   score_exact_numeric,
    "quadratic_roots": score_quadratic_roots,
    "keyphrase":       score_keyphrase,
    "code_syntax":     score_code_syntax,
    "none":            score_none,
}


def score_response(response: str, question: dict) -> ScoreResult:
    """Dispatch to the appropriate scorer based on question['scoring_type']."""
    scorer = _SCORERS.get(question.get("scoring_type", "none"), score_none)
    return scorer(response, question)


# ── Consistency ────────────────────────────────────────────────────────────────

def compute_consistency(scored_samples: list[ScoreResult]) -> float:
    """Fraction of samples sharing the majority status."""
    if not scored_samples:
        return 1.0
    statuses = [s.status for s in scored_samples]
    most_common = max(set(statuses), key=statuses.count)
    return statuses.count(most_common) / len(statuses)
