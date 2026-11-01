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


_THOUSANDS_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _parse_number(raw: str) -> float:
    """Parse a matched number, telling thousands separators ("1,191.02",
    "12,000") from a decimal comma ("1,6", the Russian/European style).
    Comma groups of exactly three digits are thousands; a single other
    comma with no decimal point is a decimal comma."""
    if _THOUSANDS_RE.fullmatch(raw):
        return float(raw.replace(",", ""))
    if raw.count(",") == 1 and "." not in raw:
        return float(raw.replace(",", "."))
    return float(raw.replace(",", ""))


# A declared final answer ("Answer: 0.05", "Ответ: 1.6", "**Answer:** $847").
# Prompts for exact_numeric questions ask for this line; when present it beats
# the last-number heuristic, which misreads restatements like "$0.05 or 5 cents".
_ANSWER_LINE_RE = re.compile(
    r"(?:answer|ответ|پاسخ|答案)\s*\**\s*[:：]\s*\**\s*[$€£]?\s*(-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def extract_declared_answer(text: str) -> Optional[float]:
    """Return the number on the LAST "Answer: <number>" line, or None."""
    matches = _ANSWER_LINE_RE.findall(text)
    for raw in reversed(matches):
        try:
            return _parse_number(raw)
        except ValueError:
            continue
    return None


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
                    return _parse_number(raw)
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

_CENTS_RE = re.compile(r"(?<![\d.$])(\d+(?:\.\d+)?)\s*(?:cents?|¢)", re.IGNORECASE)


def _cents_to_dollars(text: str) -> str:
    """Rewrite "5 cents" as "$0.05" so a dollar answer restated in cents
    ("$0.05 or 5 cents") is read in the question's unit."""
    return _CENTS_RE.sub(lambda m: f"${float(m.group(1)) / 100:g}", text)


def score_exact_numeric(response: str, question: dict) -> ScoreResult:
    """Extract the declared "Answer: <n>" (else the last number) and compare
    to ground_truth with tolerance."""
    ground_truth = float(question["ground_truth"])
    tol = question.get("tolerance") or {}
    abs_tol = float(tol.get("abs", 0))
    rel_tol = float(tol.get("rel", 0))

    scorable, truncated = _strip_think(response)
    if truncated:
        return ScoreResult("fail", "Response truncated inside <think> block",
                           None, ground_truth)

    if question.get("unit") == "dollars":
        scorable = _cents_to_dollars(scorable)
    extracted = extract_declared_answer(scorable)
    if extracted is None:
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

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _run_sql_case(query: str, setup: str, rows: list, timeout_s: float = 2.0):
    """Run *query* against a fresh in-memory DB built from *setup* + *rows*.

    Model output is untrusted: the authorizer denies ATTACH/DETACH (the only
    way SQLite SQL reaches the filesystem) and any write to the table, and a
    progress handler aborts runaway queries. Returns one set of distinct
    values per result column ([{None}] for an empty result), or raises.
    """
    import sqlite3
    import time

    db = sqlite3.connect(":memory:")
    try:
        db.execute(setup)
        if rows:
            marks = ",".join("?" * len(rows[0]))
            table = re.search(r"CREATE\s+TABLE\s+(\w+)", setup, re.I).group(1)
            db.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)

        denied = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
                  sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                  sqlite3.SQLITE_DELETE, sqlite3.SQLITE_DROP_TABLE,
                  sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_ALTER_TABLE}
        db.set_authorizer(
            lambda action, *a: sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK
        )
        deadline = time.monotonic() + timeout_s
        db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        result = db.execute(query).fetchall()
    finally:
        db.close()
    if not result:
        return [{None}]
    return [{row[i] for row in result} for i in range(len(result[0]))]


def score_sql_exec(response: str, question: dict) -> ScoreResult:
    """Execute the answer's SQL against each ground_truth case.

    ground_truth: list of {rows: [[...], ...], expected: value-or-null};
    sql_setup: the CREATE TABLE statement. A case passes when the query's
    result has a column whose values are exactly {expected} (None = no
    row, or a column of NULLs). Any column, so "SELECT salary, rank ..." and
    "SELECT MAX(salary)" are judged alike.
    """
    scorable, truncated = _strip_think(response)
    cases = question["ground_truth"]
    if truncated:
        return ScoreResult("fail", "Response truncated inside <think> block", None, cases)
    blocks = [b.strip() for b in _SQL_FENCE_RE.findall(scorable) if b.strip()]
    query = next((b for b in blocks if re.search(r"\bselect\b", b, re.I)), None)
    if query is None:
        return ScoreResult("fail", "No SQL query found", None, cases)
    query = query.rstrip().rstrip(";")
    if ";" in query:
        return ScoreResult("fail", "Multiple SQL statements; expected one query", query, cases)

    failures = []
    for i, case in enumerate(cases, 1):
        want = case.get("expected")
        label = case.get("label", f"case {i}")
        try:
            got = _run_sql_case(query, question["sql_setup"], case.get("rows") or [])
        except Exception as exc:  # noqa: BLE001 -- untrusted SQL, any error fails
            failures.append(f"{label}: SQL error: {exc}")
            continue
        if {want} not in got:
            shown = [sorted(c, key=str) for c in got]
            failures.append(f"{label}: got columns {shown}, expected {want!r}")
    if failures:
        return ScoreResult(
            "fail",
            f"{len(cases) - len(failures)}/{len(cases)} cases correct; " + "; ".join(failures),
            query, cases,
        )
    return ScoreResult("pass", f"All {len(cases)} cases correct", query, cases)


_SCORERS = {
    "exact_numeric":   score_exact_numeric,
    "quadratic_roots": score_quadratic_roots,
    "keyphrase":       score_keyphrase,
    "code_syntax":     score_code_syntax,
    "sql_exec":        score_sql_exec,
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
