"""Tests for magicquant.compare.passages"""

import pytest
from pathlib import Path

from magicquant.compare.passages import build_prompt, estimate_tokens, load_passage


DATA_DIR = Path(__file__).parent.parent.parent / "magicquant" / "data"


# ── load_passage ───────────────────────────────────────────────────────────────

def test_load_passage_none_when_no_passage_file():
    q = {"id": 1, "prompt": "test"}
    result = load_passage(q, DATA_DIR)
    assert result is None


def test_load_passage_easy_bakery():
    q = {"id": 5, "prompt": "test", "passage_file": "passages/easy_bakery.txt"}
    text = load_passage(q, DATA_DIR)
    assert text is not None
    assert "Golden Loaf" in text
    assert "Oak Street" in text


def test_load_passage_hard_catalog():
    q = {"id": 20, "prompt": "test", "passage_file": "passages/hard_catalog.txt"}
    text = load_passage(q, DATA_DIR)
    assert text is not None
    assert "Aardvark Widget" in text
    assert "$127.00" in text
    assert "Mongoose Widget" in text
    assert "$233.00" in text


def test_load_passage_missing_file_raises():
    q = {"id": 99, "prompt": "test", "passage_file": "passages/nonexistent.txt"}
    with pytest.raises(FileNotFoundError):
        load_passage(q, DATA_DIR)


# ── estimate_tokens ────────────────────────────────────────────────────────────

def test_estimate_tokens_basic():
    # 1 token ≈ 3 chars, minimum 1
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 300) == 100

def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1

def test_estimate_tokens_short():
    assert estimate_tokens("hi") == 1


# ── build_prompt ───────────────────────────────────────────────────────────────

def test_build_prompt_no_passage():
    q = {"prompt": "What is 2+2?"}
    result = build_prompt(q, None)
    assert result == "What is 2+2?"


def test_build_prompt_with_passage():
    q = {"prompt": "What is the bakery name?"}
    passage = "On Oak Street there is a bakery called Golden Loaf."
    result = build_prompt(q, passage)
    assert result.startswith(passage)
    assert "What is the bakery name?" in result
    assert "---" in result


def test_build_prompt_with_passage_order():
    q = {"prompt": "QUESTION"}
    passage = "PASSAGE"
    result = build_prompt(q, passage)
    assert result.index("PASSAGE") < result.index("QUESTION")
