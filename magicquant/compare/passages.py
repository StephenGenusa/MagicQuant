"""Passage file loading for long-context questions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_passage(question: dict, yaml_dir: Path) -> Optional[str]:
    """Load passage text for *question*, resolved relative to *yaml_dir*.

    Returns None if question has no passage_file.
    Raises FileNotFoundError if the path is set but the file doesn't exist.
    """
    passage_file = question.get("passage_file")
    if not passage_file:
        return None
    path = yaml_dir / passage_file
    if not path.exists():
        raise FileNotFoundError(
            f"Passage file not found: {path}  (question id={question.get('id')})"
        )
    return path.read_text(encoding="utf-8")


def estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 3 characters (conservative for dense text)."""
    return max(1, len(text) // 3)


def build_prompt(question: dict, passage: Optional[str]) -> str:
    """Prepend passage to question prompt if present."""
    if passage:
        return f"{passage}\n\n---\n\n{question['prompt']}"
    return question["prompt"]
