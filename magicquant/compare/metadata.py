"""Reproducibility metadata collection for magicquant compare."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ReproMetadata:
    timestamp: str
    magicquant_version: str
    git_commit: str
    cli_args: str
    questions_file: str
    questions_sha256: str
    question_count: int
    pool_size: int
    difficulty_breakdown: dict
    system_prompt: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    context_size: int
    n_samples: int
    llamacpp_path: Optional[str]
    skipped_questions: list = field(default_factory=list)


def collect_metadata(
    args: argparse.Namespace,
    questions: list,
    pool_size: int,
    questions_file: Path,
) -> ReproMetadata:
    from datetime import datetime
    import magicquant

    difficulty_breakdown = {
        "easy":   sum(1 for q in questions if q.get("difficulty") == "easy"),
        "medium": sum(1 for q in questions if q.get("difficulty") == "medium"),
        "hard":   sum(1 for q in questions if q.get("difficulty") == "hard"),
    }

    return ReproMetadata(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        magicquant_version=getattr(magicquant, "__version__", "unknown"),
        git_commit=_get_git_commit(),
        cli_args=_reconstruct_cli_args(args),
        questions_file=str(questions_file),
        questions_sha256=_hash_file(questions_file),
        question_count=len(questions),
        pool_size=pool_size,
        difficulty_breakdown=difficulty_breakdown,
        system_prompt=getattr(args, "system_prompt", ""),
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", 0),
        max_tokens=args.max_tokens,
        context_size=args.context_size,
        n_samples=getattr(args, "n_samples", 1),
        llamacpp_path=getattr(args, "llamacpp_path", None),
    )


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def _reconstruct_cli_args(args: argparse.Namespace) -> str:
    parts = ["compare"]
    mapping = {
        "output_dir":        "--output-dir",
        "questions_file":    "--questions-file",
        "question_count":    "--question-count",
        "max_tokens":        "--max-tokens",
        "context_size":      "--context-size",
        "n_samples":         "--n-samples",
        "temperature":       "--temperature",
        "top_p":             "--top-p",
        "top_k":             "--top-k",
        "llamacpp_path":     "--llamacpp-path",
    }
    for attr, flag in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            parts.append(f"{flag} {val}")
    if getattr(args, "reasoning_mode", False):
        parts.append("--reasoning-mode")
    return " ".join(parts)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
