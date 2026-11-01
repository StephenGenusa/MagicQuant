"""Run provenance: which MagicQuant produced an artifact, and how.

Stamped into ``search_results.json`` as the additive ``run_info`` key so a
published model card can name the exact tool version behind its numbers.
Best-effort throughout: a missing git checkout or package metadata yields
``None`` for that field, never an exception.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def magicquant_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("magicquant")
    except Exception:
        return None


def git_commit() -> Optional[str]:
    """Short commit of the checkout this package runs from, or None when it
    is not a git checkout (e.g. an installed wheel)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def run_info() -> Dict[str, object]:
    return {
        "magicquant_version": magicquant_version(),
        "git_commit": git_commit(),
        "argv": list(sys.argv),
    }
