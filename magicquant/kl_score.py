"""Score every GGUF in a directory against ONE reference's logits.

``magicquant kl-score`` writes ``<output-dir>/kl_scores.json``: perplexity and
KL divergence (mean, median, p99, p99.9, RMS Δp, same-top-token) for each
file, all measured by the same llama.cpp build on the same corpus, context
and reference logits. That makes MagicQuant tiers and other quantizers' files
(dropped into the directory for ``compare``) directly comparable; ``compare``
and ``card`` read the file when present.

The reference logits are large (vocab x tokens x 2 bytes: ~37 GB for a
128k-vocab model on wikitext-2 at ctx 512). Pass ``base_logits`` to reuse an
existing file. A file this module creates itself is deleted afterwards unless
``keep_base_logits`` is set; a file passed in is never deleted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

KL_SCORES_FILE = "kl_scores.json"


def score_directory(
    output_dir,
    reference: str,
    *,
    llamacpp_path: Optional[str] = None,
    base_logits: Optional[str] = None,
    keep_base_logits: bool = False,
    ctx_size: int = 512,
    verbose: bool = True,
) -> Dict[str, Any]:
    from magicquant.utils.llamacpp import LlamaCppTools

    out = Path(output_dir)
    ref = Path(reference)
    if not ref.is_file():
        raise FileNotFoundError(f"reference model not found: {ref}")
    files = sorted(p for p in out.glob("*.gguf") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no *.gguf files in {out}")

    tools = LlamaCppTools(llamacpp_path=llamacpp_path, ctx_size=ctx_size)
    corpus = tools._resolve_data_file(None)

    created = False
    if base_logits:
        logits = Path(base_logits)
        if not logits.is_file():
            raise FileNotFoundError(f"base logits not found: {logits}")
        if verbose:
            print(f"Reusing reference logits: {logits}")
    else:
        logits = out / "_kl_score_base_logits.kld"
        if verbose:
            print(f"Saving reference logits from {ref.name} -> {logits}")
        if tools.save_base_logits(str(ref), corpus, str(logits), ctx_size=ctx_size) is None:
            raise RuntimeError("could not save reference logits (see llama.cpp output)")
        created = True

    scores: Dict[str, Any] = {}
    try:
        for f in files:
            t0 = time.time()
            kl = tools.calculate_kl_divergence(str(f), str(logits), corpus, ctx_size=ctx_size)
            entry: Dict[str, Any] = {"bytes": f.stat().st_size}
            if kl is None:
                entry["error"] = "KL measurement failed"
            else:
                entry.update(kl)
            scores[f.name] = entry
            if verbose:
                shown = (f"PPL {kl.get('ppl')}, mean KLD {kl.get('mean_kl')}"
                         if kl else "FAILED")
                print(f"  {f.name}: {shown} ({time.time() - t0:.0f}s)")
    finally:
        if created and not keep_base_logits:
            logits.unlink(missing_ok=True)

    result = {
        "reference": ref.name,
        "corpus": Path(corpus).name if corpus else None,
        "ctx_size": ctx_size,
        "chunks": getattr(tools, "ppl_chunks", None),
        "llamacpp_binary": getattr(tools, "perplexity_tool", None),
        "files": scores,
    }
    path = out / KL_SCORES_FILE
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        print(f"Wrote {path}")
    return result


def load_kl_scores(output_dir) -> Optional[Dict[str, Any]]:
    """The directory's kl_scores.json, or None when absent/unreadable."""
    path = Path(output_dir) / KL_SCORES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("files"), dict) else None
