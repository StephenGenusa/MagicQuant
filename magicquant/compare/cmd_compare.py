"""
magicquant compare — New feature by Stephen Genusa.

Scans an output directory for GGUF files, runs a stratified question pool
through each model, scores responses automatically, and writes HTML/Markdown
comparison tables with per-failure-mode breakdowns.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

from magicquant.compare.metadata import collect_metadata
from magicquant.compare.output import (
    InferenceResult,
    ModelInfo,
    generate_html,
    generate_markdown,
    populate_scores,
    write_meta_json,
    write_raw_response,
)
from magicquant.compare.inference import run_inference_batch
from magicquant.compare.passages import build_prompt, estimate_tokens, load_passage
from magicquant.compare.scoring import compute_consistency, score_response


# ── Default prompts ────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and precise assistant. "
    "Answer the following question concisely and accurately."
)

_REASONING_SYSTEM_PROMPT = (
    "You are a precise and methodical reasoner. "
    "Think step by step and show all your work, then give a clear final answer."
)


# ── GGUF discovery ─────────────────────────────────────────────────────────────

def _discover_models(output_dir: Path) -> list[ModelInfo]:
    from magicquant.gguf.reader import GGUFReader
    import re as _re

    models: list[ModelInfo] = []
    for path in sorted(output_dir.glob("*.gguf")):
        try:
            reader = GGUFReader(str(path))
            reader.open()
            arch = reader.get_model_architecture() or "unknown"
            size_gb = reader.get_file_size_gb()
            bpw = reader.get_bits_per_weight()
            params = reader.get_parameter_count()
            reader.close()
        except Exception as exc:
            print(f"  Warning: could not read {path.name}: {exc}")
            continue

        # Derive quant_type from filename
        m = _re.search(
            r"\b(Q\d_K_[SM]|Q\d_K|Q\d_\d|IQ\d_\w+|BF16|F16|F32|MXFP4)\b",
            path.name, _re.IGNORECASE,
        )
        quant_type = m.group(1).upper() if m else "—"

        models.append(ModelInfo(
            path=path,
            filename=path.name,
            bpw=bpw,
            quant_type=quant_type,
            param_count=params,
            architecture=arch,
            size_gb=size_gb,
        ))

    models.sort(key=lambda m: m.bpw, reverse=True)
    return models


def _enrich_ppl(models: list[ModelInfo], output_dir: Path) -> None:
    import json
    results_file = output_dir / "search_results.json"
    if not results_file.exists():
        return
    try:
        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    tier_ppl = {
        tier: float(info["ppl"])
        for tier, info in data.get("tiered", {}).items()
        if isinstance(info, dict) and "ppl" in info
    }
    baseline = data.get("baseline_ppl")
    for model in models:
        for tier, ppl in tier_ppl.items():
            if tier in model.filename or tier.replace("_", "") in model.filename.replace("_", ""):
                model.ppl = ppl
                break
        if model.ppl is None and baseline and model.bpw >= 14.0:
            model.ppl = float(baseline)


# ── Question loading & sampling ────────────────────────────────────────────────

def _load_questions(questions_file: Path) -> list[dict]:
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml required.  Install: pip install \"magicquant[yaml]\"")
        sys.exit(1)
    with open(questions_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("questions", [])


def _stratify_count(n: int) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 0, 0, 1
    if n == 2:
        return 0, 1, 1
    n_easy = max(1, round(n * 0.2))
    n_medium = max(1, round(n * 0.4))
    n_hard = n - n_easy - n_medium
    if n_hard < 1:
        n_medium -= 1
        n_hard = 1
    return n_easy, n_medium, n_hard


def _select_questions(all_q: list[dict], count: int) -> list[dict]:
    by_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for q in all_q:
        bucket = q.get("difficulty", "hard")
        if bucket not in by_diff:
            bucket = "hard"
        by_diff[bucket].append(q)

    n_easy, n_medium, n_hard = _stratify_count(count)
    easy = by_diff["easy"][:n_easy]
    medium = by_diff["medium"][:n_medium]
    hard_pool = by_diff["hard"]
    hard = hard_pool[max(0, len(hard_pool) - n_hard):]

    shortage = count - (len(easy) + len(medium) + len(hard))
    if shortage > 0:
        extra_m = by_diff["medium"][n_medium:n_medium + shortage]
        medium = medium + extra_m
        shortage -= len(extra_m)
        if shortage > 0:
            extra_e = by_diff["easy"][n_easy:n_easy + shortage]
            easy = easy + extra_e

    return (easy + medium + hard)[:count]


# ── Context size calculation ───────────────────────────────────────────────────

def _compute_min_ctx(
    questions: list[dict],
    yaml_dir: Path,
    max_tokens: int,
    system_prompt: str,
) -> int:
    sys_tokens = estimate_tokens(system_prompt)
    min_ctx = 0
    for q in questions:
        try:
            passage = load_passage(q, yaml_dir)
        except FileNotFoundError:
            passage = None
        prompt_text = build_prompt(q, passage)
        needed = estimate_tokens(prompt_text) + sys_tokens + max_tokens + 256
        min_ctx = max(min_ctx, needed)
    return min_ctx


# ── Per-model inference ────────────────────────────────────────────────────────

def _run_model(
    model: ModelInfo,
    questions: list[dict],
    yaml_dir: Path,
    max_tokens: int,
    ctx_size: int,
    global_system_prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[int, InferenceResult]:
    # Preload passages once per model
    passages: list[Optional[str]] = []
    for q in questions:
        try:
            passages.append(load_passage(q, yaml_dir))
        except FileNotFoundError as exc:
            print(f"  Warning: {exc}")
            passages.append(None)

    # Group by effective system prompt so each model load handles questions
    # with system_prompt_override in a separate mini-batch if needed.
    from collections import defaultdict
    groups: dict[str, list[tuple[int, dict, str]]] = defaultdict(list)
    for idx, (q, passage) in enumerate(zip(questions, passages)):
        eff_sys = q.get("system_prompt_override") or global_system_prompt
        full_prompt = build_prompt(q, passage)
        groups[eff_sys].append((idx, q, full_prompt))

    results: dict[int, InferenceResult] = {}

    for eff_sys, group_items in groups.items():
        idxs = [item[0] for item in group_items]
        qs = [item[1] for item in group_items]
        prompt_texts = [item[2] for item in group_items]

        batch = run_inference_batch(
            model_path=str(model.path),
            prompts=prompt_texts,
            max_tokens=max_tokens,
            ctx_size=ctx_size,
            system_prompt=eff_sys,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        for idx, q, samples in zip(idxs, qs, batch):
            cleaned = [s if s is not None else "(no response)" for s in samples]
            scored = [score_response(s, q) for s in cleaned]
            consistency = compute_consistency(scored)
            primary_status = scored[0].status if scored else "unscored"
            results[idx] = InferenceResult(
                samples=cleaned,
                scored_samples=scored,
                consistency=consistency,
                primary_status=primary_status,
                primary_response=cleaned[0] if cleaned else "(no response)",
            )

    return results


# ── Output directory management ────────────────────────────────────────────────

def _make_run_dir(output_dir: Path, timestamp: str) -> Path:
    safe_ts = timestamp.replace(":", "-")
    run_dir = output_dir / "comparisons" / safe_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update `comparison_latest` symlink
    latest = output_dir / "comparison_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(run_dir.relative_to(output_dir))
    except Exception:
        pass  # Symlinks may fail on some OS/filesystem configs

    return run_dir


# ── Main entry point ───────────────────────────────────────────────────────────

def cmd_compare(args: Namespace) -> None:
    from datetime import datetime

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    # ── Discover GGUFs ────────────────────────────────────────────────────────
    print(f"Scanning {output_dir} for GGUF files...")
    models = _discover_models(output_dir)
    if not models:
        print(f"No GGUF files found in {output_dir}")
        sys.exit(1)

    _enrich_ppl(models, output_dir)
    print(f"Found {len(models)} model(s) (best → worst quality):")
    for m in models:
        ppl_str = f"  PPL={m.ppl:.4f}" if m.ppl else ""
        print(f"  {m.filename}  ({m.bpw:.2f} bpw, {m.size_gb:.2f} GB{ppl_str})")

    # ── Load questions ────────────────────────────────────────────────────────
    if args.questions_file:
        questions_path = Path(args.questions_file)
    else:
        questions_path = Path(__file__).parent.parent / "data" / "questions.yaml"
    yaml_dir = questions_path.parent

    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_path}")
        sys.exit(1)

    all_questions = _load_questions(questions_path)
    pool_size = len(all_questions)
    count = min(args.question_count, pool_size)
    questions = _select_questions(all_questions, count)

    diff_counts = {
        "easy":   sum(1 for q in questions if q.get("difficulty") == "easy"),
        "medium": sum(1 for q in questions if q.get("difficulty") == "medium"),
        "hard":   sum(1 for q in questions if q.get("difficulty") == "hard"),
    }
    print(
        f"\nQuestion pool: {pool_size} total, using {count} "
        f"({diff_counts['easy']} easy / {diff_counts['medium']} medium / {diff_counts['hard']} hard)"
    )

    # ── Resolve inference parameters ─────────────────────────────────────────
    n_samples = getattr(args, "n_samples", 1)
    temperature = getattr(args, "temperature", 0.0)
    top_p = getattr(args, "top_p", 1.0)
    top_k = getattr(args, "top_k", 0)
    reasoning_mode = getattr(args, "reasoning_mode", False)
    user_system_prompt = getattr(args, "system_prompt", None)

    if user_system_prompt:
        system_prompt = user_system_prompt
    elif reasoning_mode:
        system_prompt = _REASONING_SYSTEM_PROMPT
    else:
        system_prompt = _DEFAULT_SYSTEM_PROMPT

    # Pre-compute minimum context size and enforce it
    min_ctx = _compute_min_ctx(questions, yaml_dir, args.max_tokens, system_prompt)
    ctx_size = max(args.context_size, min_ctx)
    if ctx_size > args.context_size:
        print(f"  Context expanded to {ctx_size} tokens (passages need it)")

    # ── Check inference backend ───────────────────────────────────────────────
    try:
        from llama_cpp import Llama as _Llama  # noqa: F401
    except ImportError:
        print("Error: llama-cpp-python not installed.")
        print("  Install with: pip install llama-cpp-python")
        sys.exit(1)

    print(f"System prompt: \"{system_prompt[:80]}\"")
    print(f"Max tokens: {args.max_tokens}  Context: {ctx_size}  n_samples: {n_samples}")
    if n_samples > 1 and temperature == 0:
        print("  Warning: n_samples > 1 with temperature=0 — all samples will be identical")
    print()

    # ── Collect metadata ──────────────────────────────────────────────────────
    meta = collect_metadata(args, questions, pool_size, questions_path)
    # Override system_prompt in metadata with resolved value
    meta.system_prompt = system_prompt

    # ── Run inference ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = _make_run_dir(output_dir, timestamp)

    all_results: dict[str, dict[int, InferenceResult]] = {}

    for mi, model in enumerate(models):
        print(f"[Model {mi+1}/{len(models)}] {model.filename}")

        # results indexed by position in `questions` list
        by_idx = _run_model(
            model=model,
            questions=questions,
            yaml_dir=yaml_dir,
            max_tokens=args.max_tokens,
            ctx_size=ctx_size,
            global_system_prompt=system_prompt,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Remap to question-id for output
        by_qid: dict[int, InferenceResult] = {}
        for idx, result in by_idx.items():
            q = questions[idx]
            qid = q["id"]
            by_qid[qid] = result

            # Progress print
            r = result.primary_response
            preview = r[:80].replace("\n", " ")
            print(
                f"  [Q{qid:02d} {q.get('difficulty','?')[0].upper()}] "
                f"{q.get('failure_mode','?')}: "
                f"{result.primary_status.upper()} — {preview}"
                f"{'...' if len(r) > 80 else ''}"
            )

            write_raw_response(model.filename, q, result, run_dir)

        populate_scores(model, questions, {q["id"]: by_qid.get(q["id"]) for q in questions if q["id"] in by_qid})
        all_results[model.filename] = by_qid

        pct = model.total_score / model.total_questions * 100 if model.total_questions else 0
        print(f"  Score: {model.total_score}/{model.total_questions} ({pct:.0f}%)  "
              f"Consistency: {model.consistency_score:.2f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_meta_json(meta, run_dir)

    html_path = run_dir / "comparison.html"
    md_path = run_dir / "comparison.md"

    generate_html(
        models=models,
        questions=questions,
        results=all_results,
        metadata=meta,
        output_path=html_path,
        n_samples=n_samples,
    )
    generate_markdown(
        models=models,
        questions=questions,
        results=all_results,
        metadata=meta,
        output_path=md_path,
        n_samples=n_samples,
    )

    print("\nComparison written to:")
    print(f"  {html_path}")
    print(f"  {md_path}")
    print(f"  {run_dir / 'meta.json'}")
    latest_link = output_dir / "comparison_latest"
    if latest_link.exists() or latest_link.is_symlink():
        print(f"  {latest_link} → {run_dir.relative_to(output_dir)}")
