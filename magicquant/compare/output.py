"""HTML and Markdown output generation for magicquant compare."""

from __future__ import annotations

import html as html_module
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from magicquant.compare.scoring import ScoreResult
from magicquant.compare.metadata import ReproMetadata


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    samples: list[str]
    scored_samples: list[ScoreResult]
    consistency: float
    primary_status: str
    primary_response: str
    flags: list = field(default_factory=list)  # degeneration flags, primary answer


@dataclass
class ModelInfo:
    path: Path
    filename: str
    bpw: float
    quant_type: str
    param_count: Optional[int]
    architecture: str
    size_gb: float
    ppl: Optional[float] = None
    mean_kld: Optional[float] = None  # vs the reference, from kl_scores.json
    flagged_answers: int = 0          # answers with a degeneration flag
    total_score: int = 0
    total_questions: int = 0      # automatically scored questions only
    unscored_questions: int = 0   # open-ended answers left for manual review
    consistency_score: float = 1.0
    per_mode_scores: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

_FAILURE_MODES = [
    "arithmetic", "factual_recall", "multilingual", "long_context",
    "multi_hop", "code", "proof", "instruction_following",
]


def _pct_color(passes: int, total: int) -> str:
    if total == 0:
        return "neutral"
    r = passes / total
    if r == 1.0:
        return "pass"
    if r == 0.0:
        return "fail"
    return "partial"


def _why_paragraph(models: list, questions: list, results: dict) -> str:
    """Why the side-by-side comparison exists, with this run's own example
    of a failure the averages and the score miss (when there is one)."""
    from magicquant.compare.degeneration import FLAG_LABELS

    text = (
        "**Why this comparison exists.** Perplexity and KL divergence tell you "
        "how far a quantized file has drifted from the original, averaged over "
        "hundreds of thousands of tokens. They can't tell you what that drift "
        "does to an actual answer. This page puts each file's real answers to "
        "the same questions side by side, so you can see where a quant starts "
        "to break: wrong numerical answers, a question answered in the wrong "
        "language, code that stops working, or an answer that collapses into a "
        "loop. A file can look nearly lossless on the averages and still fail "
        "one of these."
    )
    flagged = [
        (m.filename, q, r)
        for m in models
        for q in questions
        for r in [results.get(m.filename, {}).get(q["id"])]
        if r is not None and r.flags
    ]
    if flagged:
        name, q, r = flagged[0]
        label = FLAG_LABELS.get(r.flags[0], r.flags[0])
        article = "an" if label[0] in "aeiou" else "a"
        counted = ("an answer the score does not count"
                   if r.primary_status == "unscored"
                   else f"an answer scored {r.primary_status.replace('_', ' ')}")
        text += (f" In this run, `{name}` produced {article} {label} on "
                 f"Q{q['id']} ({q.get('failure_mode', '')}), {counted}.")
        if len(flagged) > 1:
            text += f" {len(flagged) - 1} more flagged answer(s) are marked below."
    text += (" Use KL divergence to rank files that are close, and this page to "
             "check that the one you pick still answers the kinds of questions "
             "you care about.")
    return text


def _how_to_read_points(models: list) -> list:
    """Plain-text explanation shown under the run header (HTML and Markdown)."""
    scored = max((m.total_questions for m in models), default=0)
    step = f"about {100 / scored:.0f} points" if scored else "a large step"
    points = [
        "**Numerical answers** are the most sensitive canary. Models produce "
        "numbers by learned prediction, not exact calculation, so a small "
        "shift in predictions that leaves prose readable turns a digit wrong. "
        "The arithmetic column often falls first.",
        f"**Score** counts automatically scored questions only ({scored} here). "
        f"Open-ended answers (proofs, explanations) are left for you to judge "
        f"and are not counted as failures; near misses count as fails. One "
        f"question is {step}, so a gap of one or two between files is noise.",
    ]
    if any(m.mean_kld is not None for m in models):
        points.append(
            "**Mean KLD** is how far a file's next-token predictions drift from "
            "the reference over the whole eval corpus (lower is better; the "
            "reference against its own logits shows the noise floor, not 0). "
            "Use it and PPL to rank files that score alike.")
    points.append(
        "**Consistency** is always 1.00 at temperature 0; it only means "
        "something with --n-samples above 1 and a temperature above 0.")
    points.append(
        "**Flagged** counts answers that degenerated: a repetition loop, an "
        "answer in the wrong language, or no answer. Neither the score nor KL "
        "divergence shows these, so read the flagged answers below. Hitting "
        "the token limit alone is not a flag.")
    return points


def _md_inline_to_html(text: str) -> str:
    """Escape, then render the **bold** and `code` spans used in the texts above."""
    import re as _re

    out = html_module.escape(text)
    out = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return _re.sub(r"`(.+?)`", r"<code>\1</code>", out)


def _how_to_read_html(models: list, questions: list, results: dict) -> str:
    why = _md_inline_to_html(_why_paragraph(models, questions, results))
    items = "".join(f"<li>{_md_inline_to_html(pt)}</li>"
                    for pt in _how_to_read_points(models))
    return (f"<div class='how-to-read'><p class='why'>{why}</p>"
            f"<b>How to read this</b><ul>{items}</ul></div>")


def _flag_badges_html(flags: list) -> str:
    from magicquant.compare.degeneration import FLAG_LABELS

    return "".join(
        f"<div class='flag-badge'>⚠ {html_module.escape(FLAG_LABELS.get(f, f))}</div>"
        for f in flags or []
    )


def _kld_str(model: "ModelInfo") -> str:
    return f"{model.mean_kld:.4f}" if model.mean_kld is not None else "—"


def _summary_note(models: list) -> str:
    """Note under the summary table. The scoring rules live in the "How to
    read this" block above it; this only states where PPL/KLD came from."""
    if any(m.mean_kld is not None for m in models):
        return ("PPL and Mean KLD come from `magicquant kl-score` (one reference, "
                "same build, corpus and context), so every file is comparable.")
    return ""


def _score_label(passes: int, total: int) -> str:
    # total == 0: every question in this mode is unscored (manual review);
    # "0/0" or "0/1" would read as a failure.
    return f"{passes}/{total}" if total else "—"


def _quant_from_filename(filename: str) -> str:
    import re
    found = re.findall(r"\b(Q\d_K_[SM]|Q\d_K|Q\d_\d|IQ\d_\w+|BF16|F16|F32|MXFP4)\b",
                       filename, re.IGNORECASE)
    return found[-1].upper() if found else "—"  # last label wins (see _discover_models)


def populate_scores(
    model: ModelInfo,
    questions: list[dict],
    results: dict[int, InferenceResult],
) -> None:
    """Fill model.total_score, model.per_mode_scores, model.consistency_score.

    Unscored questions (open-ended, manual review) are counted separately and
    kept OUT of every denominator: they are not failures.
    """
    passes = 0
    total = 0
    unscored = 0
    flagged = 0
    consistencies = []
    mode_counts: dict[str, dict] = {}

    for q in questions:
        qid = q["id"]
        result = results.get(qid)
        if result is None:
            continue
        mode = q.get("failure_mode", "unknown")
        mode_counts.setdefault(mode, {"pass": 0, "total": 0})
        if result.flags:  # scored or not: a looping proof still counts
            flagged += 1
        if result.primary_status == "unscored":
            unscored += 1
            continue
        total += 1
        is_pass = result.primary_status == "pass"
        if is_pass:
            passes += 1
        consistencies.append(result.consistency)

        mode_counts[mode]["total"] += 1
        if is_pass:
            mode_counts[mode]["pass"] += 1

    model.total_score = passes
    model.total_questions = total
    model.unscored_questions = unscored
    model.flagged_answers = flagged
    model.consistency_score = (sum(consistencies) / len(consistencies)) if consistencies else 1.0
    model.per_mode_scores = mode_counts


# ── HTML generation ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; color: #1a1a2e; background: #f0f2f5; padding: 20px; }
h1 { font-size: 20px; font-weight: 700; margin-bottom: 2px; }
.meta-header { background: #fff; border-radius: 6px; padding: 12px 16px;
               margin-bottom: 16px; font-size: 12px; color: #555;
               box-shadow: 0 1px 3px rgba(0,0,0,.08); line-height: 1.8; }
h2 { font-size: 14px; font-weight: 600; color: #444; margin: 16px 0 6px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px;
        border-radius: 6px; overflow: hidden; }
th { background: #1a1a2e; color: #fff; padding: 9px 10px;
     text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 8px 10px; border-bottom: 1px solid #e8eaed;
     vertical-align: top; font-size: 12px; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #fafbfc; }

/* Score cells */
td.pass   { background: #d4edda; color: #155724; font-weight: 600; }
td.partial { background: #fff3cd; color: #856404; font-weight: 600; }
td.fail   { background: #f8d7da; color: #721c24; font-weight: 600; }
td.neutral { color: #888; }

/* Score bar above each response */
.score-bar { padding: 3px 7px; border-radius: 3px; font-weight: 700;
             font-size: 11px; margin-bottom: 5px; }
.score-bar.pass     { background: #d4edda; color: #155724; }
.score-bar.fail     { background: #f8d7da; color: #721c24; }
.score-bar.near_miss { background: #fff3cd; color: #856404; }
.score-bar.unscored { background: #e2e3e5; color: #383d41; }
.score-bar.inconsistent { background: #fff3cd; color: #856404; }

/* Response box — column-reverse so scroll starts at the bottom (answer, not <think>) */
.response { max-height: 300px; overflow-y: auto; white-space: pre-wrap;
            word-break: break-word; line-height: 1.5; font-size: 12px;
            background: #f8f9fa; border-radius: 3px; padding: 6px 8px;
            display: flex; flex-direction: column-reverse; }

/* Question header cell */
.q-header { max-width: 240px; }
.q-id   { font-weight: 700; color: #888; font-size: 11px; }
.q-mode { font-size: 11px; color: #666; margin: 2px 0; }
.q-truth { font-size: 10px; color: #888; margin-top: 3px; font-style: italic; }
.q-prompt { color: #333; font-size: 11px; line-height: 1.5; margin-top: 4px; }

/* Difficulty badges */
.how-to-read { background: #fff; border-radius: 6px; padding: 10px 16px;
               margin-bottom: 16px; font-size: 12px; color: #444;
               box-shadow: 0 1px 3px rgba(0,0,0,.08); line-height: 1.6; }
.how-to-read ul { margin: 4px 0 0 18px; }
.how-to-read .why { margin-bottom: 8px; font-size: 13px; color: #1a1a2e; }
.flag-badge { display: inline-block; margin: 2px 0 4px; padding: 2px 8px;
              border-radius: 4px; background: #721c24; color: #fff;
              font-size: 11px; font-weight: 600; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
         font-size: 10px; font-weight: 700; }
.easy   { background: #d1fae5; color: #065f46; }
.medium { background: #fef3c7; color: #92400e; }
.hard   { background: #fee2e2; color: #991b1b; }

/* Model name in summary */
.model-name { font-weight: 600; }
.score-big { font-size: 14px; font-weight: 700; }
.score-green { color: #155724; }
.score-yellow { color: #856404; }
.score-red { color: #721c24; }

details summary { cursor: pointer; color: #555; font-size: 12px; }
details[open] summary { margin-bottom: 8px; }
"""


def _e(text: str) -> str:
    return html_module.escape(str(text))


def _score_color_class(passes: int, total: int) -> str:
    if total == 0:
        return ""
    r = passes / total
    if r >= 0.8:
        return "score-green"
    if r >= 0.5:
        return "score-yellow"
    return "score-red"


def _score_bar_html(result: InferenceResult, n_samples: int) -> str:
    s = result.primary_status
    sr = result.scored_samples[0] if result.scored_samples else None
    detail = sr.detail if sr else ""

    if n_samples > 1:
        passes = sum(1 for x in result.scored_samples if x.status == "pass")
        consistent = result.consistency >= 0.999
        if not consistent:
            label = f"⚠ {passes}/{n_samples} consistent · {detail}"
            css = "inconsistent"
        elif s == "pass":
            label = f"✓ {passes}/{n_samples} · {detail}"
            css = "pass"
        else:
            label = f"✗ {passes}/{n_samples} · {detail}"
            css = s
    else:
        icons = {"pass": "✓", "fail": "✗", "near_miss": "~", "unscored": "?"}
        label = f"{icons.get(s, '?')} {s.upper()} · {detail}"
        css = s

    return f"<div class='score-bar {css}'>{_e(label)}</div>"


def generate_html(
    models: list[ModelInfo],
    questions: list[dict],
    results: dict[str, dict[int, InferenceResult]],
    metadata: ReproMetadata,
    output_path: Path,
    n_samples: int = 1,
) -> None:
    e = _e

    # ── metadata header ───────────────────────────────────────────────────────
    sha_short = metadata.questions_sha256[:8]
    diff = metadata.difficulty_breakdown
    meta_line1 = (
        f"Generated: {metadata.timestamp} &nbsp;·&nbsp; "
        f"magicquant {metadata.magicquant_version} &nbsp;·&nbsp; "
        f"Questions: {metadata.question_count} of {metadata.pool_size} "
        f"(Easy: {diff.get('easy',0)} · Medium: {diff.get('medium',0)} · Hard: {diff.get('hard',0)}) "
        f"&nbsp;·&nbsp; SHA: {sha_short}"
    )
    meta_line2 = (
        f"System prompt: &ldquo;{e(metadata.system_prompt[:80])}&rdquo; &nbsp;·&nbsp; "
        f"Temperature: {metadata.temperature} &nbsp;·&nbsp; "
        f"Max tokens: {metadata.max_tokens} &nbsp;·&nbsp; "
        f"Context: {metadata.context_size} &nbsp;·&nbsp; "
        f"n_samples: {metadata.n_samples}"
    )

    # ── summary table ─────────────────────────────────────────────────────────
    active_modes = [m for m in _FAILURE_MODES
                    if any(q.get("failure_mode") == m for q in questions)]

    sum_header = (
        "<tr>"
        "<th>Model</th><th>bpw</th><th>Quant</th><th>Arch</th><th>PPL</th>"
        "<th>Mean KLD</th><th>Flagged</th>"
        "<th>Score</th><th>Consistency</th>"
        + "".join(f"<th>{e(m)}</th>" for m in active_modes)
        + "</tr>"
    )
    sum_rows = ""
    for model in models:
        ppl = f"{model.ppl:.4f}" if model.ppl else "—"
        score_cls = _score_color_class(model.total_score, model.total_questions)
        score_cell = (
            f"<span class='score-big {score_cls}'>"
            f"{model.total_score}/{model.total_questions}</span>"
        )
        cons_pct = f"{model.consistency_score:.2f}"
        flag_cls = "fail" if model.flagged_answers else ""
        mode_cells = ""
        for mode in active_modes:
            mc = model.per_mode_scores.get(mode, {"pass": 0, "total": 0})
            css = _pct_color(mc["pass"], mc["total"])
            mode_cells += f"<td class='{css}'>{_score_label(mc['pass'],mc['total'])}</td>"
        sum_rows += (
            f"<tr>"
            f"<td class='model-name'>{e(model.filename)}</td>"
            f"<td>{model.bpw:.2f}</td>"
            f"<td>{e(model.quant_type)}</td>"
            f"<td>{e(model.architecture)}</td>"
            f"<td>{ppl}</td>"
            f"<td>{_kld_str(model)}</td>"
            f"<td class='{flag_cls}'>{model.flagged_answers}</td>"
            f"<td>{score_cell}</td>"
            f"<td>{cons_pct}</td>"
            f"{mode_cells}"
            f"</tr>\n"
        )
    note = _summary_note(models)
    summary_table = (
        f"<table><thead>{sum_header}</thead><tbody>\n{sum_rows}</tbody></table>\n"
        + (f"<p class='meta-header'>{e(note.replace('`', ''))}</p>\n" if note else "")
    )

    # ── per-question comparison table ─────────────────────────────────────────
    model_headers = "".join(f"<th>{e(m.filename)}</th>" for m in models)
    q_rows = ""
    for q in questions:
        qid = q["id"]
        diff_badge = f"<span class='badge {q['difficulty']}'>{q['difficulty']}</span>"
        mode_tag = f"<div class='q-mode'>{e(q.get('failure_mode',''))}</div>"
        truth_str = ""
        if q.get("scoring_type") not in ("none", None) and q.get("ground_truth") is not None:
            tol = q.get("tolerance") or {}
            tol_str = ""
            if tol.get("abs"):
                tol_str = f" (±{tol['abs']})"
            truth_str = f"<div class='q-truth'>Expected: {e(str(q['ground_truth']))}{tol_str}</div>"
        q_header = (
            f"<td class='q-header'>"
            f"<div class='q-id'>Q{qid} {diff_badge}</div>"
            f"{mode_tag}"
            f"<div class='q-prompt'>{e(q['prompt'][:160])}</div>"
            f"{truth_str}"
            f"</td>"
        )
        model_cells = ""
        for model in models:
            result = results.get(model.filename, {}).get(qid)
            if result is None:
                model_cells += "<td><div class='score-bar fail'>✗ No result</div></td>"
                continue
            bar = _score_bar_html(result, n_samples) + _flag_badges_html(result.flags)
            resp_html = f"<div class='response'><div>{e(result.primary_response)}</div></div>"
            if n_samples > 1 and len(result.samples) > 1:
                extra_html = ""
                for i, (samp, scored) in enumerate(
                    zip(result.samples[1:], result.scored_samples[1:]), start=2
                ):
                    s_bar = f"<div class='score-bar {scored.status}'>Sample {i}: {e(scored.detail)}</div>"
                    s_resp = f"<div class='response'><div>{e(samp)}</div></div>"
                    extra_html += f"<div>{s_bar}{s_resp}</div>"
                resp_html += (
                    f"<details class='extra-samples'>"
                    f"<summary>{len(result.samples)-1} more sample(s)</summary>"
                    f"{extra_html}</details>"
                )
            model_cells += f"<td>{bar}{resp_html}</td>"
        q_rows += f"<tr>{q_header}{model_cells}</tr>\n"

    q_table = (
        f"<table><thead><tr><th class='q-header'>Question</th>"
        f"{model_headers}</tr></thead><tbody>\n{q_rows}</tbody></table>\n"
    )

    # ── failure mode breakdown ────────────────────────────────────────────────
    mode_header = "<tr><th>Failure Mode</th>" + "".join(
        f"<th>{e(m.filename)}</th>" for m in models
    ) + "</tr>"
    mode_rows = ""
    for mode in active_modes:
        mode_q_count = sum(1 for q in questions if q.get("failure_mode") == mode)
        row = f"<td>{e(mode)} ({mode_q_count})</td>"
        for model in models:
            mc = model.per_mode_scores.get(mode, {"pass": 0, "total": 0})
            css = _pct_color(mc["pass"], mc["total"])
            row += f"<td class='{css}'>{_score_label(mc['pass'],mc['total'])}</td>"
        mode_rows += f"<tr>{row}</tr>\n"
    mode_table = (
        f"<table><thead>{mode_header}</thead><tbody>\n{mode_rows}</tbody></table>\n"
    )

    # ── reproducibility footer ────────────────────────────────────────────────
    footer_rows = (
        f"<tr><td>Git commit</td><td>{e(metadata.git_commit)}</td></tr>"
        f"<tr><td>CLI args</td><td>{e(metadata.cli_args)}</td></tr>"
        f"<tr><td>Questions file</td><td>{e(metadata.questions_file)}</td></tr>"
        f"<tr><td>Questions SHA-256</td><td>{e(metadata.questions_sha256)}</td></tr>"
    )
    footer = (
        f"<details class='metadata-footer' style='margin-top:20px'>"
        f"<summary>Reproducibility Details</summary>"
        f"<table style='margin-top:8px'><tbody>{footer_rows}</tbody></table>"
        f"</details>"
    )

    how_to_read = _how_to_read_html(models, questions, results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MagicQuant Comparison — {e(metadata.timestamp)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>MagicQuant Model Comparison</h1>
<div class="meta-header">
  <div>{meta_line1}</div>
  <div>{meta_line2}</div>
</div>

{how_to_read}
<h2>Summary</h2>
{summary_table}

<h2>Response Comparison</h2>
{q_table}

<h2>Failure Mode Breakdown</h2>
{mode_table}

{footer}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# ── Markdown generation ────────────────────────────────────────────────────────

def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def generate_markdown(
    models: list[ModelInfo],
    questions: list[dict],
    results: dict[str, dict[int, InferenceResult]],
    metadata: ReproMetadata,
    output_path: Path,
    n_samples: int = 1,
) -> None:
    lines: list[str] = []
    diff = metadata.difficulty_breakdown
    sha_short = metadata.questions_sha256[:8]

    lines += [
        "# MagicQuant Comparison",
        "",
        f"**Generated:** {metadata.timestamp}  ",
        f"**magicquant version:** {metadata.magicquant_version}  ",
        f"**Questions:** {metadata.questions_file} (SHA-256: {sha_short})  ",
        f"**Question count:** {metadata.question_count} of {metadata.pool_size} "
        f"(Easy: {diff.get('easy',0)} · Medium: {diff.get('medium',0)} · Hard: {diff.get('hard',0)})  ",
        f"**System prompt:** \"{metadata.system_prompt[:80]}\"  ",
        f"**Temperature:** {metadata.temperature} · "
        f"**Max tokens:** {metadata.max_tokens} · "
        f"**Context:** {metadata.context_size} · "
        f"**n_samples:** {metadata.n_samples}",
        "",
        "---",
        "",
    ]

    lines.append(_why_paragraph(models, questions, results))
    lines.append("")
    lines.append("**How to read this**")
    lines.append("")
    lines += [f"- {pt}" for pt in _how_to_read_points(models)]
    lines += ["", "---", ""]

    # Summary table
    active_modes = [m for m in _FAILURE_MODES
                    if any(q.get("failure_mode") == m for q in questions)]
    lines.append("## Summary")
    lines.append("")
    sum_header = ("| Model | bpw | Quant | Arch | PPL | Mean KLD | Flagged | Score | Consistency | "
                  + " | ".join(active_modes) + " |")
    lines.append(sum_header)
    lines.append("|" + "|".join(["---"] * (9 + len(active_modes))) + "|")
    for model in models:
        ppl = f"{model.ppl:.4f}" if model.ppl else "—"
        score = f"{model.total_score}/{model.total_questions}"
        cons = f"{model.consistency_score:.2f}"
        mode_cols = " | ".join(
            _score_label(
                model.per_mode_scores.get(m, {}).get("pass", 0),
                model.per_mode_scores.get(m, {}).get("total", 0),
            )
            for m in active_modes
        )
        lines.append(
            f"| {model.filename} | {model.bpw:.2f} | {model.quant_type} | "
            f"{model.architecture} | {ppl} | {_kld_str(model)} | "
            f"{model.flagged_answers} | {score} | {cons} | {mode_cols} |"
        )
    lines += [""] + ([_summary_note(models), ""] if _summary_note(models) else [])

    from magicquant.compare.degeneration import FLAG_LABELS
    flagged_rows = [
        (model.filename, q["id"], q.get("failure_mode", ""), r.primary_status, r.flags)
        for model in models
        for q in questions
        for r in [results.get(model.filename, {}).get(q["id"])]
        if r is not None and r.flags
    ]
    if flagged_rows:
        lines += ["### Flagged answers", "",
                  "| Model | Question | Failure mode | Status | Flags |",
                  "|---|---|---|---|---|"]
        for name, qid, mode, status, flags in flagged_rows:
            labels = ", ".join(FLAG_LABELS.get(f, f) for f in flags)
            lines.append(f"| {name} | Q{qid} | {mode} | {status} | {labels} |")
        lines.append("")
    lines += ["---", ""]

    # Per-question sections
    for q in questions:
        qid = q["id"]
        tol = q.get("tolerance") or {}
        tol_str = f" (±{tol['abs']})" if tol.get("abs") else ""
        gt_str = f" · **Ground truth:** {q['ground_truth']}{tol_str}" if q.get("ground_truth") is not None else ""
        lines += [
            f"## Q{qid} — {q['prompt'][:100]}",
            f"**Difficulty:** {q['difficulty']} · "
            f"**Failure mode:** {q.get('failure_mode','')}{gt_str}",
            "",
        ]
        for model in models:
            result = results.get(model.filename, {}).get(qid)
            if result is None:
                lines += [f"### {model.filename}", "❌ No result", ""]
                continue

            s = result.primary_status
            icons = {"pass": "✅", "fail": "❌", "near_miss": "⚠", "unscored": "❓"}
            icon = icons.get(s, "❓")
            sr = result.scored_samples[0] if result.scored_samples else None
            detail = sr.detail if sr else ""

            if n_samples > 1:
                passes = sum(1 for x in result.scored_samples if x.status == "pass")
                header_line = f"{icon} **{passes}/{n_samples}** — {detail}"
            else:
                header_line = f"{icon} **{s.upper()}** — {detail}"

            lines += [
                f"### {model.filename} ({model.bpw:.2f} bpw)"
                + (f" — {n_samples} samples" if n_samples > 1 else ""),
                header_line,
                "",
            ]
            for line in result.primary_response.splitlines():
                lines.append(f"> {line}")
            lines.append("")

            if n_samples > 1 and len(result.samples) > 1:
                lines.append("<details><summary>Additional samples</summary>")
                lines.append("")
                for i, (samp, scored) in enumerate(
                    zip(result.samples[1:], result.scored_samples[1:]), start=2
                ):
                    icon2 = icons.get(scored.status, "❓")
                    lines.append(f"**Sample {i}** {icon2} {scored.status.upper()} — {scored.detail}")
                    lines.append("")
                    for line in samp.splitlines():
                        lines.append(f"> {line}")
                    lines.append("")
                lines += ["</details>", ""]

        lines += ["---", ""]

    # Failure mode breakdown
    lines += ["## Failure Mode Breakdown", ""]
    mode_header_md = "| Failure Mode | " + " | ".join(m.filename for m in models) + " |"
    lines.append(mode_header_md)
    lines.append("|" + "|".join(["---"] * (1 + len(models))) + "|")
    for mode in active_modes:
        mode_q_count = sum(1 for q in questions if q.get("failure_mode") == mode)
        row = f"| {mode} ({mode_q_count}) |"
        for model in models:
            mc = model.per_mode_scores.get(mode, {"pass": 0, "total": 0})
            emoji = " ✅" if mc["pass"] == mc["total"] and mc["total"] > 0 else ""
            row += f" {_score_label(mc['pass'],mc['total'])}{emoji} |"
        lines.append(row)
    lines += ["", "---", ""]

    # Reproducibility footer
    lines += [
        "<details>",
        "<summary>Reproducibility Details</summary>",
        "",
        f"- **Git commit:** {metadata.git_commit}",
        f"- **CLI args:** {metadata.cli_args}",
        f"- **Questions file:** {metadata.questions_file}",
        f"- **Questions SHA-256:** {metadata.questions_sha256}",
        "",
        "</details>",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── Raw response JSON ──────────────────────────────────────────────────────────

def write_raw_response(
    model_filename: str,
    question: dict,
    result: InferenceResult,
    run_dir: Path,
) -> None:
    model_dir = run_dir / "raw_responses" / model_filename
    model_dir.mkdir(parents=True, exist_ok=True)
    qid = f"Q{question['id']:02d}"
    data = {
        "question_id": question["id"],
        "difficulty": question.get("difficulty"),
        "failure_mode": question.get("failure_mode"),
        "prompt": question.get("prompt"),
        "ground_truth": question.get("ground_truth"),
        "scoring_type": question.get("scoring_type"),
        "samples": result.samples,
        "scored_samples": [asdict(s) for s in result.scored_samples],
        "consistency": result.consistency,
        "primary_status": result.primary_status,
        "response_language": question.get("response_language"),
        "flags": list(result.flags),
    }
    (model_dir / f"{qid}.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def write_meta_json(metadata: ReproMetadata, run_dir: Path) -> None:
    (run_dir / "meta.json").write_text(
        json.dumps(asdict(metadata), indent=2, default=str), encoding="utf-8"
    )
