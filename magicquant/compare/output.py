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
    total_score: int = 0
    total_questions: int = 0
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


def _score_label(passes: int, total: int) -> str:
    return f"{passes}/{total}"


def _quant_from_filename(filename: str) -> str:
    import re
    m = re.search(r"\b(Q\d_K_[SM]|Q\d_K|Q\d_\d|IQ\d_\w+|BF16|F16|F32|MXFP4)\b",
                  filename, re.IGNORECASE)
    return m.group(1).upper() if m else "—"


def populate_scores(
    model: ModelInfo,
    questions: list[dict],
    results: dict[int, InferenceResult],
) -> None:
    """Fill model.total_score, model.per_mode_scores, model.consistency_score."""
    passes = 0
    total = 0
    consistencies = []
    mode_counts: dict[str, dict] = {}

    for q in questions:
        qid = q["id"]
        result = results.get(qid)
        if result is None:
            continue
        total += 1
        is_pass = result.primary_status == "pass"
        if is_pass:
            passes += 1
        consistencies.append(result.consistency)

        mode = q.get("failure_mode", "unknown")
        if mode not in mode_counts:
            mode_counts[mode] = {"pass": 0, "total": 0}
        mode_counts[mode]["total"] += 1
        if is_pass:
            mode_counts[mode]["pass"] += 1

    model.total_score = passes
    model.total_questions = total
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
            f"<td>{score_cell}</td>"
            f"<td>{cons_pct}</td>"
            f"{mode_cells}"
            f"</tr>\n"
        )
    summary_table = (
        f"<table><thead>{sum_header}</thead><tbody>\n{sum_rows}</tbody></table>\n"
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
            bar = _score_bar_html(result, n_samples)
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

    # Summary table
    active_modes = [m for m in _FAILURE_MODES
                    if any(q.get("failure_mode") == m for q in questions)]
    lines.append("## Summary")
    lines.append("")
    sum_header = ("| Model | bpw | Quant | Arch | PPL | Score | Consistency | "
                  + " | ".join(active_modes) + " |")
    lines.append(sum_header)
    lines.append("|" + "|".join(["---"] * (7 + len(active_modes))) + "|")
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
            f"{model.architecture} | {ppl} | {score} | {cons} | {mode_cols} |"
        )
    lines += ["", "---", ""]

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
    }
    (model_dir / f"{qid}.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def write_meta_json(metadata: ReproMetadata, run_dir: Path) -> None:
    (run_dir / "meta.json").write_text(
        json.dumps(asdict(metadata), indent=2, default=str), encoding="utf-8"
    )
