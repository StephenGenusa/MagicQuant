"""Model-card generation (M9).

Turns a MagicQuant output directory (``search_results.json`` plus, when
present, the shipped GGUFs, ``generated.json``, ``v2_results.json`` and the
latest ``compare`` run) into a HuggingFace README model card. Purely local --
no network. The optional ``--upload`` path lives in the CLI and is gated by
:func:`upload_blockers`.

The design follows docs/audits/2026-11-01-model-card-review.md. The rules
that matter most:

- Never present an unmeasured number as measured. A baseline counts as a
  BF16 baseline only when ``baseline_provenance`` says it was measured;
  every tier row carries a Status; predicted sizes are marked; no loss is
  derived from a placeholder baseline. (C1, C2)
- ``size_gb`` is bytes / 1024**3 (GiB). The card prints GiB and decimal GB,
  and prefers exact bytes from the shipped file. (C3)
- A tier is a SIZE BAND, not a recipe. The filename's quant label is
  disclosed as such. (C4)
- The card only READS persisted results. It never writes them back.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from magicquant.quant.tiers import (
    CURRENT_TIER_SCHEME_VERSION,
    classify_tier,
    describe_tier_band,
    tier_scheme_version,
)
from magicquant.utils.naming import config_key, hf_label_for_tier

_GIB = 1024 ** 3
_DEFAULT_ATTRIBUTION = "https://github.com/magiccodingman/MagicQuant-Wiki"
_STANDARD_TIERS = ["Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]

# Group legend (R1). Mirrors magicquant/gguf/tensor_groups.py's group set.
_GROUP_NAMES = {
    "E": "token embeddings",
    "H": "output head / MTP",
    "Q": "attention query",
    "K": "attention key + value",
    "O": "attention output",
    "U": "FFN up / gate",
    "D": "FFN down",
    "S": "SSM / linear attention",
    "N": "norms",
    "V": "vision",
    "X": "MoE experts",
    "R": "MoE router",
}

_PROBING_NOTES = {
    "partial": "some groups' sensitivities fell back to heuristic estimates",
    "heuristic": "no sensitivity probe succeeded; scheme choices rest on "
                 "heuristic estimates, not this model's measured behavior",
    "suspect": "most probe readings were clamped as implausible; treat the "
               "sensitivity ranking as unreliable",
    "insufficient": "measured probes cover too little of the model's "
                    "parameter mass to trust the sensitivity ranking",
    "unknown": "probing provenance was not determined",
}

_NOT_RECORDED = "‹not recorded›"
_OWNER_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


# ── small helpers ───────────────────────────────────────────────────────────

def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _format_scheme_map(config: Dict[str, str]) -> str:
    """Render a per-group scheme map like ``E:BF16 H:BF16 Q:Q6_K ...``."""
    return " ".join(f"{g}:{config[g]}" for g in sorted(config))


def _fmt(v, spec: str, default: str = "—") -> str:
    return format(v, spec) if _is_num(v) else default


def _or_not_recorded(v) -> str:
    if v is None or v == "":
        return _NOT_RECORDED
    return f"`{v}`"


def _scheme_category(name: str) -> Optional[str]:
    try:
        from magicquant.quant.schemes import get_scheme_by_name

        return get_scheme_by_name(name).category
    except Exception:
        return None


# ── input collection (I/O lives here, generation stays pure) ───────────────

def _read_json(path: Path) -> Optional[Dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _inspect_gguf(path: Path, hash_files: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {"filename": path.name, "bytes": path.stat().st_size}
    if hash_files:
        info["sha256"] = _sha256(path)
    try:
        from magicquant.gguf.reader import GGUFReader

        with GGUFReader(str(path)) as reader:
            meta = reader.get_metadata()
    except Exception as exc:  # unreadable file: listed, flagged, not fatal
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    for key, field in (
        ("magicquant.group_schemes", "group_schemes"),
        ("magicquant.source_precision", "source_precision"),
        ("magicquant.fallbacks", "fallbacks"),
    ):
        raw = meta.get(key)
        if isinstance(raw, str):
            try:
                info[field] = json.loads(raw)
            except ValueError:
                pass
    return info


def _latest_compare(output_dir: Path) -> Optional[Dict[str, Any]]:
    root = output_dir / "comparisons"
    if not root.is_dir():
        return None
    runs = sorted(
        (d for d in root.iterdir() if (d / "meta.json").is_file()),
        key=lambda d: d.name,
    )
    if not runs:
        return None
    run = runs[-1]
    report = run / "comparison.md"
    # Per-model status counts from raw_responses/<model>/Qnn.json
    # (primary_status: pass / near_miss / fail / unscored).
    scores: Dict[str, Dict[str, int]] = {}
    raw = run / "raw_responses"
    if raw.is_dir():
        for model_dir in sorted(d for d in raw.iterdir() if d.is_dir()):
            counts: Dict[str, int] = {}
            for q in model_dir.glob("*.json"):
                rec = _read_json(q) or {}
                status = rec.get("primary_status") or "unknown"
                counts[status] = counts.get(status, 0) + 1
                if "flags" in rec:  # absent in runs before degeneration flags
                    counts["_flag_recorded"] = 1
                    counts["_flagged"] = counts.get("_flagged", 0) + bool(rec["flags"])
            if counts:
                scores[model_dir.name] = counts
    return {
        "dir": run.relative_to(output_dir).as_posix(),
        "report": report.relative_to(output_dir).as_posix() if report.is_file() else None,
        "meta": _read_json(run / "meta.json") or {},
        "scores": scores,
    }


def _kl_scores(out: Path) -> Optional[Dict[str, Any]]:
    from magicquant.kl_score import load_kl_scores

    return load_kl_scores(out)


def _kl_for(inputs: Dict[str, Any], filename: Optional[str]) -> Dict[str, Any]:
    """kl_scores.json entry for a file, or {} (failed entries count as absent)."""
    rec = ((inputs.get("kl_scores") or {}).get("files") or {}).get(filename or "")
    return rec if isinstance(rec, dict) and not rec.get("error") else {}


def collect_card_inputs(output_dir, hash_files: bool = True) -> Dict[str, Any]:
    """Gather everything the card can use from an output directory.

    Returns a dict with keys ``files`` (one record per ``*.gguf``),
    ``generated`` (``generated.json``), ``v2_results`` and ``compare``
    (the latest ``comparisons/<run>``), each None/empty when absent.
    """
    out = Path(output_dir)
    return {
        "files": [_inspect_gguf(p, hash_files)
                  for p in sorted(out.glob("*.gguf")) if p.is_file()],
        "generated": _read_json(out / "generated.json"),
        "v2_results": _read_json(out / "v2_results.json"),
        "compare": _latest_compare(out),
        "kl_scores": _kl_scores(out),
    }


# ── row model ───────────────────────────────────────────────────────────────

def _baseline(results: Dict) -> Tuple[Optional[float], Optional[str], bool]:
    """(baseline_ppl, provenance, trusted)."""
    ppl = results.get("baseline_ppl")
    prov = results.get("baseline_provenance")
    if not _is_num(ppl):
        return None, prov, False
    if prov is None:  # predates provenance tracking: shown, but labelled
        return ppl, None, True
    return ppl, prov, prov == "measured"


def _tier_status(info: Dict, measurement: Optional[Dict]) -> str:
    if measurement and measurement.get("measurement_invalid"):
        return "invalid measurement"
    if _is_num(info.get("ppl")):
        if measurement and measurement.get("kl_timeout"):
            return "measured (KL timed out)"
        return "measured"
    if _is_num(info.get("predicted_loss")):
        return "predicted (not built/measured)"
    return "not measured"


def _match_files(rows: List[Dict], files: List[Dict], generated: Optional[Dict],
                 v2_results: Optional[Dict]) -> None:
    """Attach the shipped file to each row: generated.json first, then the
    GGUF's own magicquant.group_schemes, then v2's final_model."""
    by_name = {f["filename"]: f for f in files}
    gen_files = (generated or {}).get("files") or {}
    claimed = set()

    for row in rows:
        for fname, rec in gen_files.items():
            if fname in by_name and fname not in claimed and isinstance(rec, dict) and (
                rec.get("tier") == row["tier"]
                or (rec.get("config_key") and rec.get("config_key") == row["key"])
            ):
                row["file"] = by_name[fname]
                row["generated"] = rec
                claimed.add(fname)
                break
    for row in rows:
        if "file" in row or not row["config"] or row["v2"]:
            continue
        for f in files:
            gs = f.get("group_schemes")
            if (f["filename"] not in claimed and isinstance(gs, dict) and gs
                    and config_key(gs) == row["key"]):
                row["file"] = f
                claimed.add(f["filename"])
                break
    final = (v2_results or {}).get("final_model")
    if final:
        name = Path(final).name
        for row in rows:
            if row["v2"] and "file" not in row and name in by_name and name not in claimed:
                row["file"] = by_name[name]
                claimed.add(name)
                break


def _build_rows(results: Dict, inputs: Dict) -> List[Dict]:
    tiered: Dict[str, Dict] = results.get("tiered") or results.get(
        "tiered_survivors", {}
    ) or {}
    measurements: Dict[str, Dict] = results.get("measurements") or {}
    base_ppl, _prov, base_trusted = _baseline(results)

    order = [t for t in _STANDARD_TIERS if t in tiered]
    order += sorted(t for t in tiered if t not in _STANDARD_TIERS)  # C5

    rows = []
    for tier in order:
        info = tiered[tier] or {}
        config = info.get("config") or {}
        v2 = info.get("algo") == "v2-budget"
        row: Dict[str, Any] = {
            "tier": tier, "config": config, "v2": v2,
            "key": config_key(config) if config else "",
            "source": info.get("source"),
        }
        if v2:
            ppl = info.get("ppl")
            row["status"] = "measured" if _is_num(ppl) else "predicted (not built/measured)"
            row["ppl"] = ppl
            row["baseline"] = info.get("baseline_ppl")
            row["baseline_trusted"] = _is_num(row["baseline"])
            actual = info.get("actual_bytes")
            row["bytes"] = actual if _is_num(actual) else info.get("predicted_bytes")
            row["size_predicted"] = not _is_num(actual)
            row["measurement"] = None
        else:
            measurement = measurements.get(row["key"]) if config else None
            row["measurement"] = measurement
            row["status"] = _tier_status(info, measurement)
            row["ppl"] = info.get("ppl")
            row["baseline"] = base_ppl
            row["baseline_trusted"] = base_trusted
            size_gb = info.get("size_gb")
            row["bytes"] = size_gb * _GIB if _is_num(size_gb) else None
            row["size_predicted"] = (
                not row["status"].startswith("measured")
                and row["status"] != "invalid measurement"
            )
        rows.append(row)

    _match_files(rows, inputs.get("files") or [], inputs.get("generated"),
                 inputs.get("v2_results"))
    for row in rows:
        f = row.get("file")
        if f and _is_num(f.get("bytes")):  # exact bytes from the shipped file
            row["bytes"] = f["bytes"]
            row["size_predicted"] = False
        row["measured"] = row["status"].startswith("measured")
        m = row["measurement"] or {}
        row["kl"] = (m.get("kl") or {}) if row["measured"] else {}
        if not row["kl"] and f:
            row["kl"] = _kl_for(inputs, f.get("filename"))
        row["bench"] = (m.get("bench") or {}) if row["measured"] else {}

    # R3: largest (highest quality) first; unknown sizes last.
    rows.sort(key=lambda r: -(r["bytes"] if _is_num(r["bytes"]) else -1))
    return rows


def _tracking(results: Dict) -> Tuple[Optional[bool], Optional[float], Optional[int]]:
    """predictor_tracking as (verdict, tau, n_pairs). The orchestrator stores
    ``{"is_tracking", "tau", "n_pairs"}``; a bare bool is accepted too."""
    t = results.get("predictor_tracking")
    if isinstance(t, dict):
        verdict = t.get("is_tracking")
        return (verdict if isinstance(verdict, bool) else None,
                t.get("tau") if _is_num(t.get("tau")) else None,
                t.get("n_pairs") if _is_num(t.get("n_pairs")) else None)
    return (t if isinstance(t, bool) else None), None, None


def _dominated_by(row: Dict, rows: List[Dict]) -> Optional[Dict]:
    """A measured row another measured row beats on BOTH size and PPL
    (no worse on either, strictly better on one), or None."""
    if not (row["measured"] and _is_num(row["bytes"]) and _is_num(row["ppl"])):
        return None
    for other in rows:
        if other is row or not (other["measured"] and _is_num(other["bytes"])
                                and _is_num(other["ppl"])):
            continue
        if (other["bytes"] <= row["bytes"] and other["ppl"] <= row["ppl"]
                and (other["bytes"] < row["bytes"] or other["ppl"] < row["ppl"])):
            return other
    return None


def _delta(row) -> Tuple[Optional[float], Optional[float]]:
    if not (row["measured"] and row["baseline_trusted"]
            and _is_num(row["ppl"]) and _is_num(row["baseline"]) and row["baseline"]):
        return None, None
    d = row["ppl"] - row["baseline"]
    return d, d / row["baseline"]


# ── sections ────────────────────────────────────────────────────────────────

def _front_matter(fm: Dict[str, Any], base_model_name: str) -> List[str]:
    lines = ["---"]
    base_id = fm.get("base_model_id") or (
        base_model_name if _OWNER_NAME.match(base_model_name or "") else None
    )
    if base_id:
        lines += ["base_model:", f"- {base_id}", "base_model_relation: quantized"]
    if fm.get("license"):
        lines.append(f"license: {fm['license']}")
    if fm.get("quantized_by"):
        lines.append(f"quantized_by: {fm['quantized_by']}")
    lines.append("library_name: gguf")
    if fm.get("pipeline_tag"):
        lines.append(f"pipeline_tag: {fm['pipeline_tag']}")
    langs = fm.get("languages") or []
    if langs:
        lines.append("language:")
        lines += [f"- {lang}" for lang in langs]
    lines += ["tags:", "- gguf", "- quantized", "- mixed-precision", "- magicquant"]
    return lines + ["---", ""]


def _intro(results: Dict, rows: List[Dict], url: str) -> List[str]:
    groups = sorted({g for r in rows for g in r["config"]})
    any_v2 = any(r["v2"] for r in rows)
    any_v1 = any(not r["v2"] for r in rows)
    granularity = (
        "per-tensor and per-tensor-group" if any_v2 and any_v1
        else "per-tensor" if any_v2 else "per-tensor-group"
    )
    named = [_GROUP_NAMES.get(g, g) for g in groups if g != "N"]
    what = f" ({', '.join(named)})" if named else ""
    text = (
        f"Mixed-precision GGUF quants produced by [MagicQuant]({url}) with "
        f"{granularity} scheme selection: different parts of the model"
        f"{what} use different quantization schemes"
    )
    probing = results.get("probing_provenance")
    if probing == "measured":
        return [text + ", chosen from measured sensitivity.", ""]
    if probing is None:
        return [text + ".", ""]
    note = _PROBING_NOTES.get(probing, "unrecognized probing provenance")
    return [text + ".", "",
            f"> ⚠️ Sensitivity probing provenance: `{probing}` — {note}.", ""]


def _baseline_section(results: Dict, rows: List[Dict]) -> List[str]:
    ppl, prov, _trusted = _baseline(results)
    lines: List[str] = []
    if ppl is None and prov is None:
        v2 = next((r for r in rows if r["v2"] and _is_num(r["baseline"])), None)
        if v2:
            lines.append(f"**Baseline (BF16) perplexity:** {v2['baseline']:.4f} "
                         f"(measured by the v2 budget run)")
    elif prov is None:
        lines.append(f"**Baseline (BF16) perplexity:** {ppl} "
                     f"(provenance not recorded — older results file)")
    elif prov == "measured" and ppl is not None:
        lines.append(f"**Baseline (BF16) perplexity:** {ppl}")
    else:
        lines.append(
            f"**Baseline (BF16) perplexity:** not measured (provenance: "
            f"`{prov}`). No quality-loss figures are shown, because there is "
            f"no measured baseline to compare against."
        )
    bsize = results.get("baseline_size_gb")
    if _is_num(bsize) and bsize > 0:
        if lines:
            lines.append("")
        lines.append(f"**BF16 size:** {bsize:.2f} GiB ({bsize * _GIB / 1e9:.2f} GB)")
    return lines + ([""] if lines else [])


def _double_quant_callout(files: List[Dict]) -> List[str]:
    hits = [f for f in files if (f.get("source_precision") or {}).get("dequantized")]
    if not hits:
        return []
    lines = ["> ⚠️ **Double-quantized.** These files were built from an "
             "already-quantized source (no BF16/F16/F32 release was used), so "
             "their quality is bounded by that source quant's error. A tier at "
             "or above the source precision cannot improve on it."]
    for f in hits:
        types = (f.get("source_precision") or {}).get("source_types") or {}
        breakdown = ", ".join(f"{n}× {t}" for t, n in sorted(types.items()))
        lines.append(f"> - `{f['filename']}`: source tensors {breakdown}")
    return lines + [""]


def _legacy_banner(results: Dict, rows: List[Dict]) -> List[str]:
    version = tier_scheme_version(results)
    if version >= CURRENT_TIER_SCHEME_VERSION:
        return []
    lines = [
        f"> ⚠️ This search ran under tier_scheme_version={version} "
        f"(current: {CURRENT_TIER_SCHEME_VERSION}) -- tier labels below "
        f"follow OLDER, wider size-ratio boundaries. Compare the actual "
        f"**Size** columns rather than assuming a tier name (e.g. "
        f"\"Q5\") matches its current meaning."
    ]
    bsize = results.get("baseline_size_gb")
    if _is_num(bsize) and bsize > 0:  # C9: say what each label means today
        parts = []
        for r in rows:
            if r["v2"] or not _is_num(r["bytes"]):
                continue
            now = classify_tier(r["bytes"] / _GIB, bsize)
            parts.append(f"{r['tier']} → {now} {describe_tier_band(now)}")
        if parts:
            lines.append(">")
            lines.append(f"> Under the current bands (ratio to BF16): "
                         f"{'; '.join(parts)}.")
    else:
        lines.append(">")
        lines.append("> The BF16 size was not recorded for this run, so the "
                     "current band of each tier cannot be stated.")
    return lines + [""]


def _pick_section(rows: List[Dict]) -> List[str]:
    sized = [r for r in rows if _is_num(r["bytes"])]
    if not sized:
        return []
    lines = [
        "## Which file should I pick?",
        "",
        "Pick the largest file that fits in your GPU memory (or RAM, for CPU "
        "inference) **with headroom left for the KV cache**, which grows with "
        "context length and is not included in these sizes. Between two "
        "files of similar size, prefer the one with the lower mean KL "
        "divergence: it changes the model's token predictions less.",
        "",
    ]
    for r in sized:
        _d, pct = _delta(r)
        quality = f", ΔPPL {pct * 100:+.2f}%" if pct is not None else ""
        kld = r["kl"].get("mean_kl")
        if _is_num(kld):
            quality += f", mean KLD {kld:.4f}"
        mark = "≈" if r["size_predicted"] else ""
        better = _dominated_by(r, rows)
        advice = (f" — **not recommended**: {better['tier']} is smaller and "
                  f"has lower perplexity" if better else "")
        lines.append(f"- **{r['tier']}**: {mark}{r['bytes'] / 1e9:.2f} GB"
                     f"{quality} ({r['status']}){advice}")
    return lines + [""]


def _files_section(rows: List[Dict], files: List[Dict],
                   inputs: Optional[Dict[str, Any]] = None) -> List[str]:
    if not files:
        return []
    inputs = inputs or {}
    tier_of = {r["file"]["filename"]: r for r in rows if r.get("file")}
    any_sha = any("sha256" in f for f in files)
    kls = {f["filename"]: _kl_for(inputs, f["filename"]) for f in files}
    any_kl = any(kls.values())
    head = "| File | Tier | Size (GiB) | Size (GB) |"
    sep = "|---|---|---:|---:|"
    if any_kl:
        head += " PPL | Mean KLD | p99 KLD | Same top token |"
        sep += "---:|---:|---:|---:|"
    else:
        head += " Verified PPL |"
        sep += "---:|"
    if any_sha:
        head += " SHA-256 |"
        sep += "---|"
    lines = ["## Files", "", head, sep]
    for f in sorted(files, key=lambda f: -f["bytes"]):
        row = tier_of.get(f["filename"])
        gen = (row or {}).get("generated") or {}
        cells = [
            f"`{f['filename']}`",
            row["tier"] if row else "unmatched",
            f"{f['bytes'] / _GIB:.2f}",
            f"{f['bytes'] / 1e9:.2f}",
        ]
        if any_kl:
            k = kls[f["filename"]]
            top = k.get("same_top_p")
            cells += [
                _fmt(k.get("ppl"), ".3f") + (f" ± {k['ppl_err']:.3f}" if _is_num(k.get("ppl_err")) else ""),
                _fmt(k.get("mean_kl"), ".4f"),
                _fmt(k.get("p99_kl"), ".4f"),
                f"{top:.2f}%" if _is_num(top) else "—",
            ]
        else:
            cells.append(_fmt(gen.get("verify_ppl"), ".3f", "not verified"))
        if any_sha:
            cells.append(f"`{f['sha256']}`" if "sha256" in f else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for f in files:
        if f.get("error"):
            lines.append(f"- `{f['filename']}` could not be read: {f['error']}")
    if any(f.get("error") for f in files):
        lines.append("")
    if any(f["filename"] not in tier_of for f in files):
        lines += ["Files marked *unmatched* are in the output directory but "
                  "could not be tied to a tier in the search results (e.g. "
                  "reference files added for comparison).", ""]
    ks = inputs.get("kl_scores") or {}
    if any_kl:
        lines += [
            f"PPL and KL divergence for every file were measured by "
            f"`magicquant kl-score` against `{ks.get('reference')}` on "
            f"{_or_not_recorded(ks.get('corpus'))} at ctx "
            f"{_or_not_recorded(ks.get('ctx_size'))}, with one llama.cpp build, "
            f"so every row, references included, is directly comparable. "
            f"*Same top token* is how often the file's most likely next token "
            f"matches the reference's.",
            "",
        ]
    return lines


def _quality_section(results: Dict, rows: List[Dict]) -> List[str]:
    bsize = results.get("baseline_size_gb")
    has_ratio = _is_num(bsize) and bsize > 0
    any_kl = any(r["kl"] for r in rows)
    any_p99 = any(_is_num(r["kl"].get("p99_kl")) for r in rows)
    any_top = any(_is_num(r["kl"].get("same_top_p")) for r in rows)
    any_bench = any(r["bench"] for r in rows)

    lines = [
        "## Tiers",
        "",
        "> **The quant label in each filename (e.g. `Q5_K_M`) names a size "
        "band, not a recipe.** A tier is the lowest-loss mix of schemes "
        "whose size, as a ratio to the BF16 model, falls in that band. It "
        "may contain no tensors of the type its label names; the "
        "**Per-group schemes** column shows what the file actually holds.",
        "",
    ]
    cols = ["Tier", "Filename label", "Status", "Size (GiB)", "Size (GB)"]
    if has_ratio:
        cols.append("Ratio to BF16")
    cols += ["PPL", "ΔPPL", "ΔPPL %"]
    if any_kl:
        cols += ["Mean KLD", "p90 KLD"]
    if any_p99:
        cols.append("p99 KLD")
    if any_top:
        cols.append("Same top token")
    if any_bench:
        cols.append("Speed pp / tg (t/s)")
    cols.append("Per-group schemes")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    for r in rows:
        label = hf_label_for_tier(r["tier"])
        mark = "≈" if r["size_predicted"] else ""
        sized = _is_num(r["bytes"])
        cells = [
            r["tier"], f"`{label}`" if label else "—", r["status"],
            f"{mark}{r['bytes'] / _GIB:.2f}" if sized else "—",
            f"{mark}{r['bytes'] / 1e9:.2f}" if sized else "—",
        ]
        if has_ratio:
            cells.append(f"{mark}{r['bytes'] / (bsize * _GIB):.3f}" if sized else "—")
        if r["measured"] and _is_num(r["ppl"]):
            err = r["kl"].get("ppl_err")
            ppl_s = f"{r['ppl']:.3f}" + (f" ± {err:.3f}" if _is_num(err) else "")
        else:
            ppl_s = "—"
        d, pct = _delta(r)
        cells += [ppl_s, _fmt(d, "+.3f"),
                  f"{pct * 100:+.2f}%" if pct is not None else "—"]
        if any_kl:
            cells += [_fmt(r["kl"].get("mean_kl"), ".4f"),
                      _fmt(r["kl"].get("p90_kl"), ".4f")]
        if any_p99:
            cells.append(_fmt(r["kl"].get("p99_kl"), ".4f"))
        if any_top:
            top = r["kl"].get("same_top_p")
            cells.append(f"{top:.2f}%" if _is_num(top) else "—")
        if any_bench:
            pp, tg = r["bench"].get("pp_ts"), r["bench"].get("tg_ts")
            cells.append(f"{pp:.0f} / {tg:.1f}" if _is_num(pp) and _is_num(tg) else "—")
        cells.append(f"`{_format_scheme_map(r['config'])}`" if r["config"] else "—")
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines.append("| " + " | ".join(
            ["*no tiers recorded*"] + [""] * (len(cols) - 1)) + " |")

    lines += [
        "",
        "*ΔPPL* is the perplexity increase over the BF16 baseline (absolute, "
        "and relative: (PPL − baseline) / baseline). It is shown only for "
        "measured tiers with a measured baseline. "
        + ("*Mean / p90 KLD* is the KL divergence of the quant's token "
           "distribution from the BF16 model's on the same text (lower is "
           "better); it catches damage perplexity can hide. "
           if any_kl else "")
        + "*Size (GiB)* is bytes / 1024³; *Size (GB)* is bytes / 10⁹, the "
        "unit HuggingFace's file list uses.",
    ]
    if any(r["size_predicted"] for r in rows):
        lines += ["", "Sizes marked ≈ are predictions: those tiers were not "
                      "built and measured, so they carry no PPL or loss."]
    return lines + [""]


def _inside_section(rows: List[Dict], files: List[Dict]) -> List[str]:
    groups = sorted({g for r in rows for g in r["config"]})
    if not groups:
        return []
    lines = ["## What is inside each file", "", "| Group | Tensors |", "|---|---|"]
    lines += [f"| `{g}` | {_GROUP_NAMES.get(g, 'unrecognized group')} |" for g in groups]
    lines.append("")
    if any("BF16" in r["config"].values() for r in rows):  # C8
        lines += [
            "Groups listed as `BF16` are stored on disk as **F16** (a llama.cpp "
            "compute-graph limitation). Values outside F16's range are written "
            "as Q8_0 instead; such substitutions are listed below when the "
            "files are present.",
            "",
        ]
    if any(r["v2"] for r in rows):
        lines += [
            "Budget (v2) rows are allocated per tensor; their per-group map "
            "shows each group's most common scheme, not every tensor.",
            "",
        ]
    fb_lines = []
    for r in rows:
        f = r.get("file") or {}
        fb = f.get("fallbacks")
        if fb:
            detail = "; ".join(
                f"{info.get('count')}× {reason} (e.g. {info.get('example')})"
                for reason, info in sorted(fb.items())
            )
            fb_lines.append(f"- `{f['filename']}`: {detail}")
    if fb_lines:
        lines += ["**Tensors that fell back from their requested scheme:**", ""]
        lines += fb_lines + [""]
    elif files and all("fallbacks" in f for f in files if not f.get("error")):
        lines += ["No tensor fell back from its requested scheme.", ""]
    return lines


def _method_section(results: Dict, v2_results: Optional[Dict]) -> List[str]:
    def corpus(m):
        return _or_not_recorded(Path(m["corpus"]).name if m.get("corpus") else None)

    lines = ["## How these numbers were produced", ""]
    m = results.get("measurement")
    if isinstance(m, dict):
        chunks = m.get("chunks")
        lines += [
            "Perplexity and KL divergence were measured with `llama-perplexity`.",
            "",
            f"- Eval corpus: {corpus(m)}",
            f"- Context length: {_or_not_recorded(m.get('ctx_size'))} tokens",
            "- Chunks: " + (f"`{chunks}` (capped; not the full corpus)"
                            if _is_num(chunks) else "full corpus"),
            f"- Importance matrix: {'active' if m.get('imatrix_active') else 'not used'}",
            "- KL divergence vs BF16 logits: " + _kl_method(results, m),
        ]
    else:
        lines.append(
            f"Evaluation conditions (corpus, context length, chunk count) "
            f"were not recorded for this run: {_NOT_RECORDED}."
        )
    vm = (v2_results or {}).get("measurement")
    if isinstance(vm, dict):
        lines += [
            "",
            "Budget (v2) anchors:",
            "",
            f"- Eval corpus: {corpus(vm)}",
            f"- Context length: {_or_not_recorded(vm.get('ctx_size'))} tokens",
            f"- Anchor chunks: {_or_not_recorded(vm.get('anchor_chunks'))}",
            f"- Importance matrix: {'active' if vm.get('imatrix_active') else 'not used'}",
        ]
    lines += [
        "",
        "Perplexity on one corpus is a proxy, not a task benchmark. "
        "Differences smaller than the ± error are not meaningful.",
        "",
    ]
    return lines


def _kl_method(results: Dict, m: Dict) -> str:
    """kl_enabled only means BF16 logits were captured (the sensitivity
    probes always use them); per-candidate KL needs --enable-kl."""
    if any(isinstance(v, dict) and v.get("kl")
           for v in (results.get("measurements") or {}).values()):
        return "measured per candidate"
    if m.get("kl_enabled"):
        return ("used for the sensitivity probes only; search candidates were "
                "not KL-scored (`--enable-kl`). Shipped files are KL-scored in "
                "the Files table when `magicquant kl-score` was run")
    return "not measured"


def _candidates_section(results: Dict, rows: List[Dict]) -> List[str]:
    measurements = results.get("measurements") or {}
    usable = {k: v for k, v in measurements.items()
              if isinstance(v, dict) and not v.get("measurement_invalid")
              and _is_num(v.get("ppl")) and _is_num(v.get("size_gb"))}
    shipped = {r["key"] for r in rows}
    if not (set(usable) - shipped):
        return []
    try:
        from magicquant.pareto import pareto_frontier

        frontier = {e["key"] for e in pareto_frontier(usable)}
    except Exception:
        frontier = set()
    base_ppl, _prov, trusted = _baseline(results)
    lines = [
        "## All measured candidates",
        "",
        "<details>",
        f"<summary>{len(usable)} configurations were built and measured; "
        "the tiers above are a subset.</summary>",
        "",
        "| Size (GiB) | PPL | ΔPPL % | Mean KLD | Pareto | Shipped | Schemes |",
        "|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for key, v in sorted(usable.items(), key=lambda kv: -kv[1]["size_gb"]):
        pct = ((v["ppl"] - base_ppl) / base_ppl * 100
               if trusted and _is_num(base_ppl) and base_ppl else None)
        kld = (v.get("kl") or {}).get("mean_kl")
        pct_s = f"{pct:+.2f}%" if pct is not None else "—"
        lines.append(
            f"| {v['size_gb']:.2f} | {v['ppl']:.3f} | {pct_s} | "
            f"{_fmt(kld, '.4f')} | {'✓' if key in frontier else ''} | "
            f"{'✓' if key in shipped else ''} | `{key.replace('|', ' ')}` |"
        )
    lines += ["", "*Pareto* marks configurations no other candidate beats on "
              "both size and perplexity.", "", "</details>", ""]
    return lines


def _compare_section(compare: Optional[Dict], rows: List[Dict],
                     inputs: Optional[Dict[str, Any]] = None) -> List[str]:
    if not compare:
        return []
    meta = compare.get("meta") or {}
    where = compare.get("report") or compare.get("dir")
    lines = _compare_scores(compare.get("scores") or {}, rows, inputs)
    if not lines:
        lines = ["## Task-level validation", ""]
    return lines + [
        f"Side-by-side task validation (`magicquant compare`) was run over "
        f"{_or_not_recorded(meta.get('question_count'))} questions; the report "
        f"is `{where}`.",
        "",
        f"- Run: {_or_not_recorded(meta.get('timestamp'))}, MagicQuant commit "
        f"{_or_not_recorded(meta.get('git_commit'))}",
        f"- Questions SHA-256: {_or_not_recorded(meta.get('questions_sha256'))}",
        "",
    ]


def _compare_scores(scores: Dict[str, Dict[str, int]], rows: List[Dict],
                    inputs: Optional[Dict[str, Any]] = None) -> List[str]:
    """Per-file compare results. Unscored (open-ended, manual-review)
    questions are kept out of the pass denominator rather than counted as
    failures."""
    if not scores:
        return []
    tier_of = {r["file"]["filename"]: r["tier"] for r in rows if r.get("file")}
    lines = [
        "## Task-level validation",
        "",
        "| File | Tier | Mean KLD | Passed | Near miss | Failed | Unscored | Flagged |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(scores.items(),
                     key=lambda kv: (-kv[1].get("pass", 0), kv[0]))
    total_q = None
    for name, c in ordered:
        statuses = {st: n for st, n in c.items() if not st.startswith("_")}
        scored = sum(n for st, n in statuses.items() if st != "unscored")
        total_q = total_q or sum(statuses.values())
        flagged = str(c.get("_flagged", 0)) if c.get("_flag_recorded") else "—"
        lines.append(
            f"| `{name}` | {tier_of.get(name, 'reference')} | "
            f"{_fmt(_kl_for(inputs or {}, name).get('mean_kl'), '.4f')} | "
            f"{c.get('pass', 0)} / {scored} | {c.get('near_miss', 0)} | "
            f"{c.get('fail', 0)} | {c.get('unscored', 0)} | {flagged} |"
        )
    lines += [
        "",
        f"*Passed* is out of the automatically scored questions; *Unscored* "
        f"are open-ended answers (proofs, explanations) left for manual "
        f"review, and *Near miss* counts as a fail. *Flagged* answers "
        f"degenerated (a repetition loop, the wrong language, or no answer), "
        f"scored or not; see the compare report. With {total_q} questions "
        f"one question is a large step, so a gap of one or two between files "
        f"is noise; use this to spot where a category breaks, and KL "
        f"divergence / perplexity to rank close files.",
        "",
    ]
    return lines


def _compat_section(results: Dict, rows: List[Dict]) -> List[str]:
    schemes = sorted({s for r in rows for s in r["config"].values()})
    cats = {s: _scheme_category(s) for s in schemes}

    def named(cat):
        return ", ".join(f"`{s}`" for s, c in cats.items() if c == cat)

    notes = []
    if named("mxfp4"):
        notes.append(f"Uses {named('mxfp4')} (ggml type 39). It needs a "
                     f"llama.cpp build with MXFP4 support; older builds refuse "
                     f"to load the file.")
    if named("iq_quant"):
        notes.append(f"Uses I-quants ({named('iq_quant')}). They load "
                     f"everywhere, but their speed varies a lot between "
                     f"backends (CPU, CUDA, Metal, Vulkan, ROCm).")
    if named("rocmfpx"):
        notes.append(f"Uses ROCmFPX types ({named('rocmfpx')}). These load "
                     f"**only** on the ciru-ai/ROCmFPX llama.cpp fork, not on "
                     f"stock llama.cpp, Ollama or LM Studio.")
    if "NVFP4" in schemes:
        notes.append("Uses `NVFP4`, which needs a recent llama.cpp build with "
                     "NVFP4 support.")
    binary = results.get("llamacpp_binary")
    if binary:
        notes.append(f"Measurements were taken with `{Path(binary).name}` "
                     f"(the llama.cpp build recorded in the results).")
    if not notes:
        notes.append("Every scheme in these files is a standard ggml type that "
                     "current llama.cpp, Ollama and LM Studio load.")
    return ["## Compatibility", ""] + [f"- {n}" for n in notes] + [""]


def _usage_section(rows: List[Dict]) -> List[str]:
    name = next((r["file"]["filename"] for r in rows
                 if r.get("file") and not _dominated_by(r, rows)),
                next((r["file"]["filename"] for r in rows if r.get("file")),
                     "<file>.gguf"))
    return [
        "## Usage",
        "",
        "```bash",
        f"llama-cli -m {name} -cnv",
        f"llama-server -m {name} -c 8192",
        "```",
        "",
        f"Ollama: create a `Modelfile` containing `FROM ./{name}` and run "
        f"`ollama create <name> -f Modelfile`. LM Studio: place the file in "
        f"its models directory.",
        "",
    ]


def _provenance_section(results: Dict, v2_results: Optional[Dict]) -> List[str]:
    info = results.get("run_info") or {}
    algos = sorted({(t or {}).get("algo") or "v1-evolutionary"
                    for t in (results.get("tiered") or {}).values()})
    verdict, tau, n_pairs = _tracking(results)
    tracking_s = {True: "yes", False: "**no**"}.get(verdict, _NOT_RECORDED)
    if tau is not None:
        tracking_s += f" (Kendall τ {tau:.2f} over {n_pairs} pairs)"
    if "seed" not in results:
        seed_s = _NOT_RECORDED
    elif results["seed"] is None:
        seed_s = "none (not seeded, so not exactly reproducible)"
    else:
        seed_s = f"`{results['seed']}`"
    lines = [
        "## Provenance",
        "",
        f"- Search: {', '.join(f'`{a}`' for a in algos) or _NOT_RECORDED}; "
        f"tier_scheme_version {tier_scheme_version(results)}",
        f"- MagicQuant version: {_or_not_recorded(info.get('magicquant_version'))}, "
        f"commit {_or_not_recorded(info.get('git_commit'))}",
        f"- Search seed: {seed_s}",
        f"- llama.cpp binary: {_or_not_recorded(results.get('llamacpp_binary'))}",
        f"- Baseline provenance: {_or_not_recorded(results.get('baseline_provenance'))}",
        f"- Sensitivity probing provenance: "
        f"{_or_not_recorded(results.get('probing_provenance'))}",
        f"- Predictor tracked measurements: {tracking_s}",
    ]
    kp = (v2_results or {}).get("kappa_provenance")
    if isinstance(kp, dict) and kp:
        lines.append("- v2 group κ provenance: "
                     + ", ".join(f"{g}={p}" for g, p in sorted(kp.items())))
    if info.get("argv"):
        lines.append(f"- Invocation: `{' '.join(str(a) for a in info['argv'])}`")
    return lines + [""]


def _caveats(results: Dict, rows: List[Dict], v2_results: Optional[Dict]) -> List[str]:
    """C11: list every degraded flag present; never omit silently."""
    found: List[str] = []
    recorded = False
    for r in rows:
        m = r.get("measurement") or {}
        # Newer serializations always write these flags (False when clean);
        # their absence means an older file that never recorded them.
        if "kl_timeout" in m or "measurement_timeout" in m:
            recorded = True
        if m.get("measurement_timeout"):
            found.append(f"{r['tier']}: a measurement leg timed out "
                         f"({m.get('timeout_leg') or 'leg not recorded'}).")
        if m.get("kl_timeout"):
            found.append(f"{r['tier']}: the KL-divergence leg timed out; it is "
                         f"scored on perplexity alone.")
        if r.get("source") == "incumbent":
            found.append(f"{r['tier']}: the winner is a seeded stock llama.cpp "
                         f"mixture, not a configuration the search evolved.")
        if r["status"] == "invalid measurement":
            found.append(f"{r['tier']}: its measurement was rejected as "
                         f"physically implausible.")
        better = _dominated_by(r, rows)
        if better:
            found.append(f"{r['tier']}: dominated by {better['tier']}, which "
                         f"is smaller and has lower perplexity. The search "
                         f"found no better configuration in this size band.")
    if "predictor_tracking" in results:
        recorded = True
        if _tracking(results)[0] is False:
            found.append("The quality predictor did not track the measured "
                         "ranking in this run; treat tier winners with "
                         "suspicion.")
    m = results.get("measurement")
    if isinstance(m, dict):
        recorded = True
        if m.get("weights_degenerate"):
            found.append("Sensitivity weights were degenerate (uniform "
                         "fallback); group protection is not data-driven.")
    if results.get("llamacpp_arch_check") == "unknown":
        found.append("Architecture support in the llama.cpp build could not be "
                     "verified before measuring.")
    for miss in results.get("requested_tiers_missing") or []:
        found.append(f"Requested tier {miss.get('tier')} has no file: no "
                     f"configuration landed in its size band {miss.get('band')}.")
    if v2_results:
        recorded = True
        kp = v2_results.get("kappa_provenance") or {}
        imputed = sorted(g for g, p in kp.items() if str(p).startswith("imputed"))
        if imputed:
            found.append(f"v2 group amplification (κ) was imputed, not measured, "
                         f"for: {', '.join(imputed)} (--allow-partial-probes).")
        if v2_results.get("failures"):
            found.append(f"The v2 run recorded {len(v2_results['failures'])} "
                         f"measurement failure(s); see v2_results.json.")
    lines = ["## Caveats", ""]
    if found:
        lines += [f"- {c}" for c in found]
    elif recorded:
        lines.append("None recorded.")
    else:
        lines.append("Not recorded (older results file without degraded-run flags).")
    return lines + [""]


# ── public API ──────────────────────────────────────────────────────────────

def generate_model_card(
    results: Dict,
    base_model_name: str = "unknown",
    attribution_url: str = _DEFAULT_ATTRIBUTION,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    front_matter: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a markdown model card.

    Args:
        results: the loaded ``search_results.json``.
        base_model_name: the source model name for the card header (an
            ``owner/name`` value also becomes front-matter ``base_model``).
        attribution_url: link to the methodology wiki.
        inputs: optional :func:`collect_card_inputs` result (shipped files,
            generated.json, v2_results.json, latest compare run).
        front_matter: when not None, emit HuggingFace YAML front matter with
            these options (``license``, ``quantized_by``, ``pipeline_tag``,
            ``languages``, ``base_model_id``).

    Returns:
        A markdown string.
    """
    inputs = inputs or {}
    files = inputs.get("files") or []
    v2_results = inputs.get("v2_results")
    rows = _build_rows(results, inputs)

    lines: List[str] = []
    if front_matter is not None:
        lines += _front_matter(front_matter, base_model_name)
    lines += [f"# {base_model_name} — MagicQuant Hybrid Quants", ""]
    lines += _intro(results, rows, attribution_url)
    lines += _baseline_section(results, rows)
    lines += _double_quant_callout(files)
    lines += _legacy_banner(results, rows)
    lines += _pick_section(rows)
    lines += _files_section(rows, files, inputs)
    lines += _quality_section(results, rows)
    lines += _inside_section(rows, files)
    lines += _method_section(results, v2_results)
    lines += _candidates_section(results, rows)
    lines += _compare_section(inputs.get("compare"), rows, inputs)
    lines += _compat_section(results, rows)
    lines += _usage_section(rows)
    lines += _provenance_section(results, v2_results)
    lines += _caveats(results, rows, v2_results)
    lines += [
        "---",
        f"Methodology: [magiccodingman MagicQuant Wiki]({attribution_url}).",
        "",
    ]
    return "\n".join(lines)


def upload_blockers(results: Dict, inputs: Optional[Dict[str, Any]] = None
                    ) -> Tuple[List[str], List[str]]:
    """Reasons a card must not be uploaded without an explicit override (P8).

    Returns ``(unverified, double_quant)``: problems overridable with
    ``--allow-unverified-card`` and with ``--acknowledge-double-quant``.
    """
    inputs = inputs or {}
    unverified: List[str] = []
    _ppl, prov, trusted = _baseline(results)
    rows = _build_rows(results, inputs)
    if not trusted and not any(r["v2"] and _is_num(r["baseline"]) for r in rows):
        unverified.append(f"baseline perplexity is not measured (provenance: {prov})")
    for r in rows:
        if not r["measured"]:
            unverified.append(f"tier {r['tier']} is {r['status']}")
    if not rows:
        unverified.append("no tiers recorded")
    double_quant = [
        f["filename"] for f in inputs.get("files") or []
        if (f.get("source_precision") or {}).get("dequantized")
    ]
    return unverified, double_quant
