# Model card review (`magicquant card`), 2026-11-01

> **Resolution (2026-11-01, same day):** every finding in section 2 (C1–C13,
> P1–P9, R1–R4) and every section 3(b) data gap (D-b1–D-b6) is fixed. See
> the two 2026-11-01 entries in `CHANGELOG.md` under `[Unreleased]`. C4 was
> fixed by disclosure on the card; the filenames still carry HF quant labels.
> Two items remain open: section 3(c), which needs new measurement work
> (corpus hashes, long-context quality, runtime memory, per-backend speed),
> and a `stale` tier status, which needs a resume/binary-mismatch record
> that the results do not persist. Line numbers below refer to the code as
> reviewed, before the fixes.

Scope: `magicquant/utils/model_card.py` (91 lines), `cmd_card` and `card_parser`
in `magicquant/__main__.py`, `tests/test_model_card.py`, README "Model Card", and
the result shapes the card could read (v1 `search_results.json`, the
`sensitivity.json` probe file, the v2 `v2_results.json`/interchange block, `pareto.json`,
and `compare` outputs). This is a review only. No package code, tests or docs were
changed. Nothing was uploaded, and no search, perplexity run or build was executed.

Method: a scratchpad inventory script listed every field in both fixtures and which
ones `generate_model_card` reads. A render script produced the current card from
`tests/fixtures/search_results_sample.json`, from
`tests/fixtures/qwopus_search_results_2026-07-04/search_results.json`, and from three
synthetic dicts. Each synthetic dict copies the exact shape a code path writes: a
prediction-only run (`orchestrator.py:2427-2459`, `2741-2753`), a fresh v2 interchange
block (`v2/interchange.py:40-64`), and a v2 block merged into the qwopus file.
`.venv/bin/python -m pytest tests/test_model_card.py -q` reports 5 passed.

## 1. Summary

- **What works:** the card is short and local-only. It keeps the magiccodingman
  attribution, prints PPL with relative loss against the run's own baseline (not
  absolute PPL across models), and already discloses a legacy `tier_scheme_version`.
- **Gap 1 (honesty):** it can't tell measured, predicted, fabricated, failed and
  not-measured apart. A prediction-only run publishes a **fabricated baseline PPL of
  5.0** and predicted sizes as if they were measured, under the line "based on
  measured sensitivity".
- **Gap 2 (correctness):** sizes are **GiB labelled "GB"**, 7.4% too small. The
  shipped filenames (`-Q5_K_M.gguf` and so on) claim recipes the files don't contain,
  and the card says nothing about it.
- **Gap 3 (completeness):** there's no HF front matter, no file list, no "which file"
  guidance, no compatibility notes, no KL even though KL is already measured, and no
  provenance (corpus, ctx, chunks, imatrix, seed, llama.cpp binary). v2 budget builds
  are silently dropped from the card.

## 2. Findings

Severity: Critical = misleading or wrong. High = missing information most users need.
Medium. Low.

### 2a. Correctness and honesty

| ID | Sev | Finding | Evidence | Recommended change |
|----|-----|---------|----------|--------------------|
| C1 | **Critical** | **A fabricated baseline PPL is printed as a measured BF16 baseline.** When baseline measurement fails or llama.cpp is missing, the prediction-only path sets `baseline_ppl = 5.0` and records `baseline_provenance = "fabricated"` / `"prediction-only"`. The card never reads `baseline_provenance` and prints `**Baseline (BF16) perplexity:** 5.0`. A reader compares tiers against a number nobody measured. | `orchestrator.py:2736-2753` (5.0 default plus provenance); `model_card.py:38,50-52` (prints `baseline_ppl` unconditionally); rendered synthetic prediction-only card shows the line verbatim. | Print the baseline only when `baseline_provenance == "measured"`. Otherwise show `Baseline PPL: not measured (provenance: <value>)` and suppress every Loss/ΔPPL figure. A missing provenance key (pre-provenance files such as `search_results_sample.json`) renders as "provenance not recorded". |
| C2 | **Critical** | **Predicted and unmeasured tiers look like measured results.** Prediction-only tiers carry `ppl: null`, `measured_loss: null`, and `size_gb` = **predicted** size (`info.get("size_gb", info.get("predicted_size_gb"))`). The card prints the predicted size in the same "Size (GB)" column with "—" for PPL. It has no status column, and the intro always says schemes were chosen "based on measured sensitivity", even when `probing_provenance` is `heuristic`/`partial`/`suspect`/`insufficient`. On the 2026-09-10 audit question, the card does not distinguish not-measured, failed, stale, predicted and verified. | `orchestrator.py:2432` (size fallback); `model_card.py:44-47` (hard-coded "measured sensitivity"); `model_card.py:76-83` (no status); `probing.py:498-585` (provenance values); `docs/audits/2026-09-10-review.md:197-198`. A synthetic `ppl: null` tier with a stored `measured_loss` renders `| Q4 | — | 4.00% |`, a loss shown with no PPL behind it. | Add a **Status** column: `measured`, `predicted (not built/measured)`, `measured – KL leg timed out`, `failed`, with `stale` reserved for a resume or binary mismatch. Mark predicted sizes `≈… (predicted)`. Print Loss only when `ppl` and a measured baseline both exist. Make the intro sentence conditional on `measurement.probing_provenance` / `sensitivity.json` `probing_provenance`. |
| C3 | **Critical** | **Size is GiB but labelled "GB", 7.4% too small.** `size_gb` is `st_size / 1024**3` (v1) and `total_bytes / 1024**3` (v2). HF's file browser shows decimal GB, so the card and the download page disagree. qwopus "Q6" prints `23.15` "GB", but the file is **24.86 GB** (decimal). A reader choosing for a 24 GB card picks a file that doesn't fit, even before KV cache. | `orchestrator.py:1743`; `v2/search.py:397,632`; `model_card.py:67,82`. The v2 key itself is named `BUDGET-<N>GiB` (`v2/interchange.py:26`), so the codebase knows the unit. | Rename the column to `Size (GiB)` and add `Size (GB)` = bytes/1e9. Better still, print exact bytes from `os.stat` of the shipped file (see D-b1). |
| C4 | **Critical** | **The shipped filenames claim a recipe, and the card is silent about it.** `generate_name` rewrites tier `Q5` to the filename suffix `Q5_K_M` (Q4 to `Q4_K_M`, Q8 to `Q8_0`, and so on), so HF shows a "Q5_K_M" badge. The qwopus "Q5" winner contains **no Q5_K_M tensors** (`D:IQ4_NL … U:MXFP4_MOE … S:Q5_K`), and its "Q8" contains no Q8_0 in its largest groups (`U:BF16 D:Q6_K`). That directly contradicts the "a tier is a size band, not a recipe" invariant, and the card neither lists filenames nor explains the label. | `utils/naming.py:60-68,99-102`; `orchestrator.py:2933,2846`; qwopus `tiered.Q5.config`, `tiered.Q8.config`. | (1) The card must state in the file table: "The quant label in the filename is the **size band** (ratio to BF16), not the scheme. See the per-group map." (2) Renaming outputs (for example `-MQ-Q5band`) is a separate human decision, because filenames are consumed downstream. Raise it, don't do it from the card. |
| C5 | High | **v2 budget builds are silently dropped.** The fixed tier order `["Q8","Q6","Q5","Q4","Q3","Q2"]` skips any other key, so `BUDGET-<N>GiB` blocks never render. A fresh v2 output dir produces a card with an **empty tier table and no baseline**, because the interchange block keeps `baseline_ppl` inside the tier and doesn't set it at top level. The block also has no `measured_loss` or `size_gb` (it has `actual_bytes`/`predicted_bytes`). | `model_card.py:71-73`; `v2/interchange.py:40-64`; render: "v2 merged into qwopus: BUDGET row present? False"; the fresh-v2 render shows header + empty table. | Render unknown tier keys after the known ones. For `algo == "v2-budget"`, read `actual_bytes` (fall back to `predicted_bytes`, marked predicted), `ppl`, and the block's own `baseline_ppl`. If `v2_results.json` exists, read anchors/`measurement` from it (see 5.3). Never print an empty table: say "no tiers recorded". |
| C6 | High | **PPL-only quality hides measured distribution damage.** Every qwopus measurement carries `kl.{mean_kl,p90_kl,max_kl}`, and none is shown. In this run, KL changes the advice. "Q5" (22.29 GiB, +1.62% PPL) has **mean KLD 0.0308**, over twice "Q6" at 23.15 GiB (0.0132), and also above the *smaller* 21.95 GiB candidate (0.0264). The 23.51 GiB `U:MXFP4_MOE` candidate matches Q6's PPL loss (1.18% vs 1.17%) at 2.1× the mean KLD. | qwopus `measurements[*].kl`; `orchestrator.py:1761-1762` (persisted); `model_card.py` reads no `kl`. | Join each tier to `measurements[naming.config_key(config)]` and show `mean KLD` and `p90 KLD`. Don't headline `max_kl`: it is a single-token maximum (8–24 here) and is noise. If `kl` is absent, show "not measured". If `kl_timeout`, show "KL leg timed out". |
| C7 | High | **"Loss" has no definition, no uncertainty and no conditions.** "Loss" is `(ppl − baseline)/baseline`, a relative PPL increase. It isn't accuracy loss or KLD. PPL is printed to 4 decimals with no ±, and the card never names the corpus, `ctx_size` (default **512**), or the chunk cap (`MAGICQUANT_PPL_CHUNKS`, possibly partial corpus). In qwopus, Q6 and Q5 differ by 0.03 PPL (0.45 pp). Without an error term nobody can tell whether that is signal. | `orchestrator.py:1690`; `llamacpp.py:649,685,1044-1057`; `_measurement_metadata` at `orchestrator.py:2190-2213` (persisted as `measurement`, unused by the card); qwopus has **no** `measurement` key, so the conditions are unrecorded. | Rename the column to `ΔPPL vs BF16 (%)` and add `ΔPPL` absolute. Print `ppl_err` when `kl.ppl_err` exists. Add an "Evaluation method" section from `measurement` (corpus basename, ctx, chunks or "full corpus", imatrix active), and state "not recorded" when the key is missing (as in qwopus). |
| C8 | Medium | **The scheme map says BF16 where the file holds F16** (and per-tensor fallbacks to Q8_0 are invisible). The writer stores BF16-designated groups as F16. Values outside F16 range are substituted with Q8_0 (`reason: "bf16-f16-out-of-range"`). Every qwopus tier shows `E:BF16`. An auditor running `gguf-dump` sees F16 or Q8_0. | `writer.py:1299-1317`; `_fallbacks` at `writer.py:972,1176-1337` (logged only, see D-b4). | Footnote every `BF16` cell: "stored as F16 on disk (llama.cpp limitation)". Once fallbacks are persisted, list them per file. |
| C9 | Medium | **The legacy-version banner doesn't say what the labels actually mean.** It tells readers to "compare the Size column", but the card has no BF16 size and no ratio, so they can't. Inference (**unverified**, derived only from band arithmetic): under v1 bands, the four qwopus tier sizes are jointly consistent only with a BF16 baseline of about 50.6–51.4 GiB (the Q4 upper bound 0.33 and the Q6 lower bound 0.45 bind). Under the *current* bands, "Q6" (ratio ≈0.450–0.458) and "Q5" (≈0.433–0.441) would **both** be Q6-band files (0.375, 0.46]. | `tiers.py:77-95`; `model_card.py:54-63`; `baseline_size_gb` is computed (`orchestrator.py:1143,2789`) but never persisted (D-b1). | Persist `baseline_size_gb` (additive top-level key) and print `ratio to BF16` per tier plus the band interval (`describe_tier_band`). For legacy files, print both the v1 and the current-scheme band each tier would fall in. |
| C10 | Medium | **Double quantization can't be disclosed.** The writer warns "say so on the model card", but the dequant-source fact is never written to `search_results.json` or to GGUF metadata, so the card cannot know. | `writer.py:1513-1525`; `gguf/source.py:52`; no `dequant` in `orchestrator.py`/`v2/`. | Persist a `source_precision` record (see D-b5). The card prints a prominent warning when it's non-float, and `cmd_card` refuses `--upload` without `--acknowledge-double-quant`. |
| C11 | Medium | **Degraded measurement states never reach the card.** `kl_timeout`, `measurement_timeout`/`timeout_leg`, `source: incumbent` (the tier winner is a seeded stock-mixture config, not an evolved one), `predictor_tracking` False ("treat its tier winners with suspicion", `orchestrator.py:2370-2376`), `weights_degenerate`, `llamacpp_arch_check`, and v2 `kappa_provenance: imputed` (`--allow-partial-probes`) are all persisted and all ignored. | `orchestrator.py:2440-2452,2500-2513,2290-2292`; `v2/search.py:577-582`. | A "Measurement caveats" block lists every degraded flag present. It is never omitted silently: it shows "none recorded" only when the keys exist and are clean, and "not recorded (older run)" when they're absent. |
| C12 | Low | **The intro text is wrong for some runs.** It says "per-tensor-group" for v2 per-tensor builds, and says "MoE experts" for dense models. qwopus has no `X`/`R` group. | `model_card.py:43-48`; qwopus configs have groups `D E H K O Q S U`. | Generate the sentence from the groups present and from `algo`. |
| C13 | Low | **Tier precedence is right but only implicit.** `tiered or tiered_survivors` matches CLAUDE.md (they hold the same entries minus `predicted_loss`), but an empty `tiered: {}` silently falls back. | `model_card.py:35-37`; sample fixture has `tiered_survivors: {}`. | Keep the precedence. Add a test pinning it, and never display `predicted_loss`: it is in noise units (residuals of −0.6 to −1.7 in qwopus) and can't be read as a percentage (`orchestrator.py:1722-1731`). |

### 2b. Completeness

| ID | Sev | Finding | Evidence | Recommended change |
|----|-----|---------|----------|--------------------|
| P1 | High | **No HF YAML front matter.** The HF Hub can't link `base_model`, show the quantized-from relation, apply the base license, or index tags. `--base-model` is free text used only in the H1, and defaults to the GGUF *file stem*. | `model_card.py:41`; `__main__.py:683-690`. | Emit front matter: `base_model`, `base_model_relation: quantized`, `library_name: gguf`, `license` (new `--license`, else a placeholder plus a warning), `quantized_by` (`--quantized-by`), `tags: [gguf, quantized, magicquant, mixed-precision]`, `pipeline_tag` (`--pipeline-tag`, default `text-generation`). Validate `--base-model` as `owner/name` when front matter is emitted. |
| P2 | High | **No file table and no "which file should I pick".** The card never names the `.gguf` files, so readers can't map a tier row to a download. There's no memory guidance and no note that KV cache comes on top of file size. | `cmd_card` reads only `search_results.json` (`__main__.py:674`). Filenames come from `generate_name` (`orchestrator.py:2846`). | `cmd_card` globs `output_dir/*.gguf`, reads `magicquant.group_schemes` from each (`writer.py:1577`), matches it to a tier by `config_key`, and prints filename, exact bytes, GiB/GB and SHA-256. A config mismatch between file and tier is a hard warning. Add a size-sorted "pick" list written as conditions ("largest file that leaves room for KV cache"), never as a quality claim beyond the measured numbers. |
| P3 | High | **No compatibility section.** qwopus "Q5" uses `MXFP4_MOE` (ggml type 39), which needs a llama.cpp build with MXFP4 support. Its "Q4" uses `IQ3_S`/`IQ3_XXS`/`IQ4_XS`, whose speed varies a lot by backend. ROCmFPX types (ids 100–104) load only on the ciru-ai fork. The model has an `S` (SSM) group, so it needs an arch-supporting build. `llamacpp_binary`/`llamacpp_arch_check` are persisted but unused. | `quant/schemes.py` categories (`mxfp4`, `iq_quant`, `rocmfpx`); qwopus tier configs; `orchestrator.py:2510-2513`. | Derive notes from the scheme categories present per file: MXFP4 → "requires llama.cpp with MXFP4 (type 39) support, ‹min build›". Any `rocmfpx` → "ciru-ai/ROCmFPX fork only; will not load on stock llama.cpp". `iq_quant` → backend-speed caveat. Print the measuring binary from `llamacpp_binary`. |
| P4 | High | **No provenance or reproducibility.** `seed`, `baseline_provenance`, `measurement.*`, the imatrix state, the calibration corpus, the MagicQuant version or commit, and the llama.cpp binary are all missing. For qwopus `seed: null`, so the run isn't seed-reproducible, and that should be stated. | inventory: top-level keys unused = `baseline_provenance`, `measurements`, `seed` (qwopus); `measurement` (newer runs). | A "Provenance" section that reads every one of these, prints "not recorded" for absent keys, and includes the exact `magicquant search …` / `generate` command if one is persisted (it isn't today, D-b6). |
| P5 | Medium | **Throughput is never reported.** `bench` (`pp_ts`, `tg_ts`, ± std) is persisted per measurement when `enable_speed_bench` is on. It's null in qwopus. v2 records `bandwidth.streamed_bytes` (bytes read per token for MoE). | `orchestrator.py:1776-1782`; `llamacpp.py:1173-1215`; `v2/search.py:584-599`. | Show `pp`/`tg` t/s ± std **with the hardware/backend caveat**, only when present. Otherwise print "not measured". For MoE v2 runs, show streamed GiB/token next to storage size. |
| P6 | Medium | **Task-level validation isn't referenced.** `magicquant compare` writes `comparison.md` + `meta.json` (with git commit and question-file SHA-256), and the card can't link or summarize them. | `compare/cmd_compare.py:413-420`; `compare/output.py:530-576`. | Add `--compare-run <dir>`: per-file score `k/n` with an explicit small-n note (the default is 20 questions) and the question-set SHA. No aggregate "accuracy" claims. |
| P7 | Medium | **Other measured candidates are hidden.** qwopus measured 7 configs. The 21.95 GiB candidate (2.91% / KLD 0.0264) is a real size/quality option that sits between Q5 and Q4 and doesn't appear. `pareto.json` exists for this purpose. | qwopus `measurements`; `orchestrator.py:2527-2586`. | An optional collapsed `<details>` table, "all measured configurations", built from non-`measurement_invalid` measurements only (inert-by-filter contract). |
| P8 | Medium | **`--upload` has no honesty gate.** It pushes whatever was generated, including a card built on a fabricated baseline or predicted-only tiers. | `__main__.py:696-713`. | Refuse `--upload` when the baseline isn't measured or any listed tier is unmeasured, unless `--allow-unverified-card` is passed, and say which condition tripped. (Review note: upload was not exercised.) |
| P9 | Low | **No usage snippet.** | none in the card. | `llama-cli -m <file> …` / `llama-server`, an Ollama `FROM ./file.gguf` note, and an LM Studio note. Static text, no numbers. |

### 2c. Presentation

| ID | Sev | Finding | Evidence | Recommended change |
|----|-----|---------|----------|--------------------|
| R1 | Medium | **The per-group map is cryptic.** Single letters sorted alphabetically (`D E H K O Q S U`) with no legend. `K` is key+value, `H` is head/MTP, and `S` is SSM, none of which a reader can guess. | `model_card.py:14-16`. | Add a legend table generated from groups present, and order groups by role (E, H, Q, K, O, U, D, S, X, R, V, N), not alphabetically. Keep `naming.config_key` untouched, since this is display only. |
| R2 | Low | **The precision is false.** PPL is printed to 4 decimals with no error term. | `model_card.py:80`. | Print PPL to 3 decimals plus `± ppl_err` when present, and loss to 2 decimals. |
| R3 | Low | **The tier table is ordered by label, not size.** It mostly coincides, but legacy and v2 labels break it. | `model_card.py:71`. | Sort rows by size, descending, and keep the label as a column. |
| R4 | Low | **Tests pin almost nothing about honesty.** 5 tests cover presence of tiers, the `—` placeholder, the survivors fallback and the legacy banner. | `tests/test_model_card.py`. | See 5.4. |

## 3. Data gap analysis

### (a) Already persisted, unused by the card

| Field | Where | Use on card |
|-------|-------|-------------|
| `baseline_provenance` | `search_results.json` top level (qwopus: `"measured"`) | Gate the baseline and loss figures (C1) |
| `seed` | top level (qwopus: `null`) | Provenance (P4) |
| `measurement.{chunks,ctx_size,corpus,imatrix_active,imatrix_n_tensors,kl_enabled,kl_objective_blend_active,kl_weight,probing_provenance,weights_degenerate}` | top level (newer runs; **absent in qwopus**) | Evaluation method, caveats (C7, C11) |
| `measurements[k].kl.{mean_kl,p90_kl,max_kl,ppl_err?}` | per measurement (qwopus: present) | KLD columns (C6) |
| `measurements[k].bench.{pp_ts,tg_ts,*_std}` | per measurement (qwopus: `null`) | Speed (P5) |
| `measurements[k].{kl_timeout,measurement_timeout,timeout_leg,measurement_invalid,corpus_path,incumbent}` | per measurement | Status/caveats; filter invalid (C2, C11) |
| `tiered[t].source` (`incumbent`/`evolved`) | per tier (newer runs) | "winner was a stock mixture" disclosure (C11) |
| `fixed_groups`, `proposed_band_histogram`, `predictor_tracking`, `llamacpp_binary`, `llamacpp_arch_check`, `probing_provenance` | top level | Caveats, compatibility, provenance |
| `sensitivity.json` `sensitivity`, `probes[*].{aggressive_scheme,probe_ppl}`, `normalized_weights` | `sensitivity.json` (qwopus: present) | Optional "why these groups are protected" table |
| v2 block `tensor_config`, `tensor_actual_types`, `actual_bytes`, `predicted_bytes`, `budget_bytes`, `ppl`, `baseline_ppl`, `algo` | `search_results.json` `tiered["BUDGET-…GiB"]` | v2 rows (C5) |
| `v2_results.json` `measurement`, `anchors[*]`, `kappa_provenance`, `failures`, `bandwidth`, `report_fit_affine` | `v2_results.json` | v2 evaluation, caveats, streamed bytes |
| `pareto.json` | output dir | All-measured table (P7) |
| `compare` `comparison.md`, `meta.json` | compare run dir | Task validation (P6) |
| GGUF KV `magicquant.group_schemes`, `general.file_type` | each shipped GGUF (`writer.py:1575-1600`) | File↔tier match (P2) |

### (b) Computed during a run but not persisted

| ID | Datum | Computed at | Where to persist | Contract impact |
|----|-------|-------------|------------------|-----------------|
| D-b1 | BF16 baseline size (`baseline_size_gb`) | `orchestrator.py:1143,2789` (`_estimate_model_size`) | New **top-level** key in `_save_results` | Additive top-level key. Not in the `_serialize_measurement` order contract. No v1 exact-key-set pin was found (`test_v2_search_characterization.py:254` pins **v2_results.json**, not v1). Still announce it, since Foundry/QAT read this file. |
| D-b2 | Full llama.cpp KL statistics (median, p95/p99/p99.9, "same top p", RMS Δp) | Printed by llama-perplexity; `_parse_kl_output` keeps only mean/max/p90 (`llamacpp.py:1588-1606`) | New keys **inside** the nested `kl` dict | Doesn't alter measurement key order (`kl` is one value). Round-trips through the checkpoint. Additive. |
| D-b3 | Generate-time verification PPL of the shipped file | `orchestrator.py:2871-2880` (logged only) | New sidecar `generated.json` (filename, bytes, sha256, verify_ppl, tier, config_key), or a tail-appended key | A sidecar avoids touching `search_results.json`. Preferred. |
| D-b4 | Writer per-tensor fallbacks (BF16→Q8_0 out-of-range, block fallbacks, SSM F32) | `writer.py:972-1337` `_fallbacks` | Same `generated.json` sidecar, plus optionally a GGUF KV `magicquant.fallbacks` | Sidecar: none. GGUF KV: additive metadata. |
| D-b5 | Double-quantization fact (`dequant_sources` breakdown) | `writer.py:1513-1525` | `generated.json` **and** a GGUF KV `magicquant.source_precision` so it travels with the file | Additive. |
| D-b6 | Exact CLI invocation, MagicQuant version/commit | argv at `cmd_search`/`cmd_generate`; `compare` already records git commit (`compare/output.py:535`) | Top-level `run_info` key, tail-appended | Additive top-level key. |

### (c) Not collected at all

| Datum | Would come from | Contract impact |
|-------|-----------------|-----------------|
| Calibration-corpus identity (path + SHA-256) and eval-corpus SHA-256 | `enable_imatrix` / `tools/build_calib_corpus.py` (already assert disjointness); `_resolve_data_file` | New keys in `measurement` (nested dict, additive) |
| Long-context quality (PPL/KLD at ctx ≥ 4k) | extra llama-perplexity leg at a second ctx | New measurement leg, which is heavy. Opt-in only. |
| Runtime memory (weights + KV at stated ctx) | static estimate from GGUF hparams (n_layer, n_kv_head, head_dim) × ctx × KV dtype | Card-time computation, no persistence |
| Per-backend throughput (CUDA/Metal/Vulkan/ROCm) | llama-bench on each backend | Out of scope for one box. The card should say "measured on ‹backend›" only. |
| Base-model license / repo id | user flag (`--license`, `--base-model owner/name`) or `config.json` `_name_or_path` | CLI only |

## 4. Proposed card (qwopus fixture, real numbers only)

Every number below comes from
`tests/fixtures/qwopus_search_results_2026-07-04/search_results.json`. GB = bytes/10⁹,
computed as GiB × 1.073741824 from the persisted `size_gb` (GiB). Exact byte counts
need the shipped files and are marked as placeholders. Anything the fixture doesn't
hold is shown as `‹not recorded›` or `‹…›`.

````markdown
---
base_model: ‹owner/base-model — pass --base-model›
base_model_relation: quantized
library_name: gguf
license: ‹inherit from base model — pass --license›
quantized_by: ‹pass --quantized-by›
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - magicquant
  - mixed-precision
---

# ‹Model name› — MagicQuant mixed-precision GGUF

Per-tensor-group mixed-precision GGUFs built with MagicQuant. Each tensor group
(embeddings, output head, attention Q / K+V / O, FFN up+gate / down, SSM) gets its
own quantization scheme, chosen by a measured search: candidate files were built
and their perplexity measured against this model's own BF16 baseline.
Methodology: [magiccodingman — MagicQuant Wiki](https://github.com/magiccodingman/MagicQuant-Wiki).

> **The quant label in each filename is a size band, not a recipe.** "Q5" means
> "the lowest-loss file measured in the Q5 size band". It may contain no Q5_K tensors
> at all (this Q5 contains none). Use the per-group table below to see what is
> actually inside each file.

> ⚠️ **Legacy tier labels.** This search ran under `tier_scheme_version=1`
> (current: 2), whose bands were wider. Under the current bands, the files labelled
> "Q6" and "Q5" here would both fall in the Q6 band (inferred from sizes; BF16 size
> ‹not recorded›). Choose by size and measured quality, not by label.

## Which file should I pick?

Sizes are file sizes only. Add KV cache (grows with context length) and runtime
overhead before comparing with your VRAM/RAM.

- **Closest to BF16:** `Q8` — 41.41 GB, +0.83% PPL, mean KLD 0.0066.
- **Best quality per GB in this run:** `Q6` — 24.86 GB, +1.17% PPL, mean KLD 0.0132.
  Note: 24.86 GB does **not** fit a 24 GB GPU with any context.
- **`Q5` (23.94 GB) is only 0.92 GB smaller than `Q6`** but has ~2.3× the mean KLD
  (0.0308 vs 0.0132). Prefer `Q6` unless that 0.92 GB matters.
- **Smallest:** `Q4` — 17.92 GB, +4.00% PPL, mean KLD 0.0411.

## Files

| File | Label (size band) | Size (GB) | Size (GiB) | Bytes | SHA-256 | Status |
|------|------|----------:|-----------:|------:|---------|--------|
| ‹filename› | Q8 | 41.41 | 38.57 | ‹from file› | ‹from file› | measured |
| ‹filename› | Q6 | 24.86 | 23.15 | ‹from file› | ‹from file› | measured |
| ‹filename› | Q5 | 23.94 | 22.29 | ‹from file› | ‹from file› | measured |
| ‹filename› | Q4 | 17.92 | 16.69 | ‹from file› | ‹from file› | measured |

## Measured quality

Baseline: BF16, PPL **6.5722** (provenance: measured).

| Label | Size (GiB) | PPL | ΔPPL | ΔPPL % | Mean KLD | p90 KLD | Speed (pp / tg t/s) | Status |
|------|-----:|-----:|-----:|-----:|-----:|-----:|------|------|
| Q8 | 38.57 | 6.627 | +0.054 | +0.83% | 0.0066 | 0.0108 | not measured | measured |
| Q6 | 23.15 | 6.649 | +0.077 | +1.17% | 0.0132 | 0.0216 | not measured | measured |
| Q5 | 22.29 | 6.679 | +0.107 | +1.62% | 0.0308 | 0.0553 | not measured | measured |
| Q4 | 16.69 | 6.835 | +0.263 | +4.00% | 0.0411 | 0.0723 | not measured | measured |

PPL uncertainty (±) ‹not recorded›. Differences under ‹noise floor not recorded› are
not meaningful. KLD is measured against the BF16 model's logits on the same corpus.
Lower is better.

## What is inside each file

| Label | E embed | H head | Q attn-Q | K attn-K/V | O attn-out | U FFN up/gate | D FFN down | S SSM |
|------|------|------|------|------|------|------|------|------|
| Q8 | BF16¹ | BF16¹ | Q6_K | Q8_0 | BF16¹ | BF16¹ | Q6_K | Q5_K |
| Q6 | BF16¹ | BF16¹ | Q8_0 | Q8_0 | BF16¹ | Q4_K_M | Q5_K | Q5_K |
| Q5 | BF16¹ | BF16¹ | Q8_0 | BF16¹ | BF16¹ | MXFP4_MOE | IQ4_NL² | Q5_K |
| Q4 | BF16¹ | Q8_0 | Q6_K | Q8_0 | Q8_0 | IQ3_S | IQ4_XS | IQ3_XXS |

¹ BF16-designated groups are stored as **F16** on disk (llama.cpp limitation).
Individual tensors outside F16 range fall back to Q8_0: ‹per-file fallback list not recorded›.
² IQ4_NL is only competitive when encoded with an importance matrix. imatrix for this
run: ‹not recorded›.

## Compatibility

- **Q5** contains MXFP4 (ggml type 39): requires a llama.cpp build with MXFP4
  support (‹minimum build›).
- **Q4/Q5** contain I-quants (IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL): they load everywhere
  llama.cpp does, but speed varies a lot by backend. Measure on your hardware.
- This model has SSM tensors (group S): requires a llama.cpp build that supports
  the ‹architecture› architecture. Measuring binary: ‹not recorded›.
- No fork-only (ROCmFPX) types are used. These files load on stock llama.cpp.

## Usage

```bash
llama-cli -m ‹filename› -p "…"          # or: llama-server -m ‹filename›
```
Ollama / LM Studio: point them at the downloaded .gguf as usual.

## How these numbers were produced

- Metric: perplexity (and KL divergence vs BF16 logits) from `llama-perplexity`.
  ΔPPL % = (PPL − baseline) / baseline.
- Eval corpus: ‹not recorded› · context: ‹not recorded› · chunks: ‹not recorded›
  (full corpus or capped: ‹not recorded›).
- Calibration (imatrix): ‹not recorded›. The calibration corpus is required to be
  disjoint from the eval corpus.
- Sensitivity probing: per-group probes with KL (see `sensitivity.json`). Probing
  provenance: ‹not recorded›.
- Search: measured evolutionary search, 7 configurations built and measured.
  Seed: none (run is not seed-reproducible).
- Task-level validation (`magicquant compare`): ‹not run›.

## Provenance

- Source: ‹base model repo› at ‹revision› (BF16). Re-quantized from a quantized
  source: ‹not recorded› (double-quantization disclosure).
- Built with MagicQuant ‹version/commit not recorded›; measured with llama.cpp
  ‹binary/build not recorded›.
- Raw results: `search_results.json` and `sensitivity.json` are published alongside
  these files.

## Caveats

- Perplexity on one corpus at one context length is a proxy. It doesn't measure
  instruction following, tool use, coding or long-context behaviour.
- Single run, no error bars: rankings between files whose ΔPPL differs by less than
  ‹noise floor› are not established.
- Tier labels follow legacy (v1) bands. See the note at the top.

---
Methodology credit: [magiccodingman — MagicQuant Wiki](https://github.com/magiccodingman/MagicQuant-Wiki).
````

## 5. Implementation sketch (priority order)

All of this is read-side, except where marked "persist". Nothing here changes
`naming.config_key`, `_serialize_measurement`'s key order, the
`tiered`/`tiered_survivors` split, or the `v2_results.json` key set.

1. **Honesty gates (C1, C2, C11), in `generate_model_card`.**
   - `_baseline_line(results)`: print only if `baseline_provenance == "measured"`.
   - `_tier_status(tier_info, measurement)`: returns `measured` /
     `measured (KL timed out)` / `predicted` / `failed` / `not measured`. Join with
     `results["measurements"].get(config_key(info["config"]))`, filtering out
     `measurement_invalid` (`.get()` truthiness).
   - Loss and ΔPPL only when status is measured and the baseline was measured.
   - Make the intro sentence conditional on `probing_provenance`.
   - Add a "Measurement caveats" block for `kl_timeout`, `measurement_timeout`,
     `source=="incumbent"`, `predictor_tracking`, `weights_degenerate`,
     `kappa_provenance` containing `imputed`.
2. **Units (C3).** Show the column as `Size (GiB)` + `Size (GB)`. When files are
   resolved (step 5), use the exact bytes.
3. **v2 support (C5).** Render non-standard tier keys after the fixed order. For
   `algo=="v2-budget"`, use `actual_bytes`/`ppl`/block `baseline_ppl`. In
   `cmd_card`, if `v2_results.json` exists, load it and pass it as an optional
   `v2_results=` argument (anchors, `measurement`, `failures`,
   `bandwidth.streamed_bytes`).
4. **KL and method (C6, C7).** Add mean/p90 KLD columns and an "Evaluation method"
   section from `measurement`, printing `‹not recorded›` where it's absent.
5. **Files and labels (C4, P2).** `cmd_card` globs `*.gguf` in `--output-dir`, reads
   `magicquant.group_schemes` with `magicquant.gguf.reader`, matches files to tiers by
   `config_key`, and computes bytes and SHA-256 (a new `--no-hash` flag for speed).
   Always emit the size-band-not-recipe callout.
6. **Front matter and CLI (P1).** New flags: `--license`, `--quantized-by`,
   `--pipeline-tag`, `--language` (repeatable). Validate `--base-model` as
   `owner/name` when front matter is on, and add `--no-front-matter` for plain output.
7. **Compatibility (P3).** Derive notes from `get_scheme_by_name(s).category` over the
   schemes present per file (mxfp4 / iq_quant / rocmfpx), plus `llamacpp_binary`.
8. **Upload gate (P8, C10).** `--upload` refuses unverified cards unless
   `--allow-unverified-card` is passed, and refuses a double-quant source unless
   `--acknowledge-double-quant` is passed.
9. **Presentation (R1–R3, P7).** Group legend and role ordering, sort rows by size,
   3-decimal PPL with ± when `ppl_err` is present, and an optional `<details>`
   all-measured table.
10. **Persist (separate PRs, each needing a deliberate decision):** D-b1
    `baseline_size_gb` (top-level, additive); D-b2 extra KL stats inside `kl`; D-b3/4/5
    a `generated.json` sidecar from `generate` (verify PPL, fallbacks, dequant, bytes,
    sha256); D-b6 a `run_info` key; corpus SHA-256s inside `measurement`. Each needs
    the CHANGELOG entry and, for `search_results.json`, a check of external readers
    (QAT `load_hybrid_config`, Foundry, `tools/reselect_tiers.py`).

### 5.4 Tests to add (`tests/test_model_card.py`)

- `baseline_provenance` in {`fabricated`, `prediction-only`}: the card contains no
  `Baseline (BF16) perplexity: 5.0` and no loss figures, and states "not measured".
- Prediction-only tier (`ppl: None`, `predicted_loss` set): row status reads
  `predicted`, size is marked predicted, and the string `"measured sensitivity"` is
  absent.
- The size header says `GiB`, and a known `size_gb` renders both GiB and GB correctly.
- `tiered` containing `BUDGET-24GiB` (shape from `v2/interchange.py`): the row is
  rendered and the table is never empty.
- qwopus fixture: Q5 row shows `0.0308` mean KLD, joined via `config_key`.
- `kl_timeout: True` / `measurement_timeout: True` surface in caveats, and a
  `measurement_invalid` measurement never appears in any table.
- `predicted_loss` never appears as a percentage.
- The size-band callout is always present. `magiccodingman` attribution is still
  present (existing assertion, keep it).
- The legacy banner is still emitted for files without `tier_scheme_version` (keep
  the existing two tests).
- `tiered` takes precedence over `tiered_survivors`, and an empty `tiered` falls back
  (pin the current behaviour).
- Front matter YAML parses (with the `[yaml]` extra) and contains
  `base_model_relation: quantized`.

### Must not change

- `naming.config_key` format; `_serialize_measurement` key orders; `tiered` vs
  `tiered_survivors` as separate keys; the `v2_results.json` exact key set
  (`tests/test_v2_search_characterization.py:254`). The card only *reads*.
- Never display `predicted_loss` or `residual` as quality numbers. They are in noise
  units.
- Keep the magiccodingman methodology credit and link.
- Renaming shipped files away from `Q5_K_M`-style suffixes (C4) is a separate human
  decision, because filenames are consumed downstream. The card change (disclosure)
  doesn't depend on it.
