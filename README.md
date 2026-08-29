# MagicQuant

**Evolutionary Tensor Search for Optimal LLM GGUF Hybrid Quantization**

A Python implementation of the MagicQuant framework — an evolutionary search algorithm that discovers optimal per-group quantization configurations for LLM GGUF files. Instead of applying one quantization scheme globally, MagicQuant assigns different schemes to different tensor groups (embeddings, attention, FFN) based on measured sensitivity, producing models that break the standard size/quality/speed Pareto frontier.

This repository is a clone of [lucasmcoleman/MagicQuant](https://github.com/lucasmcoleman/MagicQuant) that adds a few features:

- **`magicquant compare`** — side-by-side inference across every generated tier, automatically scored against a curated question pool (arithmetic, factual recall, multilingual, long-context, code, and more), with an HTML comparison report. Practical validation of what a tier actually loses, beyond perplexity.
- **`magicquant fix-metadata`** — in-place repair of GGUF metadata mismatches (e.g. `block_count` higher than the tensors actually present), without rewriting the file.

## Table of Contents

- [Origin & Credit](#origin--credit)
- [How It Works](#how-it-works)
  - [Tensor Groups](#tensor-groups)
  - [Supported Quantization Schemes](#supported-quantization-schemes)
- [Installation](#installation)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Analyze](#analyze)
  - [Probe](#probe)
  - [Search](#search)
  - [Generate](#generate)
  - [Importance Matrix (imatrix)](#importance-matrix-imatrix)
  - [Quantization-Aware Training (QAT-LoRA)](#quantization-aware-training-qat-lora)
  - [Manual Hybrid from YAML Config](#manual-hybrid-from-yaml-config)
  - [Python API](#python-api)
  - [Model Card](#model-card)
  - [Compare (side-by-side tier validation)](#compare-side-by-side-tier-validation)
  - [Fix Metadata](#fix-metadata)
- [Architecture](#architecture)
- [Configuration via Environment](#configuration-via-environment)
- [Docker](#docker)
- [Development](#development)
- [Known Limitations](#known-limitations)
- [License](#license)

## Origin & Credit

The quantization engine itself — the evolutionary search, v2 budget search, QAT-LoRA training, and libggml-backed encoders — is developed at [lucasmcoleman/MagicQuant](https://github.com/lucasmcoleman/MagicQuant), the upstream of this repository.

This project is an implementation of the **MagicQuant** methodology created by [**magiccodingman**](https://github.com/magiccodingman). The original research, empirical findings, and the "MXFP4 Anomaly" discovery are documented in the [MagicQuant Wiki](https://github.com/magiccodingman/MagicQuant-Wiki). Published hybrid models are available on HuggingFace:

- [Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant) — Verified best-of-the-best hybrid quants
- [MXFP4 Hybrid GGUF Collection](https://huggingface.co/collections/magiccodingman/mxfp4-hybrid-gguf) — Experimental MXFP4 hybrids

The core insight from the original research: the vast majority of parameters in a transformer (FFN layers) can tolerate aggressive MXFP4 compression, while a small set of "brain" layers (embeddings, attention output, LM head) need protection at higher precision. This "Carbon Fiber Body, Ferrari Engine" pattern consistently produces models that are smaller than Q4 but retain Q5/Q6 quality.

## How It Works

```
BF16 Source Model
     |
     v
[1. Sensitivity Probing] — Quantize one group at a time, measure KL divergence
     |
     v
[2. Evolutionary Search] — Discover optimal hybrid configs per compression tier
     |                      Protector/Crusher mutations, epsilon-greedy exploration
     v
[3. Tiered Generation]  — Output best Q4, Q5, Q6 hybrid GGUFs
     |                     Each with per-tensor quantization via GGUF writer
     v
  Q4: E:BF16 H:Q8 Q:Q6K K:Q8 O:BF16 U:MXFP4 D:MXFP4  (24 GB from 60 GB)
  Q5: E:BF16 H:BF16 Q:Q8 K:BF16 O:BF16 U:MXFP4 D:MXFP4 (29 GB from 60 GB)
  Q6: E:BF16 H:BF16 Q:Q6K K:Q8 O:BF16 U:BF16 D:Q8       (44 GB from 60 GB)
```

**A tier name is a size band, not a recipe.** Note above that the "Q4" config
contains no Q4 tensors at all, and the "Q6" contains BF16 and Q8. A tier is
whatever mix of schemes landed in that size band with the lowest measured loss.
Grading a build by whether it "contains N-bit tensors" is a category error;
size and measured quality are the only criteria. Bands are defined in
`quant/tiers.py` as ratios to the BF16 baseline, and `tools/reselect_tiers.py`
re-derives a finished run's ladder from its stored measurements.

Probes score by **KL divergence against saved reference logits**, not
perplexity. Perplexity cannot resolve a single-group probe: the change is far
below run-to-run noise, so probes came back undifferentiated and normalization
then flattened them to zero — on one MoE model the expert group, 93% of all
parameters, ended up with sensitivity weight exactly 0.000. KL measures the
same perturbation at roughly 79 sigma instead of 0.55.

### Tensor Groups

| Group | Role | Typical Sensitivity | Default Scheme |
|-------|------|-------------------|----------------|
| **E** | Token Embeddings | Very High | BF16 |
| **H** | LM Head | Very High | BF16 |
| **O** | Attention Output | High | Q8_0 / BF16 |
| **Q** | Attention Query | Moderate | Q6_K / IQ4_NL |
| **K** | Attention Key/Value | Moderate | Q8_0 / Q6_K |
| **U** | FFN Up/Gate | Low (robust) | **MXFP4** |
| **D** | FFN Down | Low (robust) | **MXFP4** |
| **X** | MoE Experts | Very Low | MXFP4 |
| **R** | MoE Router | High | Q8_0 |

### Supported Quantization Schemes

The default search pool (what `search` samples from with no extra flags):

| Scheme | Type | bpw | Noise (heuristic) | Best For |
|--------|------|-----|-------|----------|
| BF16 | Float | 16.0 | 0.0 | Brain layers (E, H, O) |
| Q8_0 | Integer | 8.5 | 1.0 | Near-lossless protection |
| Q6_K | K-quant | 6.56 | 2.2 | High-quality attention |
| Q5_K | K-quant | 5.5 | 3.0 | Balanced attention |
| IQ4_NL | Non-linear | 4.5 | 3.8 | Best ~4-bit quality **with an imatrix** — dropped from the pool without one (see Known Limitations) |
| **MXFP4** | **FP4 (E2M1)** | **4.25** | **4.0** | **FFN / MoE experts** |
| Q4_K_M | K-quant | 4.5 | 4.5 | Fallback integer 4-bit |
| Q3_K | K-quant | 3.44 | 8.0 | Aggressive attention / robust-group compression |
| Q2_K | K-quant | 2.625 | 15.0 | Most aggressive robust-group compression |

MXFP4 implements the [OCP MX Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) FP4 format (E2M1 values with shared E8M0 exponent). Its non-uniform quantization levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near zero, naturally matching the Gaussian-like weight distribution of transformers — producing lower noise than integer Q4 at better compression.

`quant/schemes.py`'s registry holds 23 schemes total; the 9 above are the
default pool. The other 14 are real, working schemes but are **not**
uniformly "supported" in the same sense — each is gated behind an explicit
opt-in or a hardware/calibration requirement:

- **`IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_M`, `IQ1_S`**
  — opt-in via `enable_iq` (search's `--enable-iq`). `IQ2_XS`, `IQ2_XXS`, and
  `IQ1_S` additionally *require* an imatrix to encode at all.
- **`Q4_0`, `Q4_1`** — legacy block=32 quants, v2-only (`--target-profile
  q4nx`); excluded from v1's random-config sampling to keep the seed-pinned
  search fixture stable.
- **`ROCMFP3`, `ROCMFP4`, `ROCMFP6`, `ROCMFP8`** — AMD-native fork schemes,
  opt-in via `enable_rocmfpx`. Encoding requires a
  [ROCmFPX](https://github.com/ciru-ai/ROCmFPX) build of libggml, and the
  resulting GGUF loads only on that fork, not stock llama.cpp.

> **Note on BF16:** Groups designated `BF16` (brain layers E/H/O) are stored on
> disk as **F16**, not BF16. llama.cpp's compute graph has incomplete BF16
> support, so the writer downgrades to F16 (and logs a one-time warning).
> Values outside the F16 range (|x| > 65504, or subnormals) may become Inf/0.

## Installation

```bash
git clone https://github.com/lucasmcoleman/MagicQuant.git
cd MagicQuant
pip install -e .
```

Requires Python 3.9+ and NumPy. Optional: [llama.cpp](https://github.com/ggerganov/llama.cpp) for real perplexity measurement during probing.

For **Quantization-Aware Training** (`magicquant qat`) install the optional
`[qat]` extra, which pulls the heavy training stack (the core install stays
torch-free):

```bash
pip install -e ".[qat]"   # torch, transformers, peft, accelerate
```

> `torch` must be the build that works on your hardware (the ROCm wheel on AMD,
> CUDA on NVIDIA, or CPU). Install the matching wheel first if the default index
> doesn't provide it.

## Usage

### Full Pipeline

```bash
# 1. Analyze model structure
magicquant analyze model-bf16.gguf

# 2. Run sensitivity probing (uses heuristics if llama.cpp not available)
magicquant probe model-bf16.gguf --output-dir ./output

# 3. Run evolutionary search
magicquant search model-bf16.gguf --output-dir ./output --generations 50

# 4. Generate best Q4, Q5, Q6 hybrid GGUFs
magicquant generate model-bf16.gguf --output-dir ./output --tiers Q4,Q5,Q6
```

### Analyze

Inspect a GGUF's structure without touching it: architecture, tensor count, and how every tensor classifies into the sensitivity groups (E/H/Q/K/O/U/D/S/N/V/X/R) that the rest of the pipeline operates on. Run this first on an unfamiliar model — if a tensor lands in the wrong group here, every downstream decision inherits the error.

```bash
magicquant analyze model-bf16.gguf
```

Takes only the model path; no flags.

### Probe

Measure per-group sensitivity: quantize one tensor group at a time, score the damage, and write the resulting sensitivity weights to `sensitivity.json` in the output directory. `search` runs probing automatically, so the standalone command is for inspecting sensitivities on their own or pre-computing them for a later search.

```bash
magicquant probe model-bf16.gguf --output-dir ./output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir PATH` | `./output` | Where `sensitivity.json` is written |
| `--baseline-ppl FLOAT` | `5.0` | Baseline perplexity of the uncompressed model |
| `--llamacpp-path PATH` | auto-detect | Path to the llama.cpp directory |

With llama.cpp available, probes are measured (KL divergence against saved reference logits); without it, heuristic estimates are used.

### Search

The core command: discover the best hybrid config per compression tier. Two algorithms share the entry point:

- **v1 (default)** — evolutionary Predict→Build→Measure→Learn search over per-group configs. With `--rounds N` (default 3), each round actually builds the top candidates, measures them with llama-perplexity, and feeds the residuals back into the predictor. `--rounds 0` is prediction-only and needs no llama.cpp.
- **v2 (`--algo v2 --budget-gb N`)** — budget-constrained per-tensor allocation: builds a per-tensor × per-scheme distortion table through libggml, then solves an exact knapsack for a target size, verifying only 2–3 frontier anchors with full perplexity.

```bash
# Measured evolutionary search
magicquant search model-bf16.gguf --output-dir ./output --rounds 3

# Budget search: best possible model that fits in 12 GiB
magicquant search model-bf16.gguf --algo v2 --budget-gb 12
```

Results land in `search_results.json` (consumed by `generate`, `qat`, and `card`). Every flag also has a `MAGICQUANT_*` environment-variable equivalent.

Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds N` | `3` | Measurement rounds (0 = prediction only) |
| `--generations N` | `30` | Evolutionary generations |
| `--population N` | `80` | Population size |
| `--candidates N` | `4` | Candidates built and measured per round |
| `--patience N` | off | Early-stop after N generations without improvement |
| `--use-imatrix` | off | Capture/reuse an importance matrix and weight candidate builds with it |
| `--enable-kl` | off | Blend measured KL-divergence-to-base into survivor selection |
| `--enable-iq` | off | Add IQ-family quant types to the search pool |
| `--enable-speed-bench` | off | Measure real tokens/sec per candidate via llama-bench |
| `--seed N` | unset | Reproducible search |
| `--dry-run` | — | Validate config and source model, then exit |

v2-only flags (with `--algo v2`):

| Flag | Default | Description |
|------|---------|-------------|
| `--budget-gb N` | required | Target model size in GiB (weights only) |
| `--anchors N` | `2` | Frontier points to build and verify with full-corpus perplexity |
| `--floor GROUP=SCHEME` | none | Minimum scheme per group, repeatable (e.g. `--floor E=Q6_K`) |
| `--target-profile q4nx` | — | Restrict schemes to a serving container's packable types |
| `--probe-mode single\|cumulative` | `single` | How group-calibration probes measure damage |
| `--no-group-probes` | — | Skip measured calibration (pure surrogate allocation) |

Run `magicquant search --help` for the full list, including measurement caps, speed weighting, calibration I/O, and the ROCmFPX fork types.

### Generate

Build the actual hybrid GGUFs from a finished search's `search_results.json`. Each requested tier gets the best measured config from the search, written with per-tensor quantization via the streaming GGUF writer.

```bash
magicquant generate model-bf16.gguf --output-dir ./output --tiers Q4,Q5,Q6
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir PATH` | `./output` | Directory containing `search_results.json`; hybrids are written here |
| `--tiers LIST` | `Q4,Q5,Q6` | Comma-separated tiers to generate |
| `--target-quant NAME` | `MXFP4_MOE` | Target base quantization |
| `--verify` | off | Measure perplexity of each generated hybrid |
| `--adapter PATH` | none | LoRA adapter directory to merge on-the-fly during the write |
| `--llamacpp-path PATH` | auto-detect | Path to the llama.cpp directory (for `--verify`) |

Remember that a tier name is a size band, not a recipe — see [How It Works](#how-it-works).

### Importance Matrix (imatrix)

`search`/`generate` already capture and apply an importance-weighted imatrix
automatically by default, from a bundled calibration corpus (see Known
Limitations below) — most users never need to run this directly.
`magicquant imatrix` is the manual/advanced entry point: capture one against
your own corpus once and reuse the result, or produce a file for the Python
API's `create_hybrid_gguf(..., imatrix=...)`.

```bash
magicquant imatrix model-bf16.gguf \
    -f my-corpus.txt \
    -o model-bf16.imatrix.gguf \
    --chunks 200 --ctx-size 512
```

- `model` (positional) — GGUF model to instrument.
- `-f`/`--corpus` (required) — plain-text calibration corpus (e.g. a
  wikitext-2 **train** split). Must be a different file than whatever corpus
  you score perplexity against — calibrating and evaluating on the same text
  makes every measured loss optimistic with nothing in the output revealing
  it.
- `-o`/`--output` — output imatrix GGUF (default: `<model>.imatrix.gguf`).
- `--chunks` — max corpus chunks to process (default `-1` = all).
- `--ctx-size` — chunk length in tokens (default `512`).

### Quantization-Aware Training (QAT-LoRA)

`magicquant qat` fine-tunes a model to be robust to a chosen per-group hybrid
config *before* it ships as that hybrid. It freezes the base model,
fake-quantizes it to the search's per-group schemes in the forward pass (a
differentiable per-scheme fake-quant with a straight-through estimator, validated
against libggml), and trains LoRA adapters that compensate. Largest benefit at the
aggressive tiers (Q2/Q3/MXFP4). Requires the `[qat]` extra (see Installation).

```bash
# Train adapters robust to the Q4 tier from a prior search's results.
magicquant qat ./my-model \
    --config ./output/search_results.json \
    --tier Q4 \
    --dataset data/chat.jsonl \
    --out ./output/qat_adapters \
    --lora-r 32 --lora-alpha 64 --epochs 1 --lr 2e-4
```

The per-group hybrid config comes from a prior `search`'s `search_results.json`
(`--config` + `--tier`); the dataset is a chat JSONL (`{"messages": [...]}` per
line) trained with completion-only loss. Adapters + a `qat_meta.json` (base model,
scheme-by-group, hyperparams, config hash) are written to `--out`. Merge the
adapters with `magicquant qat-merge`, then pack the exact hybrid with
`magicquant generate`. In Foundry this is surfaced as the **QAT** pipeline
stage (toggle + config + live logs).

```bash
# Merge the trained adapters into the base model's safetensors (streamed
# shard-by-shard; never materializes the full model in memory).
magicquant qat-merge ./my-model \
    --adapters ./output/qat_adapters \
    --out ./output/qat_merged
```

- `base_model` (positional) — HF model id or local path to the base model
  whose safetensors get merged (the same model QAT was run against).
- `--adapters` (required) — adapter directory written by `magicquant qat`
  (needs `adapter_model.safetensors` + `qat_meta.json`).
- `--out` (required) — output directory for the merged safetensors model.

Use this rather than a generic PEFT merge — MagicQuant's adapter key naming
doesn't match PEFT's convention, and a generic merge has silently produced an
unmodified copy of the base model in the past. `qat-merge` fails loudly
instead of degrading to a no-op.

> **Budget builds interoperate here too.** `search --algo v2 --budget-gb <N>`
> now also merges a `BUDGET-<N>GiB` pseudo-tier into `search_results.json`,
> carrying a per-group projection alongside the exact per-tensor allocation.
> So `--tier BUDGET-<N>GiB` works with QAT unchanged, and Foundry's ROCmFPX
> `mq-budget` mode consumes the same block — no separate interchange format.
> The merge is additive: an existing file's Q4/Q5/Q6 tiers are left untouched,
> and a legacy file without a version stamp does not gain one.

**Validated result.** In a confound-controlled run on Qwen2.5-0.5B base with an
aggressive Q4_K-attention/MXFP4-FFN hybrid — bf16 PPL 16.35, plain quant 19.54
(+3.19 damage), quant+QAT 15.13, and a bf16+identical-LoRA control 13.46 — the
quant-vs-bf16 gap shrank from +3.19 to +1.67 once the LoRA's own domain adaptation
is held fixed on both arms. **QAT recovered 47.5% of the quantization loss beyond
plain LoRA domain-adaptation.** Recovery scales with quant aggressiveness, and the
final GGUF pack is exact-ggml (byte-identical to llama.cpp) even though training
uses a faithful torch fake-quant. See [`docs/qat.md`](docs/qat.md) for the full
methodology, the fake-quant/STE design, multimodal/bf16 support, and caveats.

### Manual Hybrid from YAML Config

```yaml
# config.yaml
model:
  name: Qwen3-30B-A3B
  source: ./Qwen3-30B-A3B-BF16.gguf
quantization:
  base: MXFP4_MOE
  groups:
    E: BF16
    H: BF16
    O: Q8_0
    Q: IQ4_NL
    K: IQ4_NL
```

```bash
magicquant hybrid config.yaml --output-dir ./output
```

### Python API

```python
from magicquant.gguf.writer import create_hybrid_gguf

create_hybrid_gguf(
    output_path="model-hybrid.gguf",
    base_model_path="model-bf16.gguf",
    quant_config={
        "base": "MXFP4_MOE",
        "groups": {
            "E": "BF16",
            "H": "BF16",
            "O": "Q8_0",
            "Q": "IQ4_NL",
            "K": "IQ4_NL",
        }
    }
)
```

### Model Card

`magicquant card` writes a HuggingFace-ready model card summarizing a
finished search's tier results:

```bash
magicquant card --output-dir ./output --model model-bf16.gguf --base-model org/Model-Name
```

- `--output-dir` — directory containing `search_results.json` (default `./output`).
- `--model` — source model path, used to derive the card title.
- `--base-model` — base model name to show on the card.
- `--upload` — upload the generated card to HuggingFace (requires
  `huggingface_hub`, `pip install 'magicquant[hf]'`); needs `--repo`.
- `--repo` — target HF repo id (`owner/name`) for `--upload`.

### Compare (side-by-side tier validation)

Run side-by-side inference across every GGUF in an output directory, score the responses automatically, and write an HTML comparison report. This is the practical validation step after `generate` — it shows which quantization tier first starts to fail on tasks that matter, rather than relying solely on perplexity as a quality proxy.
![](./Screenshot%20from%202026-06-24%2021-44-29.png)

```bash
magicquant compare [OPTIONS]
```

| Flag | Default     | Description |
|------|-------------|-------------|
| `--output-dir PATH` | `./output`  | Directory containing GGUFs to compare and where results are written |
| `--questions-file PATH` | bundled     | YAML question pool (default: built-in 20-question set) |
| `--question-count INT` | `20`        | Questions to use, sampled proportionally from easy / medium / hard |
| `--max-tokens INT` | `6000`      | Maximum tokens to generate per response. **Thinking models** (Qwen3, DeepSeek-R1, etc.) emit `<think>` chain-of-thought that can consume 1 000–3 000 tokens before the final answer is written; lower values silently truncate the response and produce wrong scores |
| `--context-size INT` | `8192`      | Context window size. Auto-expanded when passage-based questions need more. Must comfortably exceed `--max-tokens` plus prompt length |
| `--n-samples INT` | `1`         | Inference samples per question. Use with `--temperature >0` to measure answer consistency |
| `--temperature FLOAT` | `0.0`       | Sampling temperature (0 = greedy / deterministic) |
| `--top-p FLOAT` | `1.0`       | Nucleus sampling cutoff |
| `--top-k INT` | `0`         | Top-k sampling (0 = disabled) |
| `--system-prompt TEXT` | built-in    | Override the default system prompt for all questions |
| `--reasoning-mode` | false       | Switch to a step-by-step reasoning system prompt — useful for models with a built-in thinking mode |
| `--llamacpp-path PATH` | auto-detect | Path to the llama.cpp directory |

**What it measures.** The built-in question pool targets eight failure modes — cognitive axes where quantization noise is most likely to produce wrong or degraded answers:

| Failure Mode | What It Tests |
|--------------|---------------|
| `arithmetic` | Multi-step calculation, compound interest, train-meeting problems |
| `factual_recall` | Exact factual knowledge (capitals, historical facts, constants) |
| `multilingual` | Non-English prompts with expected non-English answers (Chinese, Russian, Farsi) |
| `long_context` | Reading comprehension from a prepended passage (500–4 000 tokens) |
| `multi_hop` | Chained inference with a known cognitive trap (bat-and-ball, CRT) |
| `code` | Python function generation and SQL query construction |
| `proof` | Formal mathematical argument requiring structured reasoning |
| `instruction_following` | Structured output, attention-mechanism explanation, multi-part response |

Questions are stratified: 5 easy, 7 medium, 8 hard. When fewer than 20 questions are requested with `--question-count`, the proportions are preserved. Each question has a `scoring_type` that drives automated evaluation:

| Scoring Type | How It Works |
|--------------|--------------|
| `exact_numeric` | Extracts the last number in the response and compares to ground truth with optional tolerance. Reports **near_miss** if within 2× tolerance (or ±5% when tolerance is zero) |
| `quadratic_roots` | Extracts all numbers and fractions, checks both roots are present order-independently within tolerance |
| `keyphrase` | All required phrases must appear as case-insensitive substrings |
| `code_syntax` | Extracts fenced Python blocks and validates with `ast.parse` |
| `none` | Open-ended; response preserved for manual review |

Every scored response returns one of four statuses:
- **pass** — matches ground truth within tolerance
- **near_miss** — quantization degraded precision but the model "knew" the answer (informational; counted as fail in summary scores)
- **fail** — wrong or no answer
- **unscored** — open-ended question (proofs, explanations), manual review required

**Output.** Each run writes to `{output-dir}/comparisons/{timestamp}/`:

```
comparisons/
  20260624T143021/
    comparison.html    ← score bars, side-by-side responses, breakdown tables, reproducibility footer
    comparison.md      ← same content in Markdown for git-friendly diffs
    meta.json          ← git commit, CLI args, question-file SHA-256, all inference parameters
    raw_responses/
      model-Q4.gguf/Q01.json … Q20.json   ← per-question samples + scores for scripting
      model-Q5.gguf/Q01.json … Q20.json
```

A `comparison_latest` symlink in `output-dir` always points to the most recent run, so `open output/comparison_latest/comparison.html` works without knowing the timestamp.

The HTML report includes:
- A **summary table** with total score, per-failure-mode breakdown, and optional PPL (pulled from `search_results.json` if present)
- A **per-question table** with colour-coded score bars above each response box — scroll-to-answer behaviour so long chain-of-thought preambles don't bury the result
- A **failure-mode breakdown table** showing pass/fail counts per axis per model — useful for spotting that a Q4 hybrid loses arithmetic precision before it loses factual recall
- A **reproducibility footer** (expandable) with the exact CLI invocation, git commit, and question-file hash

**Typical workflow:**

```bash
# 1. Generate hybrid tiers
magicquant generate model-bf16.gguf --output-dir ./output

# 2. Compare all tiers across the full question set
magicquant compare --output-dir ./output

# 3. Open the report
xdg-open ./output/comparison_latest/comparison.html
```

**Consistency testing.** With `--n-samples 3 --temperature 0.3`, each question is answered three times. A consistency score (fraction of samples sharing the majority status) is reported alongside the primary answer — useful for detecting models that are "technically correct" on average but unreliable on repeated runs.

**Custom question pools.** Point `--questions-file` at any YAML file matching the built-in schema to test domain-specific tasks — code review, legal reasoning, clinical notes, etc. The schema supports per-question passage files, per-question system-prompt overrides, and all five scoring types.

**GPU acceleration.** Inference runs through `llama-cpp-python` (a core dependency). Each model is loaded once; all questions are answered in a single session, so inference time scales with the number of models rather than the number of questions. Install the CUDA build for models >3B:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Fix Metadata

Repair GGUF metadata mismatches in-place.

```bash
magicquant fix-metadata model.gguf [--dry-run] [-y]
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Show proposed metadata patches without modifying the file |
| `-y`, `--yes` | Apply patches without a confirmation prompt |

When to use: if llama.cpp reports `missing tensor 'blk.N.attn_norm.weight'` on a model
that was converted from HuggingFace, the GGUF `block_count` metadata may be higher than the
actual number of layer tensors present. This is common with Qwen3.5 and other architectures
where multi-token prediction (MTP/nextn) layers are dropped during conversion but the metadata
count is not corrected.

Run with `--dry-run` first to review the proposed changes, then without it (or with `-y`) to apply.
MagicQuant's GGUF writer automatically corrects `block_count` in any model it generates;
this command repairs existing source files.

## Architecture

```
magicquant/
  gguf/
    reader.py          — GGUF binary parser
    writer.py          — Hybrid GGUF writer (two-pass streaming)
    source.py          — ModelSource: GGUF / safetensors / LoRA-merged input abstraction
    tensor_groups.py   — Tensor group classification (E, H, Q, K, O, U, D, S, N, V, X, R)
  quant/
    schemes.py         — Quantization scheme registry (single source of truth for scheme metadata)
    converters.py      — encode_to_ggml_bytes() dispatch (single encoder source of truth)
    ggml_binding.py    — ctypes binding to libggml (byte-identical to llama-quantize)
    ggml_facts.py      — Block/type-size tables derived from ggml_binding
    tiers.py           — Tier (size-band) boundary classification
    calibration.py     — Optional calibrated noise/speed-multiplier overrides
  evolution/
    probing.py         — Sensitivity measurement (real or heuristic)
    predictor.py       — Loss/size/speed prediction with collapse penalties
    survival.py        — Evolutionary search with Protector/Crusher mutations
  v2/                  — Budget search (--algo v2, NON-DEFAULT; see docs/redesign.md)
    sensitivity.py     — Per-tensor x per-scheme distortion table (imatrix-weighted)
    calibrate.py       — Group-probe kappa fitting (amplification factors)
    allocate.py        — Multiple-choice knapsack (Lagrangian greedy + polish)
    resolve.py         — Per-tensor scheme resolution -> writer override map
    interchange.py     — search_results.json BUDGET-<N>GiB pseudo-tier interop
    outcome.py         — Frontier/measurement bookkeeping
    search.py          — v2 orchestration entry point
  qat/                 — Quantization-Aware Training (QAT-LoRA); needs the [qat] extra
    fake_quant.py      — Differentiable per-scheme fake-quant + STE (vs libggml)
    wrap.py            — QATLinear (fake-quants merged base+LoRA) + wrap_model
    expert_wrap.py     — Fused 3-D MoE expert QAT (per-expert LoRA via parametrization)
    names.py           — HF module -> GGUF tensor name mapping (reuses source.py)
    config.py          — load_hybrid_config / load_tensor_config (search_results.json -> schemes)
    diskmap.py         — Base-weight map for adapter-key reconciliation preflight
    merge.py           — Streaming on-disk base+adapter merge; backs `magicquant qat-merge`
    train.py           — run_qat: the QAT-LoRA loop (completion-only) + adapters
    validate.py        — perplexity comparison (QAT hybrid vs plain hybrid)
  utils/
    llamacpp.py        — llama.cpp integration for perplexity measurement
    naming.py          — Hybrid model filename generation
    measurement.py     — Shared measurement/statistics helpers
    model_card.py      — HuggingFace model card generation; backs `magicquant card`
  imatrix.py           — Importance-matrix capture via llama-imatrix; backs `magicquant imatrix`
  incumbents.py        — Best-known-config tracking across search rounds
  pareto.py            — Pareto-frontier utilities
  config.py            — Settings from env vars / .env (single source of truth for defaults)
  logging.py           — structlog configuration
  orchestrator.py      — Pipeline coordination
  __main__.py          — CLI entry point
```

`v2/` is the 2026-07 budget-search redesign (`--algo v2 --budget-gb`); the v1
evolutionary path above remains the default. `qat/_ggml_ref.py` (a private
reference helper, not part of the public API) is omitted from this tree.

## Configuration via Environment

The `search` command (and `search --dry-run`) reads settings from environment
variables with the `MAGICQUANT_` prefix or a `.env` file; explicit CLI flags
override them. Defaults below are the single source of truth (`config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MAGICQUANT_SOURCE_MODEL_PATH` | (required) | Path to source model |
| `MAGICQUANT_OUTPUT_DIR` | `./output` | Output directory |
| `MAGICQUANT_LLAMACPP_PATH` | auto-detect | Path to llama.cpp |
| `MAGICQUANT_TARGET_BASE_QUANT` | `MXFP4_MOE` | Base quantization scheme |
| `MAGICQUANT_SEARCH_GENERATIONS` | `30` | Generations per round |
| `MAGICQUANT_POPULATION_SIZE` | `80` | Candidates per generation |
| `MAGICQUANT_MEASUREMENT_ROUNDS` | `3` | Build-measure-learn cycles |
| `MAGICQUANT_TIERS` | `["Q4","Q5","Q6"]` | Compression tiers |

## Docker

```bash
docker build -f docker/Dockerfile -t magicquant:latest .
docker run --rm -v ./output:/app/output magicquant:latest search /data/model.gguf
```

## Development

```bash
pip install -e ".[dev]"
make test    # Run pytest suite
make lint    # Ruff lint (magicquant/ + tests/)
make clean   # Remove build artifacts
```

## Known Limitations

- Tokenizer reading only handles BPE (tokenizer.json); SentencePiece (.model) is not supported
- Source models must be BF16/F16/F32 — pre-quantized sources are rejected with a clear error
- `IQ4_NL` is excluded from the search pool when no importance matrix is
  available. Uncalibrated it lost all 11 measured comparisons across two 27B
  models (3-20x worse than same-bpw MXFP4/Q4_K_M) despite *better* isolated
  weight-reconstruction error, because its lookup table places levels to
  minimise unweighted error. With an imatrix it is sampled normally.

> Encoding is byte-identical to llama.cpp — MagicQuant calls libggml directly
> rather than reimplementing the encoders, so there is no MSE quality gap.
> Importance-matrix weighting is supported and **on by default**: the bundled
> calibration corpus is ~1 MB spanning 18 languages plus code, math and agentic
> prompts, capture is bounded to 200 chunks, and the result is cached per model.
> The corpus is verified disjoint from the perplexity eval corpus, and
> calibration is refused outright if the two ever resolve to the same file —
> scoring a run on the text it was calibrated against would make every measured
> loss optimistic with nothing in the output revealing it.

## License

MIT. The MagicQuant methodology and research are credited to [magiccodingman](https://github.com/magiccodingman).
