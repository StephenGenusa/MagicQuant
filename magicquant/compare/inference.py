"""
Batch inference for tier comparison via llama-cpp-python.

The model is loaded once per (model, system-prompt) group and all prompts
are answered in that single session, so wall-clock scales with the number
of models rather than the number of questions.  Each prompt gets an
independent context; no cross-question state leaks between answers.
"""

import os
from pathlib import Path
from typing import List, Optional


def _default_n_gpu_layers() -> int:
    """GPU layers to offload (-ngl). MAGICQUANT_NGL overrides; default 99 = full offload."""
    raw = os.environ.get("MAGICQUANT_NGL")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return 99


def run_inference_batch(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 200,
    ctx_size: int = 4096,
    system_prompt: str = "",
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    n_gpu_layers: Optional[int] = None,
) -> List[List[Optional[str]]]:
    """Load the model once and answer all prompts via llama-cpp-python.

    Model loads: 1 (vs len(prompts) with per-prompt subprocess calls).
    Returns list[list[str|None]]: outer index = question, inner = sample.

    Falls back to a list of [None] lists if llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  llama-cpp-python not installed. Run: pip install llama-cpp-python")
        return [[None] * n_samples for _ in prompts]

    if n_gpu_layers is None:
        n_gpu_layers = _default_n_gpu_layers()

    print(f"  Loading {Path(model_path).name}...", end="", flush=True)
    # Redirect C-level stderr during model load to suppress llama.cpp's
    # informational messages (e.g. "n_ctx_seq < n_ctx_train") that bypass
    # Python logging and break up the "Loading... ready" status line.
    old_stderr_fd = os.dup(2)
    with open(os.devnull, "wb") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=ctx_size,
                verbose=False,
            )
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
    print(" ready")

    messages_base = [{"role": "system", "content": system_prompt}] if system_prompt else []
    results: List[List[Optional[str]]] = []

    for question in prompts:
        msgs = messages_base + [{"role": "user", "content": question}]
        samples: List[Optional[str]] = []
        for _ in range(n_samples):
            try:
                out = llm.create_chat_completion(
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                text = out["choices"][0]["message"]["content"]
                samples.append(text.strip() or None)
            except Exception as e:
                print(f"\n  Inference error: {e}")
                samples.append(None)
        results.append(samples)

    del llm
    return results
