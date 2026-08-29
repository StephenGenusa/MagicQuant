"""Make this checkout's own llama.cpp build discoverable to the test suite.

The llama.cpp-dependent tests locate their tools through ``shutil.which()``,
i.e. through PATH. Running the suite as ``.venv/bin/python -m pytest`` -- the
form CLAUDE.md documents -- does not put the venv's bin directory on PATH, so
a build sitting in this repo's gitignored ``bin/llama.cpp/`` was invisible and
``tests/integration/test_encoder_parity.py`` skipped its 9 byte-for-byte
parity checks against ``llama-quantize``.

Prepending that directory here is a no-op anywhere it does not exist (CI
included), so this can only ever turn a skip into a real run, never the
reverse. An explicit ``LLAMA_QUANTIZE`` still wins: the test reads it before
falling back to ``which``.
"""

import os
from pathlib import Path

_LLAMA_BIN = Path(__file__).resolve().parent / "bin" / "llama.cpp"

if _LLAMA_BIN.is_dir():
    _path = os.environ.get("PATH", "")
    if str(_LLAMA_BIN) not in _path.split(os.pathsep):
        os.environ["PATH"] = f"{_LLAMA_BIN}{os.pathsep}{_path}" if _path else str(_LLAMA_BIN)
