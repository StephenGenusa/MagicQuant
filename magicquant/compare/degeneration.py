"""Flag degenerate answers: the "the model spat out trash" failures that
neither the score nor KL divergence shows.

KL divergence averages over a whole corpus, so one answer that collapses into
a loop barely moves it (Llama-3.2-1B: bartowski Q8_0 had the lowest KLD of any
quant and still looped on a Russian question). The score only covers scored
questions, so a looping proof (unscored) is invisible there too.

Flags are mechanical and deliberately conservative:

- ``loop``: some 40-character passage repeats >= LOOP_MIN_REPEATS times
  (code blocks excluded). Calibrated on 476 real answers: normal answers,
  long explanations included, repeat a passage at most 4 times; the
  degenerate ones 79-330 times.
- ``wrong_language``: a question with ``response_language`` answered mostly
  in another script.
- ``empty``: no answer at all.

Hitting the token limit is NOT a flag: a long, legitimate answer can reach it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

LOOP_WINDOW = 40
LOOP_MIN_REPEATS = 10

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_SCRIPT = {
    "zh": re.compile(r"[一-鿿]"),
    "ru": re.compile(r"[Ѐ-ӿ]"),
    "fa": re.compile(r"[؀-ۿ]"),
}
_LETTER = re.compile(r"[^\W\d_]")

FLAG_LABELS = {
    "loop": "repetition loop",
    "wrong_language": "wrong language",
    "empty": "empty answer",
}


def max_repeat(text: str, window: int = LOOP_WINDOW) -> int:
    """How many times the most repeated *window*-character passage occurs
    (whitespace normalised, fenced code removed)."""
    t = re.sub(r"\s+", " ", _CODE_BLOCK.sub(" ", text or "")).strip()
    if len(t) < window * 2:
        return 0
    counts = Counter(t[i:i + window] for i in range(len(t) - window + 1))
    return counts.most_common(1)[0][1]


def target_script_share(text: str, language: str) -> Optional[float]:
    """Share of letters written in *language*'s script, or None when the
    language is unknown or the text has no letters."""
    script = _SCRIPT.get(language)
    letters = _LETTER.findall(text or "")
    if script is None or not letters:
        return None
    return sum(1 for ch in letters if script.match(ch)) / len(letters)


def detect_flags(response: Optional[str], question: dict) -> List[str]:
    """Flags for one answer, in a stable order (see FLAG_LABELS)."""
    text = (response or "").strip()
    if not text or text == "(no response)":
        return ["empty"]
    flags = []
    if max_repeat(text) >= LOOP_MIN_REPEATS:
        flags.append("loop")
    share = target_script_share(text, question.get("response_language") or "")
    if share is not None and share < 0.5:
        flags.append("wrong_language")
    return flags
