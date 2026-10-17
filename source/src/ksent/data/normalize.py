"""Fixed, rule-based text normalization (applied before splitting; not data-fitted).

Order (configs/data.yaml):
  1. ``html.unescape``
  2. Unicode NFC. NOT NFKC, which remaps compatibility jamo (ㅋ ㅠ ㅎ, U+3131..) to
     the conjoining-jamo block and can split identical-looking tokens.
  3. Remove zero-width characters (U+200B/C/D, U+FEFF) and control characters
     (Unicode category Cc) except whitespace.
  4. Lowercase Latin letters only (Hangul and other scripts are untouched).
  5. Cap any run of >= 3 identical characters to 2 (ㅋㅋㅋㅋ -> ㅋㅋ, !!!!! -> !!).
  6. Collapse whitespace and strip.

Digits, punctuation, emoji, jamo-only expressions (ㅋㅋ, ㅠㅠ, ㅅㅂ), English words and
profanity are kept. No spacing correction, stopword removal or digit replacement
("10점 만점에 1점" is genuine review content).
"""
from __future__ import annotations

import html
import re
import unicodedata

ZERO_WIDTH = "​‌‍﻿"
_ZW_RE = re.compile(f"[{ZERO_WIDTH}]")
_LATIN_UPPER_RE = re.compile(r"[A-Z]")
_WS_RE = re.compile(r"\s+")


def _remove_controls(text: str) -> str:
    return "".join(c for c in text if not (unicodedata.category(c) == "Cc" and not c.isspace()))


def cap_repeats(text: str, max_repeat: int = 2) -> str:
    """Cap runs of more than ``max_repeat`` identical characters to ``max_repeat``."""
    return re.sub(r"(.)\1{%d,}" % max_repeat, lambda m: m.group(1) * max_repeat, text, flags=re.S)


def normalize(text: object, max_repeat: int = 2) -> str:
    """Apply the six normalization rules. Non-strings become ``""``."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = _ZW_RE.sub("", text)
    text = _remove_controls(text)
    text = _LATIN_UPPER_RE.sub(lambda m: m.group(0).lower(), text)
    text = cap_repeats(text, max_repeat)
    return _WS_RE.sub(" ", text).strip()
