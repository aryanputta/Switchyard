from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer shared by every retrieval route."""
    return TOKEN_RE.findall(text.lower())
