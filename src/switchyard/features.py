from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from switchyard.text import tokenize

_MODEL_NUMBER_RE = re.compile(r"\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{4,}\b")
_MONEY_RE = re.compile(r"\$\s?\d+(?:\.\d+)?|\b\d+\s?(?:dollars|usd)\b")
_NEGATION_RE = re.compile(r"\b(without|no|not|free of|excluding|except|minus)\b")
_QUESTION_RE = re.compile(r"\b(how|what|why|when|where|who|which|does|is|are|can)\b")
_COMPAT_RE = re.compile(r"\b(for|compatible|fits|works with|replacement for)\b")
_COLORS = {
    "black", "white", "red", "blue", "green", "silver", "gray", "grey",
    "gold", "pink", "purple", "yellow", "brown", "beige", "rose",
}


@dataclass(frozen=True)
class QueryFeatures:
    token_count: int
    char_count: int
    has_question: bool
    has_model_number: bool
    has_money: bool
    has_negation: bool
    has_compatibility: bool
    has_color: bool

    def as_dict(self) -> dict[str, float]:
        # Booleans cast to 0/1 so the same vector feeds both the rule router
        # and the learned (LightGBM) router in Phase 2.
        return {k: float(v) for k, v in asdict(self).items()}


def extract_features(query: str) -> QueryFeatures:
    lowered = query.lower()
    tokens = tokenize(query)
    return QueryFeatures(
        token_count=len(tokens),
        char_count=len(query),
        has_question="?" in query or bool(_QUESTION_RE.search(lowered)),
        has_model_number=bool(_MODEL_NUMBER_RE.search(lowered)),
        has_money=bool(_MONEY_RE.search(lowered)),
        has_negation=bool(_NEGATION_RE.search(lowered)),
        has_compatibility=bool(_COMPAT_RE.search(lowered)),
        has_color=any(t in _COLORS for t in tokens),
    )
