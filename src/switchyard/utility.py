from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityWeights:
    """Weights that convert a route's quality and cost into one scalar the
    router maximizes. latency_penalty is per millisecond. With latency_penalty
    at zero the router always picks the highest-quality route, which is exactly
    the always-rerank baseline. As the penalty grows, cheaper routes win on
    queries where they tie on quality."""

    latency_penalty: float = 0.004
    violation_penalty: float = 0.5
    timeout_penalty: float = 1.0


def utility(
    ndcg: float,
    latency_ms: float,
    *,
    violations: int = 0,
    timed_out: bool = False,
    weights: UtilityWeights = UtilityWeights(),
) -> float:
    """utility = nDCG - latency penalty - constraint-violation penalty - timeout penalty."""
    score = ndcg
    score -= weights.latency_penalty * latency_ms
    score -= weights.violation_penalty * violations
    if timed_out:
        score -= weights.timeout_penalty
    return score
