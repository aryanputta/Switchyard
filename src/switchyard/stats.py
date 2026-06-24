from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int
    n: int

    @property
    def significant(self) -> bool:
        # significant at the chosen level if the CI does not straddle zero
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    iterations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 13,
) -> BootstrapResult:
    """Paired bootstrap confidence interval on the mean of (a - b), with the
    query as the statistical unit. This is the standard significance test for
    IR metric comparisons: it makes no normality assumption and respects the
    pairing of per-query scores under two systems.

    a and b are aligned per-query metric values (for example nDCG@10 for the
    learned router and for the always-rerank baseline on the same queries).
    """
    if len(a) != len(b):
        raise ValueError("a and b must be aligned per-query and equal length")
    if not a:
        raise ValueError("need at least one query")

    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)

    boot_means: list[float] = []
    for _ in range(iterations):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += diffs[rng.randrange(n)]
        boot_means.append(sample_sum / n)
    boot_means.sort()

    lo_idx = int((alpha / 2) * iterations)
    hi_idx = min(int((1 - alpha / 2) * iterations), iterations - 1)

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return BootstrapResult(
        mean_diff=sum(diffs) / n,
        ci_low=boot_means[lo_idx],
        ci_high=boot_means[hi_idx],
        wins=wins,
        ties=n - wins - losses,
        losses=losses,
        n=n,
    )
