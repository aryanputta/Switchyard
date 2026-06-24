from __future__ import annotations

from dataclasses import dataclass

"""Web data acquisition under a crawl budget.

This is the web-search counterpart to product-search routing. The scarce
resource is the crawl budget, not query latency. A value-of-crawl model scores
each candidate URL by expected relevance contribution per unit fetch cost, and
a budgeted selector picks the set that maximizes acquired relevance.

In Phase 2 the value model is trained on features with quality labels produced
by an LLM-as-judge over a sample of fetched pages. Here the model is an
interpretable linear scorer so the selection algorithm can be tested
deterministically and the LLM labeling step can be added without changing the
selector.
"""


@dataclass(frozen=True)
class CrawlCandidate:
    url: str
    fetch_cost: float          # modeled cost to fetch and process, e.g. seconds
    pred_quality: float        # value-of-crawl model output in [0, 1]
    pred_freshness: float = 0.0  # predicted change rate, higher means refresh sooner
    domain_authority: float = 0.0


@dataclass(frozen=True)
class CrawlPlan:
    selected: list[str]
    spent: float
    acquired_value: float


def value_of_crawl(
    candidate: CrawlCandidate,
    *,
    w_quality: float = 1.0,
    w_freshness: float = 0.3,
    w_authority: float = 0.2,
) -> float:
    """Expected relevance contribution of fetching this URL. A real deployment
    trains these weights; the structure stays the same."""
    return (
        w_quality * candidate.pred_quality
        + w_freshness * candidate.pred_freshness
        + w_authority * candidate.domain_authority
    )


def plan_crawl(candidates: list[CrawlCandidate], budget: float) -> CrawlPlan:
    """Greedy budgeted selection by value density (value per fetch cost). This
    is the classic greedy 0/1-knapsack heuristic: it is not optimal in general,
    but it is the standard, fast policy for crawl scheduling under a budget and
    is near-optimal when individual costs are small relative to the budget."""
    ranked = sorted(
        candidates,
        key=lambda c: (value_of_crawl(c) / c.fetch_cost if c.fetch_cost > 0 else 0.0),
        reverse=True,
    )
    selected: list[str] = []
    spent = 0.0
    acquired = 0.0
    for c in ranked:
        if c.fetch_cost <= 0:
            continue
        if spent + c.fetch_cost <= budget:
            selected.append(c.url)
            spent += c.fetch_cost
            acquired += value_of_crawl(c)
    return CrawlPlan(selected=selected, spent=spent, acquired_value=acquired)
