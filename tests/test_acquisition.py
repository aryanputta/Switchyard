from switchyard.acquisition import CrawlCandidate, plan_crawl, value_of_crawl


def test_value_of_crawl_increases_with_quality():
    low = CrawlCandidate("a", fetch_cost=1.0, pred_quality=0.1)
    high = CrawlCandidate("b", fetch_cost=1.0, pred_quality=0.9)
    assert value_of_crawl(high) > value_of_crawl(low)


def test_plan_respects_budget():
    candidates = [
        CrawlCandidate(f"u{i}", fetch_cost=2.0, pred_quality=0.5) for i in range(10)
    ]
    plan = plan_crawl(candidates, budget=5.0)
    assert plan.spent <= 5.0
    assert len(plan.selected) == 2


def test_plan_prefers_high_value_density():
    cheap_good = CrawlCandidate("cheap", fetch_cost=1.0, pred_quality=0.8)
    expensive_good = CrawlCandidate("pricey", fetch_cost=10.0, pred_quality=0.9)
    plan = plan_crawl([expensive_good, cheap_good], budget=1.0)
    assert plan.selected == ["cheap"]


def test_zero_cost_candidates_are_skipped():
    candidates = [CrawlCandidate("z", fetch_cost=0.0, pred_quality=1.0)]
    plan = plan_crawl(candidates, budget=10.0)
    assert plan.selected == []
