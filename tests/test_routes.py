from switchyard.esci import Candidate, EsciQuery
from switchyard.fusion import reciprocal_rank_fusion
from switchyard.index import Scored
from switchyard.routes import ALL_ROUTES, Route, run_route


def _query() -> EsciQuery:
    return EsciQuery(
        query_id="q",
        query="sony wh-1000xm5 headphones",
        locale="us",
        candidates=[
            Candidate("p1", "Sony WH-1000XM5 Headphones", brand="Sony", label="E"),
            Candidate("p2", "Bose QuietComfort Headphones", brand="Bose", label="I"),
            Candidate("p3", "Sony WH-1000XM4 Headphones", brand="Sony", label="S"),
        ],
    )


def test_every_route_returns_all_candidates():
    q = _query()
    for route in ALL_ROUTES:
        result = run_route(route, q)
        assert set(result.ranked_ids) == {"p1", "p2", "p3"}


def test_rerank_ranks_exact_match_first():
    result = run_route(Route.RERANK, _query())
    assert result.ranked_ids[0] == "p1"


def test_lexical_finds_exact_model_match():
    result = run_route(Route.LEXICAL, _query())
    assert result.ranked_ids[0] == "p1"


def test_latency_ordering_lexical_cheapest():
    q = _query()
    lat = {r: run_route(r, q).latency_ms for r in ALL_ROUTES}
    assert lat[Route.LEXICAL] < lat[Route.DENSE] < lat[Route.HYBRID] < lat[Route.RERANK]


def test_rrf_promotes_consensus():
    a = [Scored("x", 9), Scored("y", 1)]
    b = [Scored("x", 5), Scored("y", 4)]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].doc_id == "x"
