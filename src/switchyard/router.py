from __future__ import annotations

from switchyard.features import QueryFeatures
from switchyard.routes import Route


def rule_router(features: QueryFeatures) -> Route:
    """Hand-written routing policy. It is the interpretable baseline that the
    learned LightGBM router (Phase 2) must beat. Each branch maps a query
    signal to the cheapest route expected to satisfy it.

    - exact model numbers are a lexical-retrieval strength, so route LEXICAL
    - price, negation, or compatibility constraints need field-aware reranking
    - questions and long descriptive needs benefit from semantic plus lexical
    - everything else takes hybrid as the safe default
    """
    if features.has_negation or features.has_money or features.has_compatibility:
        return Route.RERANK
    if features.has_model_number and features.token_count <= 4:
        return Route.LEXICAL
    if features.has_question or features.token_count >= 7:
        return Route.RERANK
    return Route.HYBRID


def oracle_route(route_utilities: dict[Route, float]) -> Route:
    """The route that maximizes utility for this query, given per-route
    utilities computed after the fact. This is an upper bound, not a deployable
    policy: it sees the labels. The learned router is judged by how much of the
    oracle's gain over the always-rerank baseline it recovers without labels."""
    return max(route_utilities, key=lambda r: (route_utilities[r], -_cost_rank(r)))


def _cost_rank(route: Route) -> int:
    order = {Route.LEXICAL: 0, Route.DENSE: 1, Route.HYBRID: 2, Route.RERANK: 3}
    return order[route]
