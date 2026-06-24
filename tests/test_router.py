from switchyard.features import extract_features
from switchyard.router import oracle_route, rule_router
from switchyard.routes import Route
from switchyard.utility import UtilityWeights, utility


def test_rule_router_model_number_goes_lexical():
    assert rule_router(extract_features("sony wh-1000xm5")) == Route.LEXICAL


def test_rule_router_negation_goes_rerank():
    assert rule_router(extract_features("headphones without microphone")) == Route.RERANK


def test_rule_router_price_constraint_goes_rerank():
    assert rule_router(extract_features("black headphones under $100")) == Route.RERANK


def test_rule_router_long_descriptive_goes_rerank():
    feats = extract_features("comfortable shoes for nurses working long shifts")
    assert rule_router(feats) == Route.RERANK


def test_rule_router_default_is_hybrid():
    assert rule_router(extract_features("green tea")) == Route.HYBRID


def test_oracle_picks_highest_utility():
    utilities = {
        Route.LEXICAL: 0.4,
        Route.DENSE: 0.5,
        Route.HYBRID: 0.7,
        Route.RERANK: 0.7,
    }
    # ties broken toward the cheaper route
    assert oracle_route(utilities) == Route.HYBRID


def test_zero_latency_penalty_prefers_quality():
    weights = UtilityWeights(latency_penalty=0.0)
    cheap = utility(0.6, 2.0, weights=weights)
    expensive = utility(0.8, 35.0, weights=weights)
    assert expensive > cheap


def test_high_latency_penalty_can_flip_choice():
    weights = UtilityWeights(latency_penalty=0.02)
    cheap = utility(0.6, 2.0, weights=weights)
    expensive = utility(0.62, 35.0, weights=weights)
    assert cheap > expensive
