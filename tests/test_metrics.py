from switchyard.metrics import (
    dcg_at_k,
    ndcg_at_k,
    reciprocal_rank_first_exact,
)


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert ndcg_at_k(["a", "b", "c", "d"], relevance, 4) == 1.0


def test_ndcg_reversed_ranking_below_one():
    relevance = {"a": 3, "b": 2, "c": 1, "d": 0}
    score = ndcg_at_k(["d", "c", "b", "a"], relevance, 4)
    assert 0.0 < score < 1.0


def test_ndcg_all_irrelevant_is_zero():
    relevance = {"a": 0, "b": 0}
    assert ndcg_at_k(["a", "b"], relevance, 2) == 0.0


def test_graded_gain_rewards_exact_over_substitute():
    # putting the Exact item first must beat putting a Substitute first
    relevance = {"exact": 3, "sub": 2}
    exact_first = ndcg_at_k(["exact", "sub"], relevance, 2)
    sub_first = ndcg_at_k(["sub", "exact"], relevance, 2)
    assert exact_first > sub_first


def test_dcg_uses_2_pow_gain():
    # single item at rank 1: dcg = (2^gain - 1) / log2(2) = 2^gain - 1
    assert dcg_at_k(["a"], {"a": 3}, 1) == 7.0


def test_reciprocal_rank_first_exact():
    relevance = {"x": 2, "y": 3, "z": 0}
    assert reciprocal_rank_first_exact(["x", "y", "z"], relevance) == 0.5
    assert reciprocal_rank_first_exact(["y", "x"], relevance) == 1.0
    assert reciprocal_rank_first_exact(["z", "x"], relevance) == 0.0
