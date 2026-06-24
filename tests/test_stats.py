from switchyard.stats import paired_bootstrap


def test_clear_improvement_is_significant():
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88]
    b = [0.5, 0.4, 0.45, 0.55, 0.3, 0.48]
    res = paired_bootstrap(a, b, iterations=2000)
    assert res.mean_diff > 0
    assert res.significant
    assert res.wins == 6 and res.losses == 0


def test_no_difference_is_not_significant():
    a = [0.5, 0.6, 0.55, 0.5]
    b = [0.5, 0.6, 0.55, 0.5]
    res = paired_bootstrap(a, b, iterations=2000)
    assert res.mean_diff == 0
    assert not res.significant
    assert res.ties == 4


def test_requires_aligned_lengths():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_ci_brackets_mean():
    a = [0.7, 0.72, 0.69, 0.71, 0.68]
    b = [0.6, 0.61, 0.59, 0.62, 0.58]
    res = paired_bootstrap(a, b, iterations=3000)
    assert res.ci_low <= res.mean_diff <= res.ci_high
