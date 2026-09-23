import numpy as np

from nanoinfer.sampling import sample

logits = np.array([1.0, 5.0, 3.0, 4.5, -2.0, 0.0], dtype=np.float32)
ARGMAX = 1


def draws(n=2000, **kw):
    rng = np.random.default_rng(0)
    return np.array([sample(logits, rng=rng, **kw) for _ in range(n)])


def test_temperature_zero_is_greedy():
    assert sample(logits, temperature=0.0) == ARGMAX


def test_top_k_one_is_greedy():
    assert set(draws(200, top_k=1)) == {ARGMAX}


def test_top_k_restricts_support():
    assert set(draws(top_k=3)) <= {1, 3, 2}


def test_top_p_restricts_support():
    # softmax mass: token1 ~0.57, token3 ~0.34 -> nucleus at p=0.8 is exactly {1, 3}
    assert set(draws(top_p=0.8)) == {1, 3}


def test_matches_softmax_distribution():
    counts = np.bincount(draws(20000, temperature=0.7), minlength=len(logits)) / 20000
    z = logits / 0.7
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    np.testing.assert_allclose(counts, p, atol=0.015)
