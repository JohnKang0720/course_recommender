import numpy as np

from coursesearch import embed


def test_search_ranks_by_cosine():
    vectors = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    idx, scores = embed.search(np.array([1, 0], np.float32), vectors, k=2)
    assert idx[0] == 0
    assert scores[0] >= scores[1]


def test_cluster_labels_length():
    labels = embed.cluster(np.random.default_rng(0).random((60, 8)), k=5)
    assert len(labels) == 60 and set(labels) <= set(range(5))


def test_compare_returns_three_scores():
    scores = embed.compare_clusterings(np.random.default_rng(0).random((80, 8)), k=4)
    assert set(scores) == {"K-Means", "Hierarchical", "Spectral"}
    assert all(isinstance(v, float) for v in scores.values())
