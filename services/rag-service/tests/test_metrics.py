from eval.metrics import hit_at_k, mrr, relevance_at_ranks


def test_relevance_normalizes_case_and_whitespace():
    flags = relevance_at_ranks(["Revenue was  $4.82 Billion."], ["$4.82 billion"])
    assert flags == [True]


def test_relevance_any_span_matches():
    flags = relevance_at_ranks(["margin was 58.2%", "nothing"], ["99%", "58.2%"])
    assert flags == [True, False]


def test_hit_and_mrr_known_values():
    relevance = [
        [True, False, False],   # rank 1
        [False, False, True],   # rank 3
        [False, False, False],  # miss
    ]
    assert hit_at_k(relevance, 1) == 1 / 3
    assert hit_at_k(relevance, 3) == 2 / 3
    assert abs(mrr(relevance) - (1.0 + 1 / 3) / 3) < 1e-9
