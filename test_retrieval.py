from retrieval import retrieval_quality


def test_retrieval_quality_handles_empty_results():
    quality = retrieval_quality({"distances": [[]], "documents": [[]]}, query="test")
    assert quality["score"] == 0.0
    assert quality["available_contexts"] == 0


def test_retrieval_quality_rewards_close_relevant_match():
    results = {
        "distances": [[0.1, 0.9, 1.0]],
        "documents": [[
            "binary search runs in log n time on a sorted array",
            "unrelated text about cooking recipes",
            "more unrelated text about gardening",
        ]],
        "metadatas": [[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}]],
    }
    quality = retrieval_quality(results, query="binary search time complexity")
    assert 0.0 < quality["score"] <= 1.0
    assert quality["available_contexts"] == 3


def test_retrieval_quality_score_is_bounded():
    results = {
        "distances": [[0.0, 0.0, 0.0]],
        "documents": [["same text"] * 3],
        "metadatas": [[{"source": "a.md"}] * 3],
    }
    quality = retrieval_quality(results, query="same text")
    assert 0.0 <= quality["score"] <= 1.0
