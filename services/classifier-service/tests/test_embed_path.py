from model import load_news
from model.embed_path import EmbeddingClassifier

TRAIN = [
    {"text": "profit beat estimates revenue grew guidance raised", "sentiment": "bullish", "topics": ["earnings"]},
    {"text": "quarterly profit beat forecasts strong revenue growth", "sentiment": "bullish", "topics": ["earnings"]},
    {"text": "profit missed estimates revenue fell guidance cut", "sentiment": "bearish", "topics": ["earnings"]},
    {"text": "quarterly loss missed forecasts weak revenue decline", "sentiment": "bearish", "topics": ["earnings"]},
    {"text": "regulator fined the bank over compliance failures", "sentiment": "bearish", "topics": ["regulation"]},
    {"text": "regulators sued to block the deal citing competition", "sentiment": "bearish", "topics": ["m&a", "regulation"]},
    {"text": "company agreed to acquire rival in cash deal merger", "sentiment": "neutral", "topics": ["m&a"]},
    {"text": "company will report results next week on schedule", "sentiment": "neutral", "topics": ["earnings"]},
    {"text": "annual meeting scheduled next month routine agenda", "sentiment": "neutral", "topics": ["other"]},
]


def test_learns_separable_vocabulary(fake_embedder):
    clf = EmbeddingClassifier(fake_embedder)
    clf.fit(TRAIN)
    preds = clf.classify(
        [
            "profit beat estimates guidance raised",
            "profit missed estimates guidance cut",
            "regulator fined the company over failures",
        ]
    )
    assert preds[0].sentiment == "bullish"
    assert preds[1].sentiment == "bearish"
    assert "earnings" in preds[0].topics
    assert "regulation" in preds[2].topics


def test_always_at_least_one_topic(fake_embedder):
    clf = EmbeddingClassifier(fake_embedder)
    clf.fit(TRAIN)
    preds = clf.classify(["zzz qqq unrelated gibberish tokens"])
    assert len(preds[0].topics) >= 1


def test_trains_on_fixture_dataset(fake_embedder):
    """The real train split must be fit-able (labels present, no degenerate crash)."""
    clf = EmbeddingClassifier(fake_embedder)
    clf.fit(load_news("train"))
    preds = clf.classify(["Company beat quarterly profit estimates and raised guidance."])
    assert preds[0].sentiment in ("bullish", "bearish", "neutral")
    assert preds[0].topics
