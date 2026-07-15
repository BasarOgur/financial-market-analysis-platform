from model.llm_path import LLMClassifier, parse_reply
from tests.conftest import FakeLLM


def test_parses_clean_json():
    p = parse_reply('{"sentiment": "bearish", "topics": ["m&a", "regulation"]}')
    assert p.sentiment == "bearish"
    assert p.topics == ["m&a", "regulation"]


def test_parses_fenced_json():
    p = parse_reply('```json\n{"sentiment": "bullish", "topics": ["earnings"]}\n```')
    assert p.sentiment == "bullish"


def test_coerces_garbage_to_defaults():
    p = parse_reply("I think this is positive news!")
    assert p.sentiment == "neutral"
    assert p.topics == ["other"]


def test_coerces_off_vocabulary_labels():
    p = parse_reply('{"sentiment": "positive", "topics": ["mergers", "earnings"]}')
    assert p.sentiment == "neutral"  # invalid sentiment
    assert p.topics == ["earnings"]  # unknown topic dropped


def test_classify_sends_system_prompt_and_snippet(fake_llm):
    clf = LLMClassifier(fake_llm)
    preds = clf.classify(["Acme beat estimates.", "Regulator fined Acme."])
    assert len(preds) == 2
    assert len(fake_llm.calls) == 2  # one call per snippet
    assert "Acme beat estimates." in fake_llm.calls[0]["prompt"]
    assert "investment advice" in fake_llm.calls[0]["system"].lower()


def test_empty_topics_defaults_to_other():
    llm = FakeLLM(reply='{"sentiment": "neutral", "topics": []}')
    preds = LLMClassifier(llm).classify(["whatever"])
    assert preds[0].topics == ["other"]
