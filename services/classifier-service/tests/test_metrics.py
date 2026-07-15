from eval.metrics import macro_f1, per_label_scores


def test_per_label_scores_hand_computed():
    y_true = [{"a"}, {"a", "b"}, {"b"}, {"c"}]
    y_pred = [{"a"}, {"a"}, {"a", "b"}, {"b"}]
    scores = per_label_scores(y_true, y_pred, ("a", "b", "c"))
    # a: tp=2 fp=1 fn=0 -> p=2/3 r=1
    assert scores["a"].precision == 2 / 3
    assert scores["a"].recall == 1.0
    assert scores["a"].support == 2
    # b: tp=1 fp=1 fn=1 -> p=r=f1=0.5
    assert scores["b"].f1 == 0.5
    # c: never predicted -> all zero, support 1
    assert scores["c"].f1 == 0.0
    assert scores["c"].support == 1


def test_macro_f1_ignores_unsupported_labels():
    scores = per_label_scores([{"a"}], [{"a"}], ("a", "unused"))
    assert scores["unused"].support == 0
    assert macro_f1(scores) == 1.0  # only 'a' counts


def test_empty_prediction_sets_score_zero():
    scores = per_label_scores([{"a"}], [set()], ("a",))
    assert scores["a"].recall == 0.0
    assert scores["a"].precision == 0.0
