"""Run: cd ci && pytest -q   (needs only jiwer; no model, no GPU, no network)"""
from evaluate_and_promote import decide, eval_set_id, normalize_text, wer_pct


def test_tamil_vowel_signs_survive_normalisation():
    s = "வணக்கம், உலகம்!"
    n = normalize_text(s)
    assert "," not in n and "!" not in n
    assert n == "வணக்கம் உலகம்"          # combining vowel signs / pulli intact


def test_wer_basic_and_punctuation_insensitive():
    assert wer_pct(["a b c"], ["a b c"]) == 0.0
    assert abs(wer_pct(["a b c d"], ["a b c"]) - 25.0) < 1e-9
    assert wer_pct(["வணக்கம் உலகம்"], ["வணக்கம், உலகம்."]) == 0.0


def test_decide_tolerance_and_regression():
    ok, _ = decide(10.3, 10.0, tolerance=0.5)
    assert ok
    ok, _ = decide(11.0, 10.0, tolerance=0.5)
    assert not ok
    ok, _ = decide(10.0, 10.0, tolerance=0.5, require_improvement=True)
    assert not ok
    ok, _ = decide(9.9, 10.0, tolerance=0.5, require_improvement=True)
    assert ok


def test_eval_set_id_changes_with_inputs():
    assert eval_set_id(b"x", 100) == eval_set_id(b"x", 100)
    assert eval_set_id(b"x", 100) != eval_set_id(b"y", 100)
    assert eval_set_id(b"x", 100) != eval_set_id(b"x", 50)
