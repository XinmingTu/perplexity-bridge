import math

from cbridge.docscore import summarize
from cbridge.text import Document


def test_summarize_counts_every_scored_token_once_and_splits_labels():
    doc = Document("d1", "uk_commons", "2026-10-01", "t", [("Mr Speaker", "Order, order."), ("", "Thank you.")],
                   {"kind": "oral_questions"})
    text, labels = doc.render()
    assert text == "Mr Speaker: Order, order.\n\nThank you.\n"
    # tokens start at these characters; the first token is never scored
    offs = [0, 3, 12, 18, 25, 27, 32]
    lps = [0.0] + [-math.log(2)] * 6  # exactly one bit per scored token
    row = summarize(doc, text, labels, offs, lps)
    assert row["n_tokens"] == 7
    assert row["bits"] == 6.0
    assert row["bytes"] == len(text.encode()) - 3  # everything after the unscored first token
    # "Speaker: " (offs 3..12) touches the label span (0, 12): counted in all text, not in speech
    assert row["bits_speech"] == 5.0
    assert row["bytes_speech"] == row["bytes"] - 9


def test_summarize_stops_at_truncation_point():
    doc = Document("d2", "canada_commons", "2026-10-01", "t", [("", "abc def ghi")], {"kind": "debate"})
    text, labels = doc.render()
    row = summarize(doc, text, labels, [0, 4], [0.0, -math.log(2)], stop=7)
    assert (row["bits"], row["bytes"]) == (1.0, 3)  # "def" only: the dropped tail is not counted
