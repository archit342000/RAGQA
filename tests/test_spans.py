from gold.spans import resolve_evidence_spans, sentence_spans


def test_resolve_evidence_spans_handles_whitespace_and_case() -> None:
    window_text = "The Policy covers data   retention requirements.\nIt applies company-wide."
    sentences = sentence_spans(window_text)
    evidence = ["  the policy covers DATA retention  requirements.  "]

    spans = resolve_evidence_spans(evidence, sentences, window_text)

    assert spans, "Expected at least one span"
    start, end = spans[0]
    expected_snippet = "The Policy covers data   retention requirements."
    expected_start = window_text.index(expected_snippet)
    expected_end = expected_start + len(expected_snippet)
    assert (start, end) == (expected_start, expected_end)


def test_resolve_evidence_spans_matches_across_newlines() -> None:
    window_text = "Responsibilities:\n- Maintain systems\n- Report outages"
    sentences = sentence_spans(window_text)
    evidence = ["Responsibilities: - Maintain systems"]

    spans = resolve_evidence_spans(evidence, sentences, window_text)

    assert spans, "Expected evidence span for newline-normalised text"
    expected_snippet = "Responsibilities:\n- Maintain systems"
    expected_start = window_text.index(expected_snippet)
    expected_end = expected_start + len(expected_snippet)
    assert spans[0] == (expected_start, expected_end)


def test_resolve_evidence_spans_supports_index_fallback() -> None:
    window_text = "First sentence. Second sentence with Evidence. Third one."
    sentences = sentence_spans(window_text)
    assert len(sentences) >= 3
    evidence = [{"index": 1}]

    spans = resolve_evidence_spans(evidence, sentences, window_text)

    assert spans == [sentences[1]]
