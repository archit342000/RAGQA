from pdf_ingest.chunker import ChunkBuilder, estimate_tokens, paragraphs_from_lines
from pdf_ingest.config import Config
from pdf_ingest.pdf_io import Line


def make_line(page: int, idx: int, text: str) -> Line:
    return Line(page_index=page, line_index=idx, text=text, bbox=(0.0, float(idx), 100.0, float(idx) + 10.0), x_center=50.0)


def test_token_estimate_bounds() -> None:
    text = "a" * 2400  # ~600 estimated tokens
    tokens = estimate_tokens(text)
    assert 550 <= tokens <= 650


def test_chunk_builder_emits_overlapping_chunks() -> None:
    config = Config()
    config.chunk_tokens_min = 120
    config.chunk_tokens_max = 160
    lines = [make_line(0, i, "word " * 40) for i in range(12)]
    paragraphs, dropped = paragraphs_from_lines(lines, config=config)
    assert dropped == 0
    builder = ChunkBuilder(config)
    emitted = []
    for paragraph in paragraphs:
        emitted.extend(builder.add_paragraph(paragraph))
    emitted.extend(builder.finalize())
    assert emitted, "at least one chunk emitted"
    if len(emitted) > 1:
        assert emitted[0].neighbors["next"] == emitted[1].chunk_id
        assert emitted[1].neighbors["prev"] == emitted[0].chunk_id


def test_paragraph_noise_filter() -> None:
    config = Config()
    noisy_line = make_line(0, 0, "@@@###%%%$$$")
    separator = make_line(0, 1, "")
    clean_line = make_line(0, 2, "Heading:")
    paragraphs, dropped = paragraphs_from_lines([noisy_line, separator, clean_line], config=config)
    assert dropped == 1
    assert any(p.heading for p in paragraphs)
