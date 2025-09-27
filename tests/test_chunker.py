from pdf_ingest.chunker import build_chunks, build_paragraphs, estimate_tokens
from pdf_ingest.config import IngestConfig
from pdf_ingest.pdf_io import Line


def _line(page: int, idx: int, text: str) -> Line:
    return Line(page_index=page, line_index=idx, text=text, bbox=(0.0, float(idx), 100.0, float(idx) + 10.0), x_center=50.0, y_top=float(idx))


def test_token_estimate_bounds():
    text = "a" * 2400  # ~600 estimated tokens
    tokens = estimate_tokens(text)
    assert 550 <= tokens <= 650


def test_chunk_overlap_and_neighbors():
    config = IngestConfig()
    config.chunk_target_min = 120
    config.chunk_target_max = 180
    config.overlap_min = config.overlap_max = 0.12
    lines = [_line(0, i, "word " * 80) for i in range(12)]
    paragraphs = build_paragraphs([lines])
    chunks, noise_ratio = build_chunks("doc", paragraphs, config, provenance_hash="hash")
    assert chunks, "at least one chunk is produced"
    assert noise_ratio == 0
    if len(chunks) > 1:
        assert chunks[0].neighbors["next"] == chunks[1].chunk_id
        assert chunks[1].neighbors["prev"] == chunks[0].chunk_id
        # ensure overlap generated extra context at the start of chunk 2
        assert chunks[1].text.split()[0] in chunks[0].text.split()
