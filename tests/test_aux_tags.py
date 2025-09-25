from __future__ import annotations

from pdfchunks.audit.guards import run_audits
from pdfchunks.chunking.chunker import Chunker
from pdfchunks.config import ParserConfig
from pdfchunks.threading.threader import ThreadResult, ThreadUnit


def make_unit(text: str, para: int, sent: int, emit_phase: int, section: int = 1, subtype: str | None = None) -> ThreadUnit:
    return ThreadUnit(
        doc_id="doc",
        section_seq=section,
        para_seq=para,
        sent_seq=sent,
        emit_phase=emit_phase,
        text=text,
        role="MAIN" if emit_phase == 0 else "AUX",
        subtype=subtype,
    )


def test_all_aux_units_and_chunks_wrapped():
    config = ParserConfig()
    units = [
        make_unit("Lead paragraph ends here.", para=1, sent=0, emit_phase=0),
        make_unit("<aux>Caption text</aux>", para=1, sent=1, emit_phase=1, subtype="caption"),
    ]
    result = ThreadResult(units=units, delayed_aux_counts={1: 0})
    chunker = Chunker(config.chunker)
    chunks = chunker.chunk(units)
    run_audits(result, chunks, config.audits)
    for unit in units:
        if unit.emit_phase == 1:
            assert unit.text.startswith("<aux>") and unit.text.endswith("</aux>")
    for chunk in chunks:
        if chunk.role == "AUX":
            assert chunk.text.startswith("<aux>") and chunk.text.strip().endswith("</aux>")


def test_preamble_aux_passes_audit_without_main_sentences():
    config = ParserConfig()
    units = [
        make_unit("<aux>Front matter</aux>", para=0, sent=0, emit_phase=1, section=0, subtype="callout"),
    ]
    result = ThreadResult(units=units, delayed_aux_counts={0: 1})
    chunker = Chunker(config.chunker)
    chunks = chunker.chunk(units)
    run_audits(result, chunks, config.audits)

