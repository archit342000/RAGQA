from __future__ import annotations

from pdfchunks.chunking.chunker import Chunker, ChunkerConfig
from pdfchunks.threading.threader import ThreadUnit


def unit(text: str, para: int, sent: int, emit_phase: int = 0, section: int = 1, subtype: str | None = None) -> ThreadUnit:
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


def test_chunker_packs_main_with_overlap_and_isolates_aux():
    config = ChunkerConfig(
        main_target_tokens=10,
        main_min_tokens=5,
        main_max_tokens=12,
        main_overlap_tokens=4,
        main_small_overlap_tokens=2,
        aux_max_tokens=50,
    )
    chunker = Chunker(config)
    sentences = [
        unit("one two three four five", para=1, sent=0),
        unit("six seven eight nine ten", para=1, sent=1),
        unit("eleven twelve thirteen fourteen fifteen", para=2, sent=0),
        unit("sixteen seventeen eighteen nineteen twenty", para=2, sent=1),
    ]
    aux_units = [
        unit("<aux>Caption 1</aux>", para=2, sent=2, emit_phase=1, subtype="caption"),
        unit("<aux>Caption 2</aux>", para=2, sent=3, emit_phase=1, subtype="caption"),
        unit("<aux>Activity</aux>", para=3, sent=1, emit_phase=1, section=2, subtype="activity"),
    ]
    chunks = chunker.chunk(sentences + aux_units)
    main_chunks = [chunk for chunk in chunks if chunk.role == "MAIN"]
    aux_chunks = [chunk for chunk in chunks if chunk.role == "AUX"]

    assert len(main_chunks) >= 1
    assert all("<aux>" not in chunk.text for chunk in main_chunks)
    # AUX chunks grouped by section/subtype
    assert len(aux_chunks) == 2
    caption_chunk = next(chunk for chunk in aux_chunks if chunk.section_seq == 1)
    assert caption_chunk.text.count("<aux>") == 2
    activity_chunk = next(chunk for chunk in aux_chunks if chunk.section_seq == 2)
    assert activity_chunk.subtype == "activity"

