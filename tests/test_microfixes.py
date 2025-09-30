from pipeline.chunker import Chunk
from pipeline.microfixes.cross_page_stitch import cross_page_stitch
from pipeline.microfixes.sentence_gate import sentence_closure_gate

_COUNTER = 0


def make_chunk(text: str, *, main: bool = True, section: str = "sec") -> Chunk:
    global _COUNTER
    _COUNTER += 1
    return Chunk(
        chunk_id=f"c{_COUNTER}", doc_id="d", page_span=[1, 1], heading_path=[], text=text,
        token_count=len(text.split()), sidecars=[], evidence_spans=[],
        quality={}, aux_groups={"sidecars": [], "footnotes": [], "other": []},
        notes=None, limits={"target": 1600, "soft": 2000, "hard": 2400, "min": 900},
        flow_overflow=0, closed_at_boundary="Para", aux_in_followup=False,
        link_prev=None, link_next=None, segment_id=section, segment_seq=0,
        is_main_only=main, is_aux_only=not main, aux_subtypes_present=[],
        aux_group_seq=None, section_id=section, thread_id=None, debug_block_ids=[],
    )


def test_sentence_closure_gate_delays_until_period():
    main = make_chunk("This is incomplete")
    aux = make_chunk("Figure 1.", main=False)
    closed = make_chunk("Now complete.")
    ordered, delayed = sentence_closure_gate([main, aux, closed])
    assert ordered[0] is main
    assert ordered[1] is closed
    assert ordered[2] is aux
    assert delayed >= 0


def test_cross_page_stitch_merges_small_followup():
    first = make_chunk("Intro without stop", section="s1")
    second = make_chunk("Continuation sentence.", section="s1")
    stitched, count = cross_page_stitch([first, second])
    assert count == 1
    assert len(stitched) == 1
    assert "Continuation" in stitched[0].text
