from pipeline.flow_fence import flow_fence_tail_sanitize


def test_flow_fence_diverts_caption_like_block() -> None:
    blocks = [
        {"type": "paragraph", "text": "Narrative text", "aux": {}},
        {"type": "paragraph", "text": "Figure 2. Diagram", "aux": {}},
    ]
    kept, diverted, hits = flow_fence_tail_sanitize(blocks)
    assert len(kept) == 1
    assert hits == 1
    assert diverted and "caption_prefix" in diverted[0]["rejection_reasons"]
