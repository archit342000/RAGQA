from parser.fsm import FlowState, fsm_stitch_and_buffer


def test_fsm_buffers_aux_until_main():
    state = FlowState()
    page_blocks = [
        {"id": "a1", "type": "aux", "subtype": "caption", "text": "Fig. 1"},
        {"id": "m1", "type": "main", "subtype": "paragraph", "text": "This is main."},
    ]
    page = fsm_stitch_and_buffer("p_0001", page_blocks, state, {})
    flushed_ids = [blk["id"] for blk in page["blocks"]]
    assert flushed_ids[0] == "a1"
    assert page["blocks"][1]["flow_id"].startswith("flow_main_")
    assert page["aux_queue"][0]["items"] == ["a1"]
