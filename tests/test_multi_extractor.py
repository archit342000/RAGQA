from pipeline.multi_extractor import _vote


def test_multi_extractor_vote_detects_digital_text():
    assert _vote(10, 0, 200, 150) is True
    assert _vote(0, 0, 149, 150) is False
