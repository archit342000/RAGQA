from __future__ import annotations

import pytest

from pdfchunks.serialize.serializer import Serializer
from pdfchunks.threading.threader import ThreadUnit


def make_unit(section: int, para: int, sent: int, emit_phase: int) -> ThreadUnit:
    return ThreadUnit(
        doc_id="doc",
        section_seq=section,
        para_seq=para,
        sent_seq=sent,
        emit_phase=emit_phase,
        text=f"unit {section}-{para}-{sent}-{emit_phase}",
        role="MAIN" if emit_phase == 0 else "AUX",
    )


def test_serializer_accepts_monotonic_sequence():
    serializer = Serializer()
    units = [
        make_unit(1, 0, 0, 0),
        make_unit(1, 1, 0, 0),
        make_unit(1, 1, 1, 0),
        make_unit(1, 1, 2, 1),
    ]
    ordered = serializer.serialize(units)
    assert ordered == units


def test_serializer_rejects_regressions():
    serializer = Serializer()
    good = make_unit(1, 1, 0, 0)
    bad = make_unit(1, 0, 0, 0)
    with pytest.raises(ValueError):
        serializer.serialize([good, bad])

