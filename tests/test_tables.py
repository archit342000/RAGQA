from pathlib import Path

from pdf_ingest.pdf_io import Line
from pdf_ingest.table_detect import detect_tables


def _line(page: int, idx: int, text: str) -> Line:
    return Line(page_index=page, line_index=idx, text=text, bbox=(0.0, float(idx), 120.0, float(idx) + 12.0), x_center=60.0, y_top=float(idx))


def test_confident_table_emits_csv(tmp_path):
    rows = [
        _line(0, 0, "A,B,C"),
        _line(0, 1, "1,2,3"),
        _line(0, 2, "4,5,6"),
    ]
    artifacts, skipped = detect_tables([rows], tmp_path, confidence_threshold=0.3)
    assert skipped == 0
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.csv_path is not None and Path(artifact.csv_path).exists()
    contents = Path(artifact.csv_path).read_text().splitlines()
    assert contents[0] == "source_page,row_idx,col_idx,cell_bbox,text"


def test_low_confidence_skipped(tmp_path):
    rows = [[_line(0, 0, "free text line")]]
    artifacts, skipped = detect_tables(rows, tmp_path, confidence_threshold=0.5)
    assert not artifacts
    assert skipped == 0
