from pathlib import Path

from pdf_ingest.pdf_io import Line
from pdf_ingest.table_detect import detect_tables_for_page, write_table_csv


def make_line(page: int, idx: int, text: str) -> Line:
    return Line(page_index=page, line_index=idx, text=text, bbox=(0.0, float(idx), 120.0, float(idx) + 12.0), x_center=60.0)


def test_confident_table_emits_csv(tmp_path: Path) -> None:
    lines = [
        make_line(0, 0, "A,B,C"),
        make_line(0, 1, "1,2,3"),
        make_line(0, 2, "4,5,6"),
    ]
    results = detect_tables_for_page(lines, page_index=0)
    assert results, "expected a table candidate"
    table = results[0]
    path = write_table_csv(table, tmp_path)
    csv_path = Path(path)
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0]
    assert header == "source_page,row_idx,col_idx,cell_bbox,text"


def test_low_confidence_is_skipped() -> None:
    lines = [make_line(0, 0, "free text line"), make_line(0, 1, "another line")]
    results = detect_tables_for_page(lines, page_index=0)
    assert all(result.confidence < 0.1 for result in results)
