from tests.test_classifier import _make_block

from parser.classifier import classify_blocks
from parser.utils import DocumentLayout, PageLayout


def _doc_with_pages(page_blocks):
    pages = []
    for idx, blocks in enumerate(page_blocks):
        pages.append(PageLayout(page_number=idx, width=600, height=800, blocks=blocks))
    return DocumentLayout(pages=pages)


def test_false_header_guard_continuation_in_header_band_is_main():
    opener = _make_block("This paragraph continues to the next page-", (80, 200, 520, 320), font_size=12, col_id=1)
    opener.attrs["indent"] = 24

    continuation = _make_block(
        "continues with lowercase text even though it sits in the header band.",
        (80, 40, 520, 120),
        font_size=12,
        col_id=1,
    )
    continuation.attrs["indent"] = 24

    doc = _doc_with_pages([[opener], [continuation]])
    preds = classify_blocks(doc)
    cont_pred = next(p for p in preds if "continues with lowercase" in p.text)
    assert cont_pred.kind == "main"
    assert "FalseHeaderGuard" in cont_pred.reason
