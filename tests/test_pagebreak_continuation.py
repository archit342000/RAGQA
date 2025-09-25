from tests.test_classifier import _make_block

from parser.classifier import classify_blocks
from parser.utils import DocumentLayout, PageLayout


def _doc_with_pages(page_blocks):
    pages = []
    for idx, blocks in enumerate(page_blocks):
        pages.append(PageLayout(page_number=idx, width=600, height=800, blocks=blocks))
    return DocumentLayout(pages=pages)


def test_aux_parked_until_paragraph_closed_across_pagebreak():
    opener = _make_block("This paragraph bridges to the next page-", (80, 200, 520, 320), font_size=12, col_id=1)
    opener.attrs["indent"] = 20

    callout = _make_block("Activity: Quick recap question.", (100, 60, 520, 140), font_size=12, col_id=1)
    callout.attrs["indent"] = 44

    continuation = _make_block(
        "continues the narrative to finish the thought.",
        (80, 160, 520, 280),
        font_size=12,
        col_id=1,
    )
    continuation.attrs["indent"] = 20

    doc = _doc_with_pages([[opener], [callout, continuation]])
    preds = classify_blocks(doc)
    callout_pred = next(p for p in preds if p.text.startswith("Activity"))
    continuation_pred = next(p for p in preds if "continues the narrative" in p.text)
    opener_pred = next(p for p in preds if "bridges" in p.text)

    assert opener_pred.kind == "main"
    assert continuation_pred.kind == "main"
    assert callout_pred.kind == "aux"
    # Callout should have been parked while the paragraph was open
    assert "AuxParked" in callout_pred.reason
