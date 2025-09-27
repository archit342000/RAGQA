from parser import DocumentParser, PageInput
from parser.grouping import Span


def test_document_parser_pipeline_outputs_schema():
    page = PageInput(
        page_id="p_0001",
        number=1,
        width=400,
        height=600,
        spans=[
            Span(text="Figure block", bbox=(10, 200, 90, 260), font_size=11),
            Span(text="Figure 1. Sample caption", bbox=(10, 270, 200, 300), font_size=10),
            Span(text="Paragraph body text that continues.", bbox=(10, 320, 380, 360), font_size=11),
        ],
        meta={
            "detections": [
                {"class": "figure", "bbox": (10, 200, 90, 260), "score": 0.9},
                {"class": "caption", "bbox": (10, 270, 200, 300), "score": 0.8},
            ]
        },
    )
    parser = DocumentParser()
    doc = parser.parse([page])
    assert "pages" in doc
    page_out = doc["pages"][0]
    caption = next(blk for blk in page_out["blocks"] if blk["subtype"] == "caption")
    assert caption["text"].startswith("<aux>")
    assert page_out["aux_queue"] == []
