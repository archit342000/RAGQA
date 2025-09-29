from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

try:  # pragma: no cover - optional dependency guards
    import fitz  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - runtime availability handled elsewhere
    fitz = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be absent
    pdfium = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover - optional dependency may be absent
    extract_text = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractorPageVote:
    page_number: int
    text_fitz: str
    text_pdfium: str
    text_pdfminer: str
    len_text_fitz: int
    len_text_pdfium: int
    len_text_pdfminer: int
    has_type3: bool
    has_cid: bool
    has_tounicode: bool
    force_ocr: bool
    digital_text: bool

    def best_extractor_text(self) -> str:
        if self.text_pdfminer.strip():
            return self.text_pdfminer
        if self.text_fitz.strip():
            return self.text_fitz
        return self.text_pdfium

    def extractor_texts(self) -> Dict[str, str]:
        return {
            "pdfminer": self.text_pdfminer,
            "fitz": self.text_fitz,
            "pdfium": self.text_pdfium,
        }


def _fonts_diagnostics(page) -> tuple[bool, bool, bool]:  # pragma: no cover - external
    if page is None:
        return False, False, True
    try:
        fonts = page.get_fonts(full=True)
    except Exception:  # pragma: no cover - some PDFs lack font tables
        return False, False, False
    has_type3 = False
    has_cid = False
    all_tounicode = True
    for font in fonts or []:
        font_type = font[2] if len(font) > 2 else ""
        to_unicode = font[7] if len(font) > 7 else ""
        if isinstance(font_type, bytes):
            font_type = font_type.decode("latin-1", errors="ignore")
        if isinstance(to_unicode, bytes):
            to_unicode = to_unicode.decode("latin-1", errors="ignore")
        if "Type3" in str(font_type):
            has_type3 = True
        if "CID" in str(font_type):
            has_cid = True
        if not to_unicode:
            all_tounicode = False
    return has_type3, has_cid, all_tounicode


def _vote(len_fitz: int, len_pdfium: int, len_pdfminer: int, threshold: int) -> bool:
    return max(len_fitz, len_pdfium, len_pdfminer) >= threshold


def run_multi_extractor(pdf_bytes: bytes, threshold: int) -> List[ExtractorPageVote]:
    if fitz is None:
        raise RuntimeError("pymupdf is required for multi-extractor triage")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    votes: List[ExtractorPageVote] = []
    pdfium_doc = None
    if pdfium is not None:
        try:  # pragma: no cover - heavy dependency
            pdfium_doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
        except Exception as exc:  # pragma: no cover - fallback when pdfium fails
            logger.warning("pypdfium2 failed to open document: %s", exc)
            pdfium_doc = None

    for index, page in enumerate(doc, start=1):
        text_fitz = ""
        text_pdfium = ""
        text_pdfminer = ""
        try:
            text_fitz = page.get_text("text") or ""
        except Exception as exc:  # pragma: no cover - extraction errors rare
            logger.debug("PyMuPDF extraction error on page %s: %s", index, exc)

        if pdfium_doc is not None:
            try:  # pragma: no cover - heavy dependency
                pdfium_page = pdfium_doc.get_page(index - 1)
                with pdfium_page as p:
                    text_page = p.get_textpage("text")
                    text_pdfium = text_page.get_text_range() or ""
            except Exception as exc:  # pragma: no cover
                logger.debug("pypdfium2 extraction error on page %s: %s", index, exc)

        if extract_text is not None:
            try:
                text_pdfminer = extract_text(BytesIO(pdf_bytes), page_numbers=[index - 1]) or ""
            except Exception as exc:  # pragma: no cover
                logger.debug("pdfminer extraction error on page %s: %s", index, exc)

        len_fitz = len(text_fitz)
        len_pdfium = len(text_pdfium)
        len_pdfminer = len(text_pdfminer)
        has_type3, has_cid, has_tounicode = _fonts_diagnostics(page)
        digital_text = _vote(len_fitz, len_pdfium, len_pdfminer, threshold)
        force_ocr = (has_type3 or has_cid or not has_tounicode) and not digital_text
        votes.append(
            ExtractorPageVote(
                page_number=index,
                text_fitz=text_fitz,
                text_pdfium=text_pdfium,
                text_pdfminer=text_pdfminer,
                len_text_fitz=len_fitz,
                len_text_pdfium=len_pdfium,
                len_text_pdfminer=len_pdfminer,
                has_type3=has_type3,
                has_cid=has_cid,
                has_tounicode=has_tounicode,
                force_ocr=force_ocr,
                digital_text=digital_text,
            )
        )

    doc.close()
    if pdfium_doc is not None:
        try:
            pdfium_doc.close()  # type: ignore[attr-defined]  # pragma: no cover
        except Exception:
            pass
    return votes


__all__ = ["ExtractorPageVote", "run_multi_extractor", "_vote"]
