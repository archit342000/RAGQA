Per-doc summary telemetry
- pages:int
- doc_time_ms:int
- docling_pages:int
- ocr_pages_cpu:int
- ocr_pages_gpu:int
- layout_pages:int
- emitted_chunks:int
- text_layer_pages:int
- relaxed_filter_pages:int
- fallbacks_used:{docling_fail:int, ocr:int, degraded:int}
- first_error_code:str|null

Per-page CSV
- doc_id
- page
- stage_used:str
- latency_ms:int
- text_len:int
- fallback_applied:bool
- error_codes:[str]
- len_text_fitz:int
- len_text_pdfium:int
- len_text_pdfminer:int
- has_type3:bool
- has_cid:bool
- has_tounicode:bool
- path_used:str
- filter_relaxed:bool
