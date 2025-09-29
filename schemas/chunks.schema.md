BEGIN VERBATIM SCHEMA
Chunks JSONL (one per chunk)
- chunk_id:str
- doc_id:str
- page_span:[int,int]
- heading_path:str[]
- text:str
- token_count:int
- sidecars:[{type, page, text}]
- evidence_spans:[{para_block_id:str, start:int, end:int}]
- quality:{ocr_pages:int, rescued:bool, notes:str}
END VERBATIM
