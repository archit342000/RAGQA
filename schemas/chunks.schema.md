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
// Additions (backward compatible):
// BEGIN ADDITIONS
// aux_groups: {
//   sidecars: [{parent_block_id, type:"figure"|"table", page, text}],
//   footnotes: [{ref_id, text}],
//   other: [{aux_subtype, text}]
// }
// notes: str|null
// END ADDITIONS
END VERBATIM
