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
// limits: {target:int, soft:int, hard:int, min:int}
// flow_overflow: int
// closed_at_boundary: "H1"|"H2"|"H3"|"Sub"|"Para"|"Sent"|"EOF"
// aux_in_followup: bool
// link_prev: str|null
// link_next: str|null
// section_id: str
// thread_id: str|null
// is_main_only: bool
// is_aux_only: bool
// aux_subtypes_present: [str]
// segment_id: str
// segment_seq: int
// is_main_only: bool
// is_aux_only: bool
// aux_subtypes_present: [str]
// aux_group_seq: int|null
// END ADDITIONS
END VERBATIM
