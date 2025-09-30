BEGIN VERBATIM SCHEMA
Segment (internal)
- segment_id: str
- heading_path: [str]
- main_blocks: [BlockRef]
- aux_blocks: [BlockRef]
- pages: [int]
- anchors: [{aux_block_id:str, parent_block_id:str|null, status:"prev|prev_page|next_page|orphan"}]
END VERBATIM
