BEGIN VERBATIM SCHEMA
Blocks JSON (array of objects)
- doc_id:str
- block_id:str
- page:int
- order:int
- type:enum[heading,paragraph,list,item,table,figure,caption,code,footnote]
- text:str
- bbox:{x0,y0,x1,y1}
- heading_level:int|null
- heading_path:str[]
- source:{stage:enum[triage,docling,ocr,layout], tool:str, version:str}
- aux:{...}
END VERBATIM
