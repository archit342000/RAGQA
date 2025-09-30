"""Gradio app wiring the Step-1 pipeline to the existing retrieval console."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence

import gradio as gr

from env_loader import load_dotenv_once
from pipeline import PipelineConfig, PipelineService, PipelineResult
from retrieval import RetrievalConfig, build_indexes, retrieve

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

load_dotenv_once()

PIPELINE_CFG = PipelineConfig.from_mapping({})
PIPELINE = PipelineService(PIPELINE_CFG)
RETRIEVAL_CFG = RetrievalConfig.from_env()

RETRIEVAL_LABEL_TO_KEY = {
    "Semantic → Rerank": "semantic",
    "Lexical → Rerank": "lexical",
    "Hybrid → Rerank": "hybrid",
}
RETRIEVAL_KEY_TO_LABEL = {v: k for k, v in RETRIEVAL_LABEL_TO_KEY.items()}
DEFAULT_RETRIEVAL_LABEL = RETRIEVAL_KEY_TO_LABEL.get(RETRIEVAL_CFG.default_engine, "Semantic → Rerank")


def _build_doc_entry(result: PipelineResult) -> dict:
    blocks = result.blocks_json()
    chunks = result.chunks_jsonl()
    retrieval_chunks = [
        {
            "id": chunk["chunk_id"],
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "doc_name": result.doc_name,
            "page_start": chunk["page_span"][0],
            "page_end": chunk["page_span"][1],
            "token_len": chunk["token_count"],
            "meta": {
                "heading_path": chunk["heading_path"],
                "sidecars": chunk["sidecars"],
            },
        }
        for chunk in chunks
    ]
    return {
        "doc_id": result.doc_id,
        "doc_name": result.doc_name,
        "blocks": blocks,
        "chunks": chunks,
        "triage": result.triage_rows,
        "telemetry": result.telemetry.to_dict(),
        "retrieval_chunks": retrieval_chunks,
    }


def _build_indexes_from_state(state: dict) -> dict:
    docs = state.get("docs") or {}
    all_chunks: List[MutableMapping[str, object]] = []
    for entry in docs.values():
        all_chunks.extend(entry.get("retrieval_chunks", []))
    indexes = build_indexes(all_chunks, RETRIEVAL_CFG) if all_chunks else {}
    return {"indexes": indexes, "chunks": all_chunks}


def parse_batch_ui(files: Sequence[str] | None) -> tuple:
    docs: Dict[str, dict] = {}
    doc_choices: List[tuple[str, str]] = []
    chunk_choices: List[tuple[str, str]] = []
    telemetry_summary: List[dict] = []
    total_chunks = 0
    first_chunk_text = ""
    first_doc_id: str | None = None
    first_chunk_id: str | None = None

    file_list = [Path(f) for f in files or [] if f]
    for file_path in file_list:
        try:
            result = PIPELINE.process_pdf(str(file_path))
        except Exception as exc:
            logger.exception("Failed to parse %s: %s", file_path, exc)
            continue
        entry = _build_doc_entry(result)
        docs[entry["doc_id"]] = entry
        label = f"{entry['doc_name']} ({entry['doc_id']})"
        doc_choices.append((label, entry["doc_id"]))
        telemetry_summary.append(entry["telemetry"])
        total_chunks += len(entry["chunks"])
        if first_doc_id is None:
            first_doc_id = entry["doc_id"]
        if entry["chunks"] and first_chunk_id is None:
            first_chunk = entry["chunks"][0]
            first_chunk_id = first_chunk["chunk_id"]
            first_chunk_text = first_chunk["text"]

    if first_doc_id:
        first_entry = docs[first_doc_id]
        for chunk in first_entry["chunks"]:
            heading = " > ".join(chunk["heading_path"]) or "Untitled"
            span = chunk["page_span"]
            label = f"{span[0]}-{span[1]} · {heading[:80]}"
            chunk_choices.append((label, chunk["chunk_id"]))

    state = {"docs": docs, "order": [choice[1] for choice in doc_choices]}
    retrieval_state = _build_indexes_from_state(state)

    status_lines = [
        f"Processed {len(docs)} document(s)",
        f"Total chunks: {total_chunks}",
    ]
    status_md = "\n".join(status_lines)
    debug_payload = {
        "telemetry": telemetry_summary,
        "triage_preview": {doc_id: entry["triage"][:3] for doc_id, entry in docs.items()},
    }

    return (
        state,
        gr.update(choices=doc_choices, value=first_doc_id),
        gr.update(choices=chunk_choices, value=first_chunk_id),
        first_chunk_text,
        gr.update(value=status_md),
        gr.update(value=debug_payload),
        retrieval_state,
        DEFAULT_RETRIEVAL_LABEL,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value={}),
        gr.update(value=""),
    )


def update_chunk_selector(doc_id: str, state: dict) -> tuple:
    docs = state.get("docs") or {}
    entry = docs.get(doc_id)
    if not entry:
        return gr.update(choices=[], value=None), ""
    choices = []
    first_chunk_text = ""
    first_chunk_id = None
    for chunk in entry["chunks"]:
        heading = " > ".join(chunk["heading_path"]) or "Untitled"
        span = chunk["page_span"]
        label = f"{span[0]}-{span[1]} · {heading[:80]}"
        choices.append((label, chunk["chunk_id"]))
        if first_chunk_id is None:
            first_chunk_id = chunk["chunk_id"]
            first_chunk_text = chunk["text"]
    return gr.update(choices=choices, value=first_chunk_id), first_chunk_text


def show_chunk(chunk_id: str, state: dict) -> str:
    docs = state.get("docs") or {}
    for entry in docs.values():
        for chunk in entry["chunks"]:
            if chunk["chunk_id"] == chunk_id:
                return chunk["text"]
    return ""


def run_retrieval(query: str, engine_label: str, state: dict, retrieval_state: dict) -> tuple:
    engine_key = RETRIEVAL_LABEL_TO_KEY.get(engine_label, "semantic")
    indexes = retrieval_state.get("indexes") if retrieval_state else None
    if not indexes:
        return engine_label, "", "", "No indexes available. Parse documents first.", {}, retrieval_state
    try:
        result = retrieve(query, engine_key, indexes, RETRIEVAL_CFG)
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        return engine_label, "", "", f"Retrieval error: {exc}", {}, retrieval_state

    final_chunks = result.get("final_chunks", [])
    if final_chunks:
        preview = final_chunks[0]["text"]
        detail_lines = []
        for chunk in final_chunks:
            doc_id = chunk.get("doc_id", "")
            heading = " > ".join(chunk.get("meta", {}).get("heading_path", []))
            detail_lines.append(f"### {doc_id}\n{heading}\n\n{chunk['text']}")
        full_view = "\n\n".join(detail_lines)
    else:
        preview = ""
        full_view = "No chunks returned"

    provenance = result.get("provenance", {})
    status_lines = [
        f"Engine: {provenance.get('engine', engine_key)}",
        f"Cache hit: {provenance.get('cache_hit', False)}",
        f"Candidates: {len(final_chunks)}",
    ]
    status_md = "\n".join(status_lines)
    debug_payload = {
        "provenance": provenance,
        "timings": result.get("timings", {}),
    }
    return engine_label, preview, full_view, status_md, debug_payload, retrieval_state


def rebuild_indexes_action(engine_label: str, state: dict, retrieval_state: dict) -> tuple:
    new_state = _build_indexes_from_state(state)
    message = "Indexes rebuilt from current documents."
    return engine_label, "", "", message, {}, new_state


def prepare_gold_inputs(state: dict) -> str:
    docs = state.get("docs") or {}
    if not docs:
        return "No parsed documents available."
    return "Gold export integration pending for the new pipeline."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Blocks & Chunking Preview") as demo:
        gr.Markdown("## PDF Blocks → Chunks Pipeline")
        with gr.Row():
            file_input = gr.File(
                label="PDF documents",
                file_types=[".pdf"],
                type="filepath",
                file_count="multiple",
            )
        parse_button = gr.Button("Parse documents")

        state = gr.State({})
        retrieval_state = gr.State({})

        doc_selector = gr.Dropdown(label="Document", choices=[])
        chunk_selector = gr.Dropdown(label="Chunk", choices=[])
        chunk_text = gr.Textbox(label="Chunk text", lines=20)
        status = gr.Markdown()
        debug_output = gr.JSON(label="Telemetry & Debug")

        with gr.Column():
            gr.Markdown("### Retrieval console")
            with gr.Row():
                retrieval_engine_input = gr.Dropdown(
                    choices=list(RETRIEVAL_LABEL_TO_KEY.keys()),
                    value=DEFAULT_RETRIEVAL_LABEL,
                    label="Retrieval engine",
                    allow_custom_value=False,
                )
                query_input = gr.Textbox(label="User query", placeholder="Ask something about the uploaded documents…", lines=2)
            with gr.Row():
                retrieval_button = gr.Button("Run retrieval")
                rebuild_button = gr.Button("Rebuild indexes", variant="secondary")
            context_preview = gr.Markdown(label="Answer context preview")
            full_chunk_view = gr.Markdown(label="Retrieved chunks")
            retrieval_status = gr.Markdown(label="Retrieval status")
            retrieval_debug = gr.JSON(label="Retrieval debug")
            gold_status = gr.Markdown()
            gold_button = gr.Button("Prepare Gold Inputs", variant="secondary")

        parse_button.click(
            parse_batch_ui,
            inputs=[file_input],
            outputs=[
                state,
                doc_selector,
                chunk_selector,
                chunk_text,
                status,
                debug_output,
                retrieval_state,
                retrieval_engine_input,
                context_preview,
                full_chunk_view,
                retrieval_status,
                retrieval_debug,
                gold_status,
            ],
        )

        doc_selector.change(update_chunk_selector, inputs=[doc_selector, state], outputs=[chunk_selector, chunk_text])
        chunk_selector.change(show_chunk, inputs=[chunk_selector, state], outputs=chunk_text)

        retrieval_button.click(
            run_retrieval,
            inputs=[query_input, retrieval_engine_input, state, retrieval_state],
            outputs=[retrieval_engine_input, context_preview, full_chunk_view, retrieval_status, retrieval_debug, retrieval_state],
        )

        rebuild_button.click(
            rebuild_indexes_action,
            inputs=[retrieval_engine_input, state, retrieval_state],
            outputs=[retrieval_engine_input, context_preview, full_chunk_view, retrieval_status, retrieval_debug, retrieval_state],
        )

        gold_button.click(prepare_gold_inputs, inputs=[state], outputs=[gold_status])

    return demo


def main() -> None:  # pragma: no cover - manual launch
    interface = build_interface()
    interface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":  # pragma: no cover
    main()
