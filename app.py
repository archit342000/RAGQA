"""Interactive Gradio interface for the document parsing and retrieval pipeline.

This module wires the parsing driver's batch API into a lightweight Hugging
Face Spaces-compatible UI. The page offers:

* **Uploader** – accepts one or more PDF/TXT/MD files and streams them to the
  parser.
* **Controls** – lets the user pick a parsing mode (fast/auto/high-res) while
  hiding advanced tuning knobs behind environment variables.
* **Inspector** – shows available documents, renders individual pages, and,
  when enabled, surfaces an aggregated debug payload.
* **Retriever** – allows operators to run lexical/semantic retrieval with a
  shared cross-encoder reranker and inspect provenance diagnostics.

The functions below are organised roughly in the order the UI consumes them:
configuration helpers, parsing callbacks, retrieval orchestration, and finally
``build_interface`` which stitches everything together.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
from html import escape
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence

import gradio as gr

from env_loader import load_dotenv_once
from pdf_ingest import Config, run_pipeline
from pdf_ingest.pipeline import PipelineResult
from pdf_ingest.pdf_io import load_document_payload
from retrieval import RetrievalConfig, build_indexes, retrieve

# Configure module-level logging using an environment-driven log level so that
# Spaces deployments can flip verbosity without touching source code.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Load .env defaults exactly once before reading environment-driven knobs. The
# helper is intentionally dependency-free so the Spaces build stays minimal.
load_dotenv_once()

# Mapping between user-facing parsing mode labels and the backend strategies
# accepted by the parsing driver. The UI dropdown uses the keys, while the
# driver expects the values.
MODE_LABEL_TO_STRATEGY = {
    "Fast": "fast",
    "Thorough": "thorough",
}

# Retrieval engine labels surfaced in the UI mapped to the engine keys consumed
# by the retrieval driver, plus a helper reverse map for convenience.
RETRIEVAL_LABEL_TO_KEY = {
    "Semantic → Rerank": "semantic",
    "Lexical → Rerank": "lexical",
    "Hybrid → Rerank": "hybrid",
}
RETRIEVAL_KEY_TO_LABEL = {v: k for k, v in RETRIEVAL_LABEL_TO_KEY.items()}

# Initialise a shared retrieval configuration from environment variables so all
# callbacks operate on the same tunable defaults.
RETRIEVAL_CFG = RetrievalConfig.from_env()
DEFAULT_RETRIEVAL_LABEL = RETRIEVAL_KEY_TO_LABEL.get(RETRIEVAL_CFG.default_engine, "Semantic → Rerank")

ARTIFACTS_ROOT = Path(os.getenv("INGEST_ARTIFACT_DIR", "artifacts/ui"))
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


def _artifact_dir_for_doc(doc_id: str) -> Path:
    """Return a unique artifact directory for the provided document."""

    candidate = ARTIFACTS_ROOT / doc_id
    if not candidate.exists():
        return candidate
    for idx in itertools.count(1):
        alt = ARTIFACTS_ROOT / f"{doc_id}_{idx:02d}"
        if not alt.exists():
            return alt
    raise RuntimeError("Unable to allocate artifact directory")


def _show_debug() -> bool:
    """Return True when developer debug output should be visible."""

    # Debug output is often noisy, so we hide it behind an environment switch
    # to avoid overwhelming standard users.
    return os.getenv("SHOW_DEBUG", "false").strip().lower() == "true"


def _gold_export_enabled() -> bool:
    """Return True when gold-set export helpers should be exposed."""

    return os.getenv("ENABLE_GOLD_EXPORT", "false").strip().lower() == "true"


def _mode_choices() -> List[str]:
    """List user-facing parsing options based on server capabilities."""

    return ["Fast", "Thorough"]


def _default_mode() -> str:
    """Resolve the initial parsing mode selection from env defaults."""

    # Read the strategy requested by the operator, convert it back into a UI
    # label, and fall back to a safe default when the value is unrecognised.
    env_default = os.getenv("INGEST_MODE_DEFAULT", "fast").strip().lower()
    reverse_map = {v: k for k, v in MODE_LABEL_TO_STRATEGY.items()}
    candidate = reverse_map.get(env_default, "Fast")
    return candidate if candidate in _mode_choices() else "Fast"


def _default_retrieval_label() -> str:
    """Resolve the default retrieval engine label for the dropdown."""

    # The retrieval engine defaults to the semantic path unless an alternate
    # default is explicitly configured. Guard against stale env overrides.
    return DEFAULT_RETRIEVAL_LABEL if DEFAULT_RETRIEVAL_LABEL in RETRIEVAL_LABEL_TO_KEY else "Semantic → Rerank"


def _build_debug_payload(results: Sequence[PipelineResult]) -> Dict[str, object]:
    """Assemble per-document and aggregate statistics for the debug panel."""

    per_doc: Dict[str, object] = {}
    total_time = 0.0
    total_chunks = 0
    total_tables = 0
    for result in results:
        per_doc[result.doc_id] = {
            "file_name": result.doc_name,
            "stats": result.stats,
            "progress": result.progress,
        }
        total_time += float(result.stats.get("parse_time_s", 0.0))
        total_chunks += len(result.chunks)
        total_tables += sum(1 for chunk in result.chunks if chunk.get("type") == "table")

    aggregate = {
        "documents": len(results),
        "total_chunks": total_chunks,
        "total_tables": total_tables,
        "avg_parse_time_s": round(total_time / max(len(results), 1), 3) if results else 0.0,
    }

    return {
        "aggregate": aggregate,
        "documents": per_doc,
    }


def _fingerprint_chunks(chunks: Sequence[MutableMapping[str, object]]) -> str:
    """Produce a stable fingerprint for a list of chunk dictionaries."""

    # The fingerprint lets us invalidate retrieval indexes when the user parses
    # a new batch of chunks without storing large blobs in Gradio state.
    digest = hashlib.sha1()
    for chunk in chunks:
        chunk_id = str(chunk.get("id"))
        digest.update(chunk_id.encode("utf-8"))
        digest.update(str(chunk.get("token_len", 0)).encode("utf-8"))
    return digest.hexdigest()


def _available_engines(index_map: Dict[str, object]) -> set[str]:
    """Return the set of retrieval engines that are safe to expose."""

    available: set[str] = set()
    lexical_ready = bool(index_map.get("lexical"))
    models = index_map.get("models") or {}
    semantic_index = index_map.get("semantic")
    semantic_ready = bool(semantic_index and models.get("encoder"))

    if lexical_ready:
        available.add("lexical")
    if semantic_ready:
        available.add("semantic")
    if lexical_ready and semantic_ready:
        available.add("hybrid")
    return available


def _engine_dropdown_update(state: Dict[str, object], requested_label: str | None = None):
    """Return an updated dropdown value/choices constrained by availability."""

    # Build the list of choices dynamically so that we never present engines
    # that would immediately fail (for example, semantic retrieval without
    # embeddings). When a requested label is invalid we gracefully fall back to
    # the configured default.
    available = state.get("available")
    choices: List[str] = []
    for label, key in RETRIEVAL_LABEL_TO_KEY.items():
        if available and key not in available:
            continue
        choices.append(label)
    if not choices:
        choices = list(RETRIEVAL_LABEL_TO_KEY.keys())
    preferred = requested_label or _default_retrieval_label()
    if preferred not in choices:
        fallback_key = state.get("cfg", RETRIEVAL_CFG).default_engine if state else RETRIEVAL_CFG.default_engine
        preferred = RETRIEVAL_KEY_TO_LABEL.get(fallback_key, choices[0])
    return gr.update(choices=choices, value=preferred)


def _format_context_preview(chunks: Sequence[Dict[str, object]]) -> str:
    """Return a condensed context preview highlighting top chunks."""

    # Users expect a quick glance at the snippets feeding the model. We keep the
    # preview short to fit on screen while still surfacing score and location.
    if not chunks:
        return "No context selected yet. Run a query to populate this view."
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("doc_name", "Unknown document")
        page_start = int(chunk.get("page_start", 0) or 0)
        page_end = int(chunk.get("page_end", page_start) or page_start)
        page_label = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
        score = float(chunk.get("score", 0.0))
        snippet = str(chunk.get("text", "")).strip()
        # Collapse whitespace to keep the preview compact and trim overly long
        # passages that would dominate the panel.
        snippet_single = " ".join(snippet.split())
        if len(snippet_single) > 420:
            snippet_single = snippet_single[:417] + "..."
        lines.append(
            f"{idx}. **{doc_name} {page_label}** (score {score:.3f})\n> {snippet_single}"
        )
    return "\n\n".join(lines)



def _format_full_chunks(chunks: Sequence[Dict[str, object]]) -> str:
    """Render the full reranked chunks in Markdown for operator inspection."""

    # When no chunks are available we provide a friendly hint to run retrieval.
    if not chunks:
        return "No chunks retrieved yet."
    sections: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("doc_name", "Unknown document")
        page_start = int(chunk.get("page_start", 0) or 0)
        page_end = int(chunk.get("page_end", page_start) or page_start)
        page_label = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
        score = float(chunk.get("score", 0.0))
        chunk_id = chunk.get("id", "")
        raw_text = str(chunk.get("text", "")).strip()
        # Backticks can break fenced code blocks; insert a space to keep the fence intact.
        safe_text = raw_text.replace("```", "`` `")
        meta = f"ID: {chunk_id} | Score: {score:.3f}"
        sections.append(
            f"#### Chunk {idx}: {doc_name} {page_label}\n{meta}\n```text\n{safe_text}\n```"
        )
    return "\n\n".join(sections)


def _format_status(provenance: Dict[str, object], timings: Dict[str, float]) -> str:
    """Summarise retrieval provenance, notices, and timing information."""

    engine_label = RETRIEVAL_KEY_TO_LABEL.get(str(provenance.get("engine", "lexical")), "Lexical → Rerank")
    cache_hit = bool(provenance.get("cache_hit"))
    notices = provenance.get("notices") or []
    # Translate short error codes into user-friendly explanations. This keeps
    # the UI compact while still providing actionable diagnostics.
    notice_labels = {
        "semantic_unavailable": "Semantic index unavailable; fell back to lexical.",
        "reranker_unavailable": "Reranker unavailable; used first-pass ranking.",
        "no_candidates": "No candidates returned by the first-pass index.",
        "empty_query": "Query was empty.",
        "hybrid_semantic_unavailable": "Hybrid fallback: semantic index missing.",
        "hybrid_lexical_unavailable": "Hybrid fallback: lexical index missing.",
    }
    lines = [f"Engine: {engine_label}", f"Cache: {'hit' if cache_hit else 'miss'}"]
    if notices:
        pretty = [notice_labels.get(str(code), str(code)) for code in notices]
        lines.append("Notices: " + "; ".join(pretty))
    if timings:
        encode_ms = timings.get("encode_ms", 0.0)
        search_ms = timings.get("search_ms", 0.0)
        rerank_ms = timings.get("rerank_ms", 0.0)
        pack_ms = timings.get("pack_ms", 0.0)
        total_ms = timings.get("total_ms", 0.0)
        lines.append(
            "Timings (ms): encode {:.1f} | search {:.1f} | rerank {:.1f} | pack {:.1f} | total {:.1f}".format(
                encode_ms,
                search_ms,
                rerank_ms,
                pack_ms,
                total_ms,
            )
        )
        bm25_ms = timings.get("bm25_ms", 0.0)
        faiss_ms = timings.get("faiss_ms", 0.0)
        fusion_ms = timings.get("fusion_ms", 0.0)
        sub_parts = []
        if bm25_ms:
            sub_parts.append(f"bm25 {bm25_ms:.1f}")
        if faiss_ms:
            sub_parts.append(f"faiss {faiss_ms:.1f}")
        if fusion_ms:
            sub_parts.append(f"fusion {fusion_ms:.1f}")
        if sub_parts:
            lines.append("Hybrid timings (ms): " + " | ".join(sub_parts))
    fusion_info = provenance.get("fusion")
    if fusion_info:
        method = fusion_info.get("method") or "unknown"
        counts = fusion_info.get("candidate_counts", {})
        count_str = ", ".join(
            f"{key}={counts.get(key, 0)}" for key in ("semantic", "lexical", "fused")
        )
        lines.append(f"Fusion: {method} ({count_str})")
    return "\n".join(lines)


def _ensure_retrieval_state(
    app_state: Dict[str, object] | None,
    retrieval_state: Dict[str, object],
    *,
    force: bool = False,
) -> Dict[str, object]:
    """Ensure retrieval indexes exist and are synced with the parsed chunks."""

    # Users must upload documents before retrieval can operate. Guard early so
    # the UI surfaces a helpful error instead of crashing later.
    if not app_state:
        raise gr.Error("Parse documents before running retrieval.")
    chunk_list = app_state.get("chunk_list") or []
    if not chunk_list:
        raise gr.Error("No chunks available; parsing did not produce content.")

    fingerprint = app_state.get("chunk_fingerprint")
    cfg: RetrievalConfig = retrieval_state.get("cfg") or RETRIEVAL_CFG
    retrieval_state["cfg"] = cfg

    # Rebuild retrieval indexes when forced or when the chunk fingerprint no
    # longer matches the cached copy. This keeps indexes aligned with the most
    # recent parsing run without rebuilding on every query.
    if force or retrieval_state.get("indexes") is None or retrieval_state.get("fingerprint") != fingerprint:
        indexes = build_indexes(chunk_list, cfg)
        retrieval_state["indexes"] = indexes
        retrieval_state["fingerprint"] = fingerprint
        retrieval_state["available"] = _available_engines(indexes)
        retrieval_state["errors"] = indexes.get("errors", {})
    return retrieval_state


def parse_batch(files, mode_label: str, *, full_output: bool = False):
    """Handle a UI request to parse documents and emit retrieval chunks."""

    if not files:
        raise gr.Error("Please upload at least one document.")

    if not isinstance(files, Sequence):
        files = [files]
    file_paths = [Path(f) for f in files if f]
    if not file_paths:
        raise gr.Error("No readable files provided.")

    mode = MODE_LABEL_TO_STRATEGY.get(mode_label or "Fast", "fast")
    status_lines: List[str] = []

    ingest_results: List[PipelineResult] = []
    chunk_registry: Dict[str, dict] = {}
    doc_chunk_map: Dict[str, dict] = {}
    chunk_list: List[Dict[str, object]] = []
    parsed_docs_payload: List[Dict[str, object]] = []
    ingest_stats: Dict[str, dict] = {}

    for file_path in file_paths:
        if not file_path.exists():
            status_lines.append(f"Missing file: {file_path}")
            continue
        artifact_dir = _artifact_dir_for_doc(file_path.stem)
        config = Config()
        config.mode = mode
        try:
            result = run_pipeline(file_path, artifact_dir, config)
            ingest_results.append(result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to parse %s", file_path)
            status_lines.append(f"Failed to parse {file_path.name}: {exc}")
            continue

        ingest_stats[result.doc_id] = result.stats
        status_lines.append(
            f"{result.doc_name}: {len(result.chunks)} chunks ({result.stats.get('parse_time_s', 0)}s)"
        )

        pages_payload = []
        pages, _ = load_document_payload(file_path)
        for page in pages:
            page_text = "\n".join(line.text for line in page.lines)
            pages_payload.append({"page_num": page.index + 1, "text": page_text, "glyphs": page.glyph_count})
        parsed_docs_payload.append(
            {
                "doc_id": result.doc_id,
                "doc_name": result.doc_name,
                "pages": pages_payload,
                "artifact_dir": str(artifact_dir),
            }
        )

        doc_entry = doc_chunk_map.setdefault(
            result.doc_id,
            {"doc_name": result.doc_name, "chunk_keys": [], "artifact_dir": str(artifact_dir)},
        )

        for chunk in result.chunks:
            chunk_id = str(chunk.get("chunk_id") or len(doc_entry["chunk_keys"]))
            chunk_key = f"{result.doc_id}:{chunk_id}"
            page_spans = chunk.get("page_spans") or []
            page_numbers = [span[0] for span in page_spans if span]
            page_start = min(page_numbers) if page_numbers else 0
            page_end = max(page_numbers) if page_numbers else page_start
            section_title = None
            hints = chunk.get("section_hints") or []
            if hints:
                section_title = hints[0]
            record = {
                "id": chunk_key,
                "doc_id": result.doc_id,
                "doc_name": result.doc_name,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_title,
                "text": chunk.get("text", ""),
                "token_len": int(chunk.get("tokens_est", 0) or 0),
                "meta": {
                    "type": chunk.get("type", "body"),
                    "neighbors": chunk.get("neighbors", {}),
                    "table_csv": chunk.get("table_csv"),
                },
            }
            chunk_registry[chunk_key] = record
            doc_entry["chunk_keys"].append(chunk_key)
            if chunk.get("type") == "body":
                chunk_list.append({k: v for k, v in record.items() if k != "section_title"})

        if not any(chunk.get("type") == "body" for chunk in result.chunks):
            status_lines.append(f"No body chunks generated for {result.doc_name}.")

    if not ingest_results and not chunk_list:
        raise gr.Error("Parsing failed; check logs for details.")

    doc_choices = [
        (entry["doc_name"], doc_id)
        for doc_id, entry in doc_chunk_map.items()
    ]

    default_doc = doc_choices[0][1] if doc_choices else None
    chunk_choices: List[tuple[str, str]] = []
    default_chunk_key = None
    default_text = ""
    if default_doc:
        keys = doc_chunk_map[default_doc]["chunk_keys"]
        chunk_choices = [
            (
                f"{chunk_registry[key]['meta']['type'].title()} · Pages {chunk_registry[key]['page_start']}-{chunk_registry[key]['page_end']}",
                key,
            )
            for key in keys
        ]
        if chunk_choices:
            default_chunk_key = chunk_choices[0][1]
            default_text = chunk_registry[default_chunk_key]["text"]

    fingerprint = _fingerprint_chunks(chunk_list)
    state_payload = {
        "chunks": chunk_registry,
        "doc_map": doc_chunk_map,
        "chunk_list": chunk_list,
        "chunk_fingerprint": fingerprint,
        "parsed_docs": parsed_docs_payload,
        "ingest_stats": ingest_stats,
    }

    status_lines.append("Ready.")

    debug_enabled = _show_debug()
    debug_payload = _build_debug_payload(ingest_results) if debug_enabled else None
    debug_update = gr.update(value=debug_payload, visible=debug_enabled)

    retrieval_state = {"cfg": RETRIEVAL_CFG, "indexes": None, "available": None, "fingerprint": None, "errors": {}}
    engine_dropdown_reset = _engine_dropdown_update(retrieval_state)
    retrieval_debug_reset = gr.update(value=None, visible=debug_enabled)
    gold_status_reset = gr.update(value="", visible=_gold_export_enabled())

    base_outputs = (
        state_payload,
        gr.update(choices=doc_choices, value=default_doc or None),
        gr.update(choices=chunk_choices, value=default_chunk_key or None),
        default_text,
        "\n".join(status_lines),
        debug_update,
    )
    if not full_output:
        return base_outputs

    return base_outputs + (
        retrieval_state,
        engine_dropdown_reset,
        "",
        "No chunks retrieved yet.",
        "Awaiting query.",
        retrieval_debug_reset,
        gold_status_reset,
    )


def parse_batch_ui(files, mode_label: str):
    """UI wrapper returning the expanded set of outputs for Gradio."""

    return parse_batch(files, mode_label, full_output=True)


def prepare_gold_inputs(state: Dict[str, object]) -> str:
    """Persist parsed documents and chunks to disk for gold-set authoring."""

    if not _gold_export_enabled():
        return "Gold export disabled by configuration."
    if not state:
        return "Parse documents before exporting gold inputs."

    parsed_docs = state.get("parsed_docs") or []
    chunk_list = state.get("chunk_list") or []
    if not parsed_docs:
        return "No parsed documents found; run parsing again."

    export_root = Path(os.getenv("GOLD_EXPORT_ROOT", "artifacts"))
    parsed_dir = export_root / "parsed_docs"
    chunks_path = export_root / "chunks.jsonl"

    parsed_dir.mkdir(parents=True, exist_ok=True)
    for doc in parsed_docs:
        doc_path = parsed_dir / f"{doc['doc_id']}.json"
        doc_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    if chunk_list:
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in chunk_list:
                fh.write(json.dumps(chunk, ensure_ascii=False))
                fh.write("\n")

    return (
        f"Exported {len(parsed_docs)} parsed documents and {len(chunk_list)} chunks to "
        f"{export_root.resolve()}"
    )


def update_chunk_selector(selected_doc: str, state: Dict[str, object]):
    """Refresh chunk selector options when the chosen document changes."""

    # Guard against missing state when the user clears uploads or switches tabs.
    if not selected_doc or not state:
        return gr.update(choices=[], value=None), ""

    doc_map = state.get("doc_map", {})
    chunk_registry = state.get("chunks", {})
    entry = doc_map.get(selected_doc, {})
    keys = entry.get("chunk_keys", [])
    chunk_choices = [
        (
            f"{chunk_registry[key]['meta']['type'].title()} · Pages {chunk_registry[key]['page_start']}-{chunk_registry[key]['page_end']}",
            key,
        )
        for key in keys
    ]
    default_key = chunk_choices[0][1] if chunk_choices else None
    default_text = chunk_registry[default_key]["text"] if default_key else ""
    return gr.update(choices=chunk_choices, value=default_key), default_text


def show_chunk(selected_chunk: str, state: Dict[str, object]):
    """Return the full chunk text for the viewer component."""

    # The inspector simply displays the stored chunk text. Missing keys are
    # tolerated so the UI does not crash if state desynchronises.
    if not selected_chunk or not state:
        return ""
    chunk_registry = state.get("chunks", {})
    chunk = chunk_registry.get(selected_chunk)
    if not chunk:
        return ""
    return chunk["text"]


def run_retrieval(query: str, engine_label: str, app_state: Dict[str, object], retrieval_state: Dict[str, object]):
    """Execute retrieval for the current session and update UI components."""

    retrieval_state = retrieval_state or {"cfg": RETRIEVAL_CFG}
    try:
        retrieval_state = _ensure_retrieval_state(app_state, retrieval_state)
    except gr.Error as err:
        # Early validation issues (e.g. no chunks) map to a blank preview but we
        # still surface the exception message in the status panel.
        return (
            _engine_dropdown_update(retrieval_state),
            gr.update(value=""),
            "No chunks retrieved yet.",
            str(err),
            gr.update(value=None, visible=_show_debug()),
            retrieval_state,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to prepare retrieval state: %s", exc)
        return (
            _engine_dropdown_update(retrieval_state),
            gr.update(value=""),
            "No chunks retrieved yet.",
            f"Failed to build retrieval indexes: {exc}",
            gr.update(value=None, visible=_show_debug()),
            retrieval_state,
        )

    # Resolve the engine key from the selected UI label and fall back to lexical
    # retrieval automatically when semantic retrieval is unavailable.
    engine_key = RETRIEVAL_LABEL_TO_KEY.get(engine_label, retrieval_state["cfg"].default_engine)
    available = retrieval_state.get("available") or set()
    if engine_key == "semantic" and "semantic" not in available:
        engine_key = "lexical"

    try:
        # Execute the retrieval driver, which returns ranked chunks plus timing
        # diagnostics and provenance metadata.
        result = retrieve(query, engine_key, retrieval_state.get("indexes"), retrieval_state["cfg"])
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        return (
            _engine_dropdown_update(retrieval_state, RETRIEVAL_KEY_TO_LABEL.get(engine_key)),
            gr.update(value=""),
            "No chunks retrieved yet.",
            f"Retrieval failed: {exc}",
            gr.update(value=None, visible=_show_debug()),
            retrieval_state,
        )

    provenance = result.get("provenance", {})
    timings = result.get("timings", {})
    final_chunks = result.get("final_chunks", [])

    # Update the engine dropdown in case semantic retrieval fell back to
    # lexical mid-flight, keeping the UI in sync with runtime reality.
    engine_actual = str(provenance.get("engine", engine_key))
    dropdown_update = _engine_dropdown_update(
        retrieval_state,
        RETRIEVAL_KEY_TO_LABEL.get(engine_actual, engine_label),
    )

    context_preview = _format_context_preview(final_chunks)
    status_text = _format_status(provenance, timings)

    # Default to clearing the debug panel; repopulate when debug mode is active.
    debug_update = gr.update(value=None, visible=_show_debug())
    if _show_debug():
        debug_payload = {
            "engine": engine_actual,
            "requested_engine": provenance.get("requested_engine", engine_actual),
            "first_pass": [{"id": cid, "score": score} for cid, score in provenance.get("first_pass", [])],
            "rerank": [{"id": cid, "score": score} for cid, score in provenance.get("rerank", [])],
            "notices": provenance.get("notices", []),
            "cache_hit": provenance.get("cache_hit", False),
            "timings_ms": timings,
            "selected_chunks": [chunk.get("id") for chunk in final_chunks],
        }
        fusion_info = provenance.get("fusion")
        if fusion_info:
            debug_payload["fusion"] = fusion_info
        debug_update = gr.update(value=debug_payload, visible=True)

    return dropdown_update, context_preview, _format_full_chunks(final_chunks), status_text, debug_update, retrieval_state


def rebuild_indexes_action(engine_label: str, app_state: Dict[str, object], retrieval_state: Dict[str, object]):
    """Force a rebuild of retrieval indexes and clear previous results."""

    retrieval_state = retrieval_state or {"cfg": RETRIEVAL_CFG}
    retrieval_state.pop("indexes", None)
    retrieval_state.pop("fingerprint", None)
    try:
        retrieval_state = _ensure_retrieval_state(app_state, retrieval_state, force=True)
    except gr.Error as err:
        # Propagate the friendly error while clearing chunk previews.
        return (
            _engine_dropdown_update(retrieval_state),
            gr.update(value=""),
            "No chunks retrieved yet.",
            str(err),
            gr.update(value=None, visible=_show_debug()),
            retrieval_state,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to rebuild retrieval indexes: %s", exc)
        return (
            _engine_dropdown_update(retrieval_state),
            gr.update(value=""),
            "No chunks retrieved yet.",
            f"Rebuild failed: {exc}",
            gr.update(value=None, visible=_show_debug()),
            retrieval_state,
        )

    available = retrieval_state.get("available") or set()
    status_notes = ["Indexes rebuilt."]
    if "semantic" not in available:
        status_notes.append("Semantic retrieval disabled (index unavailable).")

    dropdown_update = _engine_dropdown_update(retrieval_state, RETRIEVAL_KEY_TO_LABEL.get(retrieval_state["cfg"].default_engine))
    debug_update = gr.update(value=None, visible=_show_debug())
    return dropdown_update, gr.update(value=""), "No chunks retrieved yet.", " ".join(status_notes), debug_update, retrieval_state


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks layout for the demo."""

    # Resolve static dropdown defaults up-front so the layout code reads
    # linearly below.
    mode_choices = _mode_choices()
    default_mode = _default_mode()
    debug_visible = _show_debug()

    with gr.Blocks(title="Document Parser Preview") as demo:
        gr.Markdown("## Rapid Document Parser Preview")
        with gr.Row():
            file_input = gr.File(
                label="Documents",
                file_types=[".pdf", ".txt", ".md"],
                type="filepath",
                file_count="multiple",
            )
            mode_input = gr.Dropdown(
                choices=mode_choices,
                value=default_mode,
                label="Parsing mode",
                allow_custom_value=False,
            )

        parse_button = gr.Button("Parse documents")

        # Gradio state objects hold structured results from parsing and
        # retrieval so subsequent callbacks can operate without re-running
        # heavy pipelines.
        state = gr.State({})
        retrieval_state = gr.State({})

        doc_selector = gr.Dropdown(label="Document", choices=[])
        chunk_selector = gr.Dropdown(label="Chunk", choices=[])
        chunk_text = gr.Textbox(label="Chunk text", lines=20)
        status = gr.Markdown()
        debug_output = gr.JSON(label="Debug", visible=debug_visible)

        with gr.Column():
            gr.Markdown("### Retrieval console")
            with gr.Row():
                retrieval_engine_input = gr.Dropdown(
                    choices=list(RETRIEVAL_LABEL_TO_KEY.keys()),
                    value=_default_retrieval_label(),
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
            retrieval_debug = gr.JSON(label="Retrieval debug", visible=debug_visible)
            gold_status = gr.Markdown(visible=_gold_export_enabled())
            gold_button = gr.Button("Prepare Gold Inputs", variant="secondary", visible=_gold_export_enabled())

        # Wire the parse button to the parser callback, refreshing all dependent
        # UI components (state blobs, dropdown choices, chunk preview panes, and
        # debug outputs).
        parse_button.click(
            parse_batch_ui,
            inputs=[file_input, mode_input],
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

        # Keep the chunk preview in sync when the operator selects a different
        # document or chunk from the inspector dropdowns.
        doc_selector.change(update_chunk_selector, inputs=[doc_selector, state], outputs=[chunk_selector, chunk_text])
        chunk_selector.change(show_chunk, inputs=[chunk_selector, state], outputs=chunk_text)

        # Attach retrieval actions to their respective buttons so the UI updates
        # the answer preview, full chunk list, and diagnostic panes in one go.
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

        if _gold_export_enabled():
            gold_button.click(prepare_gold_inputs, inputs=[state], outputs=[gold_status])

    return demo


def main():  # pragma: no cover - manual launch
    """Launch the Gradio interface with sensible defaults."""

    interface = build_interface()
    interface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
