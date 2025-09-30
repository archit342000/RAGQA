from __future__ import annotations

import hashlib
try:
    from uuid6 import uuid7
except Exception:
    from uuid import uuid4 as _uuid4

    def uuid7():
        return _uuid4()


DOC_ID_PREFIX_LEN = 16


def compute_doc_id(file_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(file_bytes)
    return digest.hexdigest()[:DOC_ID_PREFIX_LEN]


def make_block_id(page: int, seq: int) -> str:
    return f"b_p{page:02d}_{seq:04d}"


def make_chunk_id() -> str:
    return str(uuid7())
