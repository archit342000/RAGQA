"""Test configuration ensuring local packages are importable."""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "fitz" not in sys.modules:
    stub = types.SimpleNamespace()
    sys.modules["fitz"] = stub

if "pytesseract" not in sys.modules:
    pytesseract_stub = types.SimpleNamespace(image_to_string=lambda *args, **kwargs: "")
    sys.modules["pytesseract"] = pytesseract_stub

if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.open = lambda *args, **kwargs: types.SimpleNamespace()
    pil_module.Image = image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module
