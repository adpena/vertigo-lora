#!/usr/bin/env python3
from __future__ import annotations

"""Storage resolver — external SSD with local fallback.

Large training artifacts (raw data, eval results, adapter weights) go to
APDataStore when mounted, falling back to the local project directory.

Usage:
    from storage import resolve_data_root, resolve_eval_root, resolve_adapters_root

    data_dir = resolve_data_root()      # /Volumes/APDataStore/vertigo/lora/data or local
    eval_dir = resolve_eval_root()      # /Volumes/APDataStore/vertigo/lora/eval or local
"""

from pathlib import Path
import os

EXTERNAL_ROOT = Path(os.environ.get("VERTIGO_LORA_EXTERNAL_ROOT", "/Volumes/APDataStore/vertigo/lora"))
LOCAL_ROOT = Path(__file__).resolve().parent.parent

def _external_available() -> bool:
    return Path("/Volumes/APDataStore").is_dir()

def resolve_data_root() -> Path:
    """Resolve raw data directory — external if available."""
    override = os.environ.get("VERTIGO_LORA_DATA_ROOT")
    if override:
        return Path(override)
    if _external_available():
        p = EXTERNAL_ROOT / "data" / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return LOCAL_ROOT / "data" / "raw"

def resolve_eval_root() -> Path:
    """Resolve eval results directory — external if available."""
    override = os.environ.get("VERTIGO_LORA_EVAL_ROOT")
    if override:
        return Path(override)
    if _external_available():
        p = EXTERNAL_ROOT / "eval"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return LOCAL_ROOT / "data" / "eval"

def resolve_adapters_root() -> Path:
    """Resolve adapters directory — external if available."""
    override = os.environ.get("VERTIGO_LORA_ADAPTERS_ROOT")
    if override:
        return Path(override)
    if _external_available():
        p = EXTERNAL_ROOT / "adapters"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return LOCAL_ROOT / "adapters"

def resolve_runs_root() -> Path:
    """Resolve analysis reports directory — external if available."""
    override = os.environ.get("VERTIGO_LORA_RUNS_ROOT")
    if override:
        return Path(override)
    if _external_available():
        p = EXTERNAL_ROOT / "runs"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return LOCAL_ROOT / "docs" / "runs"

def storage_info() -> dict:
    """Return current storage configuration for diagnostics."""
    external = _external_available()
    return {
        "external_available": external,
        "external_root": str(EXTERNAL_ROOT),
        "data_root": str(resolve_data_root()),
        "eval_root": str(resolve_eval_root()),
        "adapters_root": str(resolve_adapters_root()),
        "runs_root": str(resolve_runs_root()),
    }

if __name__ == "__main__":
    import json
    print(json.dumps(storage_info(), indent=2))
