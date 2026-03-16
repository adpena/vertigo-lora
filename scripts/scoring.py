"""Shared scoring functions for Vertigo LoRA evaluation pipelines."""

from __future__ import annotations
import re, shutil, subprocess  # noqa: E401

from prompts import VERTIGO_SYSTEM_PROMPT  # noqa: F401 — re-exported for consumers

CONVENTION_PATTERNS = {
    "strict_mode": r"--!strict",
    "native_annotation": r"@native",
    "init_lifecycle": r":Init\(\)",
    "start_lifecycle": r":Start\(\)",
    "vector_simd": r"vector\.(create|normalize|magnitude|dot|cross)",
    "table_freeze": r"table\.freeze",
}

CATEGORY_CONVENTIONS = {
    "coding": ["strict_mode", "native_annotation", "init_lifecycle", "start_lifecycle"],
    "bugfix": ["strict_mode"],
    "architecture": ["strict_mode", "init_lifecycle", "start_lifecycle"],
    "mcp_tool_calling": [],
    "embodiment": [],
}

CODE_EXPECTED_CATEGORIES = {"coding", "bugfix", "architecture", "mcp_tool_calling"}

# Luau compile stubs — generated at import time
_TBL = (
    "game workspace script plugin shared Enum Instance Vector3 Vector2 CFrame "
    "Color3 BrickColor UDim2 UDim Rect Region3 Ray TweenInfo NumberRange "
    "NumberSequence NumberSequenceKeypoint ColorSequence ColorSequenceKeypoint "
    "PhysicalProperties Random DateTime Os task debug buffer bit32 utf8"
).split()
_FN = {
    "tick": "() -> number",
    "time": "() -> number",
    "wait": "(n: number?) -> number",
    "delay": "(t: number, f: () -> ()) -> ()",
    "spawn": "(f: () -> ()) -> ()",
    "warn": "(...any) -> ()",
    "typeof": "(any) -> string",
}
ROBLOX_GLOBAL_STUBS = "".join(f"local {g} = {{}} :: any\n" for g in _TBL) + "".join(
    f"local {n} = (nil :: any) :: {s}\n" for n, s in _FN.items()
)

_LUAU_RE = re.compile(r"(local\s+\w+\s*=|function\s+\w+|return\s+\w+|\bend\b.*\n|--!strict)")
_BLOCK_RE = re.compile(r"```(?:lua|luau)\s*\n(.*?)```", re.DOTALL)


def score_correctness(response: str, expected_patterns: list[str]) -> float:
    if not expected_patterns:
        return 1.0
    return sum(1 for p in expected_patterns if re.search(p, response, re.IGNORECASE)) / len(expected_patterns)


def score_convention(response: str, category: str) -> float | None:
    keys = CATEGORY_CONVENTIONS.get(category, [])
    if not keys:
        return None
    pats = {k: CONVENTION_PATTERNS[k] for k in keys if k in CONVENTION_PATTERNS}
    return sum(1 for p in pats.values() if re.search(p, response)) / len(pats) if pats else None


def score_tool_selection(response: str, task: dict) -> float | None:
    tools = task.get("tools")
    if not tools:
        return None
    names = [t["function"]["name"] for t in tools if "function" in t]
    return sum(1 for n in names if n in response) / len(names) if names else None


def score_code_presence(response: str, category: str | None = None) -> float | None:
    if category is not None and category not in CODE_EXPECTED_CATEGORIES:
        return None
    return 1.0 if ("```" in response or _LUAU_RE.search(response)) else 0.0


def score_luau_compiles(response: str) -> bool | None:
    """Check if Luau code blocks compile via luau-compile. None if binary missing."""
    luau_bin = shutil.which("luau-compile")
    if luau_bin is None:
        return None
    blocks = _BLOCK_RE.findall(response)
    if not blocks:
        return False
    for block in blocks:
        try:
            r = subprocess.run(
                [luau_bin, "--stdin"],
                input=ROBLOX_GLOBAL_STUBS + "\n" + block,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, OSError):
            continue
    return False


def score_failure_penalty(response: str) -> float:
    """Penalty multiplier (0.0-1.0) for empty/error responses."""
    s = response.strip()
    if len(s) < 10 or s.startswith("[TIMEOUT]") or s.startswith("[ERROR"):
        return 0.0
    return 0.3 if len(s) < 50 else 1.0


def score_task(response: str, task: dict) -> dict:
    """Composite scorer returning all dimensions + overall score."""
    cat = task.get("category", "coding")
    correctness = score_correctness(response, task.get("expected_patterns", []))
    convention = score_convention(response, cat)
    tool_sel = score_tool_selection(response, task)
    code_pres = score_code_presence(response, cat)
    penalty = score_failure_penalty(response)
    w: dict[str, tuple[float, float]] = {"correctness": (correctness, 0.4)}
    if convention is not None:
        w["convention"] = (convention, 0.2)
    if tool_sel is not None:
        w["tool_selection"] = (tool_sel, 0.2)
    if code_pres is not None:
        w["code_presence"] = (code_pres, 0.2)
    tw = sum(wt for _, wt in w.values())
    overall = sum(sc * wt / tw for sc, wt in w.values()) * penalty
    return {
        "correctness": round(correctness, 3),
        "convention_score": round(convention, 3) if convention is not None else None,
        "tool_selection": round(tool_sel, 3) if tool_sel is not None else None,
        "code_presence": round(code_pres, 3) if code_pres is not None else None,
        "failure_penalty": round(penalty, 3),
        "overall": round(overall, 3),
        "luau_compiles": score_luau_compiles(response),
        "applicable_dimensions": list(w.keys()),
    }
