#!/usr/bin/env python3
from __future__ import annotations

"""Extract training data from Roblox creator-docs. Output: data/raw/roblox_creator_docs_full.jsonl"""

import hashlib, json, re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent / "data" / "sources" / "creator-docs" / "content" / "en-us"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "roblox_creator_docs_full.jsonl"
SYSTEM = (
    "You are an expert Roblox game developer specializing in Luau. "
    "You write production-grade Luau code with --!strict mode, full type annotations, "
    "and follow modern Roblox best practices. When explaining code, include reasoning "
    "about WHY each design choice was made."
)
MIN_FILE, MIN_CODE_LINES, MIN_PROSE = 200, 3, 500
CAT_MAP = dict(
    luau="general_luau",
    reference="api_reference",
    cloud="cloud_services",
    physics="physics",
    animation="animation",
    players="players",
    chat="chat",
    input="input",
    audio="audio",
    environment="environment",
    effects="effects",
    parts="parts",
    avatar="avatar",
    art="art",
    ui="ui",
    production="production",
    education="education",
    get_started="getting_started",
    resources="resources",
)


def _sid(t: str) -> str:
    return hashlib.blake2b(t.encode(), digest_size=8).hexdigest()


def strip_fm(text: str) -> tuple[dict[str, str], str]:
    fm: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            for ln in text[3:end].strip().splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fm[k.strip()] = v.strip().strip('"').strip("'")
            text = text[end + 3 :].strip()
    return fm, text


_CODE_RE = re.compile(r"```(?:lua|luau)(?:\s+title=\"([^\"]*)\")?\s*\n(.*?)```", re.DOTALL)


def code_blocks(text: str) -> list[dict[str, str]]:
    return [
        {"title": m.group(1) or "", "code": m.group(2).strip()}
        for m in _CODE_RE.finditer(text)
        if m.group(2).strip().count("\n") + 1 >= MIN_CODE_LINES
    ]


def sections(body: str) -> list[dict[str, str]]:
    parts = re.split(r"^(#{2,3}\s+.+)$", body, flags=re.MULTILINE)
    secs: list[dict[str, str]] = []
    if parts[0].strip():
        secs.append({"heading": "", "content": parts[0].strip()})
    for i in range(1, len(parts), 2):
        secs.append(
            {"heading": parts[i].lstrip("#").strip(), "content": parts[i + 1].strip() if i + 1 < len(parts) else ""}
        )
    return secs


def categorize(path: Path) -> str:
    rel = path.relative_to(DOCS_ROOT)
    return CAT_MAP.get(rel.parts[0].lower().replace("-", "_"), "api_usage") if rel.parts else "general"


def difficulty(content: str, cbs: list[dict[str, str]]) -> int:
    t, s = content.lower(), 1
    if any(k in t for k in ("pcall", "coroutine", "metatab", "promise", "async")):
        s += 1
    if any(k in t for k in ("remotevent", "remotefunction", "bindable", "server", "client")):
        s += 1
    if any(k in t for k in ("datastore", "memory store", "ordered data", "messaging")):
        s += 1
    if sum(b["code"].count("\n") + 1 for b in cbs) > 30:
        s += 1
    return min(s, 5)


_STRIP = [
    (r"<Alert[^>]*>", ""),
    (r"</Alert>", ""),
    (r"<img[^>]*/>", ""),
    (r"<video[^>]*>.*?</video>", ""),
    (r"<GridContainer[^>]*>", ""),
    (r"</GridContainer>", ""),
    (r"<figure>.*?</figure>", ""),
    (r"<Typography[^>]*>(.*?)</Typography>", r"\1"),
    (r"</?[A-Z][a-zA-Z]*[^>]*>", ""),
    (r"\n{3,}", "\n\n"),
]


def clean(text: str) -> str:
    for pat, rep in _STRIP:
        text = re.sub(pat, rep, text, flags=re.DOTALL)
    return text.strip()


def _trim(prose: str, limit: int = 1000) -> str:
    if len(prose) > limit:
        prose = prose[:limit].rsplit(".", 1)[0] + "."
    return prose


def _no_code(prose: str) -> str:
    return re.sub(r"```(?:lua|luau).*?```", "", prose, flags=re.DOTALL).strip()


def _think(topic: str, cat: str) -> str:
    return f"<think>\nThe user is asking about {topic}. This involves {cat.replace('_', ' ')} in Roblox. Let me provide a clear explanation with code.\n</think>\n\n"


def _ex(msgs: list[dict[str, str]], cat: str, diff: int, fp: Path) -> dict:
    return {
        "messages": msgs,
        "source": "roblox_creator_docs",
        "category": cat,
        "has_reasoning": True,
        "difficulty": diff,
        "verified": False,
        "provenance": {
            "source_id": _sid(msgs[-1]["content"]),
            "rights_basis": "open_source",
            "license": "CC-BY-4.0",
            "task_family": "sft_scripter",
            "modality": "code",
            "project_id": "roblox/creator-docs",
        },
        "file_path": str(fp),
    }


def style_a(sec: dict, cb: dict, title: str, cat: str, diff: int, fp: Path) -> dict | None:
    heading = sec["heading"] or title
    if not heading:
        return None
    prose = _trim(_no_code(clean(sec["content"])), 1200)
    if len(prose) < 80:
        return None
    body = _think(heading, cat) + prose + f"\n\n```luau\n{cb['code']}\n```"
    return _ex(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Explain this Roblox API pattern: {heading}"},
            {"role": "assistant", "content": body},
        ],
        cat,
        diff,
        fp,
    )


def style_b(sec: dict, title: str, cat: str, diff: int, fp: Path) -> dict | None:
    heading = sec["heading"]
    if not heading:
        return None
    codes = code_blocks(sec["content"])
    if not codes:
        return None
    prose = _trim(_no_code(clean(sec["content"])))
    if len(prose) < 100:
        return None
    h = heading.lower()
    q = (
        f"How do I {h} in Roblox?"
        if h.startswith(("create", "set", "add", "remove", "iterate", "read", "write"))
        else f"How do I implement {heading} in Roblox?"
    )
    code_str = "\n\n".join(f"```luau\n{c['code']}\n```" for c in codes)
    return _ex(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": _think(heading, cat) + prose + "\n\n" + code_str},
        ],
        cat,
        diff,
        fp,
    )


def style_c(cls: str, member: str, body: str, cat: str, diff: int, fp: Path) -> dict | None:
    for sec in sections(body):
        if f"Class.{cls}" not in sec["content"] or member not in sec["content"]:
            continue
        codes = code_blocks(sec["content"])
        if not codes:
            continue
        prose = _trim(_no_code(clean(sec["content"])), 800)
        if len(prose) < 60:
            continue
        code_str = "\n\n".join(f"```luau\n{c['code']}\n```" for c in codes[:2])
        return _ex(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"What is {cls}:{member} and how do I use it?"},
                {
                    "role": "assistant",
                    "content": _think(f"{cls}.{member}", "api_reference") + prose + "\n\n" + code_str,
                },
            ],
            cat,
            diff,
            fp,
        )
    return None


_API_RE = re.compile(r"`Class\.(\w+)(?:[.:|](\w+))?`")


def process(fp: Path) -> list[dict]:
    text = fp.read_text(errors="replace")
    if len(text) < MIN_FILE:
        return []
    fm, body = strip_fm(text)
    title = fm.get("title", "")
    if len(body) < 100:
        return []
    cbs = code_blocks(body)
    if not cbs and len(clean(body)) < MIN_PROSE:
        return []
    cat, diff = categorize(fp), difficulty(body, cbs)
    secs, exs, seen = sections(body), [], set()

    def add(e):
        if e is None:
            return
        k = _sid(e["messages"][-1]["content"][:300])
        if k not in seen:
            seen.add(k)
            exs.append(e)

    for s in secs:
        sc = code_blocks(s["content"])
        for cb in sc[:2]:
            add(style_a(s, cb, title, cat, diff, fp))
        if sc:
            add(style_b(s, title, cat, diff, fp))
    apis = set()
    for m in _API_RE.finditer(body):
        member = m.group(2)
        if not member:
            continue
        key = f"{m.group(1)}.{member}"
        if key in apis:
            continue
        e = style_c(m.group(1), member, body, cat, diff, fp)
        if e:
            apis.add(key)
            add(e)
        if len(apis) >= 3:
            break
    return exs


def main() -> None:
    files = sorted(DOCS_ROOT.rglob("*.md"))
    print(f"Found {len(files)} markdown files in {DOCS_ROOT}")
    all_ex, n_files = [], 0
    for f in files:
        ex = process(f)
        if ex:
            n_files += 1
            all_ex.extend(ex)
    seen, deduped = set(), []
    for e in all_ex:
        h = _sid(e["messages"][-1]["content"])
        if h not in seen:
            seen.add(h)
            deduped.append(e)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as fout:
        for e in deduped:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    cats: dict[str, int] = {}
    diffs: dict[int, int] = {}
    for e in deduped:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
        diffs[e["difficulty"]] = diffs.get(e["difficulty"], 0) + 1
    print(f"Files: {len(files)} scanned, {n_files} with examples")
    print(f"Total examples: {len(deduped)}")
    print("\nBy category:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\nBy difficulty:")
    for k, v in sorted(diffs.items()):
        print(f"  {k}: {v}")
    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
