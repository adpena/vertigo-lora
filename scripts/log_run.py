#!/usr/bin/env python3
from __future__ import annotations

"""Append a training run summary to docs/training-log.md.

Called automatically by auto_train.sh after every run. Captures:
- adapter name, config, model, timestamp
- val loss from adapter_config.json
- per-category benchmark scores
- promotion decision (promoted / rejected + reason)
- hardware (hostname, peak mem if available)

Entries are appended to a structured table at the end of the log.
"""

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOG_FILE = PROJECT_DIR / "docs" / "training-log.md"
RUN_LOG_SECTION = "\n## Run History\n"
TABLE_HEADER = (
    "| Date | Adapter | Model | Config | Val Loss | "
    "Coding | Bugfix | Arch | MCP | Embodiment | Overall | Promoted | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def extract_val_loss(adapter_dir: Path) -> str:
    """Get final val loss from adapter_config.json or training output."""
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        try:
            json.loads(config_path.read_text())
            # adapter_config.json doesn't store val_loss directly
            # Check for a training_log.jsonl or similar
        except Exception:
            pass

    # Try to find val loss from the most recent results
    return "—"


def extract_scores(results_path: Path) -> dict:
    """Extract per-category scores from benchmark results JSONL."""
    if not results_path.exists():
        return {}

    from collections import defaultdict

    cats: dict[str, list[float]] = defaultdict(list)
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            cats[r["category"]].append(r["scores"]["overall"])
        except (json.JSONDecodeError, KeyError):
            continue

    scores = {}
    for cat, vals in cats.items():
        scores[cat] = round(sum(vals) / len(vals) * 100, 1)
    if scores:
        all_vals = [v for vals in cats.values() for v in vals]
        scores["overall"] = round(sum(all_vals) / len(all_vals) * 100, 1)
    return scores


def extract_promotion(registry_path: Path, adapter_name: str) -> tuple[bool, str]:
    """Check if adapter was promoted or rejected."""
    if not registry_path.exists():
        return False, "no registry"

    reg = json.loads(registry_path.read_text())
    if reg.get("production", {}).get("adapter") == adapter_name:
        return True, "production"

    for entry in reg.get("history", []):
        if entry.get("adapter") == adapter_name:
            return True, f"demoted: {entry.get('reason', '?')}"

    return False, "rejected"


def format_row(
    adapter: str,
    config: str,
    results_path: Path,
    registry_path: Path,
) -> str:
    """Format a single log table row."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    host = socket.gethostname().split(".")[0]

    # Extract model from config
    config_path = Path(config)
    model = "?"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            if line.startswith("model:"):
                model = line.split(":")[1].strip().strip('"').split("/")[-1]
                break

    config_name = config_path.stem

    # Scores
    scores = extract_scores(results_path)
    coding = f"{scores.get('coding', 0):.1f}%" if scores else "—"
    bugfix = f"{scores.get('bugfix', 0):.1f}%" if scores else "—"
    arch = f"{scores.get('architecture', 0):.1f}%" if scores else "—"
    mcp = f"{scores.get('mcp_tool_calling', 0):.1f}%" if scores else "—"
    embodiment = f"{scores.get('embodiment', 0):.1f}%" if scores else "—"
    overall = f"**{scores.get('overall', 0):.1f}%**" if scores else "—"

    # Val loss
    adapter_dir = PROJECT_DIR / "adapters" / adapter
    val_loss = extract_val_loss(adapter_dir)

    # Promotion
    promoted, reason = extract_promotion(registry_path, adapter)
    prom_str = f"{'yes' if promoted else 'no'} ({reason})"

    return (
        f"| {now} | {adapter} | {model} | {config_name} | {val_loss} | "
        f"{coding} | {bugfix} | {arch} | {mcp} | {embodiment} | {overall} | "
        f"{prom_str} | {host} |\n"
    )


def append_to_log(row: str) -> None:
    """Append a row to the training log's Run History table."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        LOG_FILE.write_text(f"# Training Log\n{RUN_LOG_SECTION}{TABLE_HEADER}{row}")
        return

    content = LOG_FILE.read_text()

    if RUN_LOG_SECTION.strip() not in content:
        # Add the section at the end
        content += f"\n{RUN_LOG_SECTION}{TABLE_HEADER}{row}"
    elif TABLE_HEADER.splitlines()[0] not in content:
        # Section exists but no table — add table
        idx = content.index(RUN_LOG_SECTION.strip()) + len(RUN_LOG_SECTION.strip())
        content = content[:idx] + f"\n\n{TABLE_HEADER}{row}" + content[idx:]
    else:
        # Table exists — append row
        content = content.rstrip() + "\n" + row

    LOG_FILE.write_text(content)


def main() -> None:
    ap = argparse.ArgumentParser(description="Log a training run to docs/training-log.md")
    ap.add_argument("--adapter", required=True, help="Adapter name (directory under adapters/)")
    ap.add_argument("--config", required=True, help="Config file used for training")
    ap.add_argument("--results", required=True, help="Benchmark results JSONL")
    ap.add_argument("--registry", default="adapters/REGISTRY.json", help="Adapter registry")
    args = ap.parse_args()

    row = format_row(
        adapter=args.adapter,
        config=args.config,
        results_path=Path(args.results),
        registry_path=Path(args.registry),
    )

    append_to_log(row)
    print(f"  Logged run: {args.adapter} → docs/training-log.md")


if __name__ == "__main__":
    main()
