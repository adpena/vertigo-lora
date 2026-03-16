#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark Runner — evaluate LoRA adapter quality against structured tasks.

Loads a benchmark JSONL file, runs each task through the model + adapter,
scores responses on correctness, convention adherence, and tool selection,
then prints a formatted report and saves detailed results.

Usage:
    python3 scripts/run_benchmark.py --adapter adapters/my_adapter
    python3 scripts/run_benchmark.py --adapter adapters/v1 --categories coding bugfix
    python3 scripts/run_benchmark.py --compare data/eval/results_v1.jsonl data/eval/results_v2.jsonl
"""

import argparse
import json
import os
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from scoring import (
    VERTIGO_SYSTEM_PROMPT,
    score_task,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BENCHMARK = PROJECT_DIR / "data" / "eval" / "benchmark.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "eval" / "results.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
GENERATION_TIMEOUT_API = 120  # seconds per request for API mode

BENCHMARK_VERSION = "0.2.0"
SCORING_METHOD = "pattern_match_v2"

GENERATION_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Loading — scoring functions imported from scoring.py
# ---------------------------------------------------------------------------


def load_benchmark(path: Path) -> list[dict]:
    """Load benchmark tasks from JSONL."""
    tasks = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping line {i}: {e}", file=sys.stderr)
    return tasks


def resolve_output_path(path: Path) -> Path:
    """Never overwrite existing results — append timestamp if file exists."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.parent / f"{stem}_{ts}{suffix}"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_messages(task: dict) -> list[dict]:
    """Build chat messages for a benchmark task."""
    messages = [
        {"role": "system", "content": VERTIGO_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    return messages


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int = 2048,
    temp: float = 0.1,
) -> str:
    """Generate a response using mlx_lm. Raises on timeout."""
    from mlx_lm import generate

    # Build the prompt using the tokenizer's chat template
    prompt_kwargs = {"messages": messages, "add_generation_prompt": True}
    if tools:
        prompt_kwargs["tools"] = tools

    try:
        prompt = tokenizer.apply_chat_template(
            **prompt_kwargs,
            tokenize=False,
        )
    except Exception:
        # Fallback: simple concatenation if chat template fails
        prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
        prompt += "\n<|assistant|>\n"

    # Timeout handler
    def _timeout_handler(signum, frame):
        raise TimeoutError("Generation exceeded timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(GENERATION_TIMEOUT_SECONDS)

    # mlx_lm >= 0.31 uses a sampler callable instead of temp kwarg
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        make_sampler = None

    sampler_kwargs = {}
    if make_sampler is not None:
        sampler_kwargs["sampler"] = make_sampler(temp=temp)
    else:
        sampler_kwargs["temp"] = temp

    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            **sampler_kwargs,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return response


def generate_response_api(
    api_base: str,
    api_model: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int = 2048,
    temp: float = 0.1,
) -> str:
    """Generate a response via an OpenAI-compatible API (LM Studio, etc.)."""
    from api import call_llm

    # Inline tools into the system message rather than using the tools field,
    # since many API servers (LM Studio, etc.) don't support it reliably.
    if tools:
        tool_desc = json.dumps(tools, indent=2)
        messages = [dict(m) for m in messages]  # shallow copy
        messages[0] = dict(messages[0])
        messages[0]["content"] = (
            messages[0]["content"]
            + "\n\nYou have access to the following tools:\n```json\n"
            + tool_desc
            + "\n```\nCall tools by responding with a JSON tool_call block."
        )

    try:
        result = call_llm(
            messages,
            model=api_model,
            api_base=api_base,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=GENERATION_TIMEOUT_API,
        )
        choice = result["choices"][0]
        content = choice.get("message", {}).get("content", "")
        # Include tool calls in response text for scoring
        tool_calls = choice.get("message", {}).get("tool_calls")
        if tool_calls:
            content += "\n" + json.dumps(tool_calls)
        return content
    except Exception as e:
        raise ConnectionError(f"API request failed: {e}") from e


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_score(val: float | None) -> str:
    """Format a score value for table display: percentage or '-' for None."""
    if val is None:
        return "      -"
    return f"{val:>7.1%}"


def print_results_table(results: list[dict]):
    """Print formatted results table to stdout."""
    w = 96
    print(f"\n{'=' * w}")
    print(f"  Benchmark Results (scoring {SCORING_METHOD} v{BENCHMARK_VERSION})")
    print(f"{'=' * w}")

    header = (
        f"  {'ID':<12} {'Category':<18} {'Diff':>4} "
        f"{'Correct':>8} {'Conv':>8} {'Tool':>8} {'Code':>8} {'Pen':>5} {'Overall':>8}"
    )
    print(header)
    print(f"  {'-' * (w - 4)}")

    for r in results:
        scores = r["scores"]
        pen = scores.get("failure_penalty", 1.0)
        pen_str = f"{pen:.1f}" if pen < 1.0 else "  -"
        print(
            f"  {r['id']:<12} {r['category']:<18} {r['difficulty']:>4} "
            f"{_fmt_score(scores['correctness'])} {_fmt_score(scores.get('convention_score'))} "
            f"{_fmt_score(scores.get('tool_selection'))} {_fmt_score(scores.get('code_presence'))} "
            f"{pen_str:>5} {_fmt_score(scores['overall'])}"
        )


def print_category_summary(results: list[dict]):
    """Print summary stats per category."""
    from collections import defaultdict

    w = 96
    cats: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["scores"]["overall"])

    print(f"\n{'CATEGORY SUMMARY':─^{w}}")
    print(f"  {'Category':<20} {'Count':>6} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * (w - 4)}")

    for cat in sorted(cats.keys()):
        scores = cats[cat]
        mean = sum(scores) / len(scores)
        lo = min(scores)
        hi = max(scores)
        print(f"  {cat:<20} {len(scores):>6} {mean:>8.1%} {lo:>8.1%} {hi:>8.1%}")

    # Luau compile rate
    luau_results = [r["scores"].get("luau_compiles") for r in results]
    luau_attempted = [v for v in luau_results if v is not None]
    if luau_attempted:
        luau_pass = sum(1 for v in luau_attempted if v is True)
        luau_total = len(luau_attempted)
        pct = luau_pass / luau_total * 100 if luau_total else 0
        print(f"\n  Luau compile rate: {luau_pass}/{luau_total} ({pct:.0f}%)")

    # Overall
    all_scores = [r["scores"]["overall"] for r in results]
    mean = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n  {'OVERALL':─^{w - 4}}")
    print(f"  Total tasks: {len(all_scores)}   Mean overall: {mean:.1%}")
    print(f"{'=' * w}\n")


def print_comparison(paths: list[str]):
    """Compare results from multiple adapter runs side-by-side."""
    from collections import defaultdict

    runs: list[tuple[str, dict[str, dict]]] = []
    for p in paths:
        results_by_id: dict[str, dict] = {}
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results_by_id[r["id"]] = r
        runs.append((Path(p).stem, results_by_id))

    # Collect all task IDs
    all_ids = sorted({tid for _, rmap in runs for tid in rmap})

    w = 20 + len(runs) * 12
    print(f"\n{'=' * w}")
    print("  Adapter Comparison")
    print(f"{'=' * w}")

    header = f"  {'Task ID':<18}"
    for name, _ in runs:
        header += f" {name[:10]:>10}"
    print(header)
    print(f"  {'-' * (w - 4)}")

    cat_scores: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in runs])

    for tid in all_ids:
        row = f"  {tid:<18}"
        for i, (_, rmap) in enumerate(runs):
            r = rmap.get(tid)
            if r:
                score = r["scores"]["overall"]
                row += f" {score:>10.1%}"
                cat_scores[r.get("category", "?")][i].append(score)
            else:
                row += f" {'—':>10}"
        print(row)

    # Category means
    print(f"\n  {'Category Means':─^{w - 4}}")
    header = f"  {'Category':<18}"
    for name, _ in runs:
        header += f" {name[:10]:>10}"
    print(header)
    print(f"  {'-' * (w - 4)}")

    for cat in sorted(cat_scores.keys()):
        row = f"  {cat:<18}"
        for i in range(len(runs)):
            scores = cat_scores[cat][i]
            if scores:
                row += f" {sum(scores) / len(scores):>10.1%}"
            else:
                row += f" {'—':>10}"
        print(row)

    print(f"{'=' * w}\n")


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def atomic_write_jsonl(path: Path, results: list[dict]):
    """Write results atomically: write to .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".bench_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        # Clean up on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Vertigo LoRA Benchmark Runner")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to adapter directory (required unless --compare)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write detailed results JSONL",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="PATH",
        help="Compare multiple adapter result files (no generation)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default=None,
        metavar="URL",
        help=f"Use OpenAI-compatible API instead of local mlx_lm (default: {DEFAULT_API_BASE})",
        nargs="?",
        const=DEFAULT_API_BASE,
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="Model name to send in API requests (auto-detected from endpoint if omitted)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter to specific categories",
    )
    args = parser.parse_args()

    # Compare mode — no model needed
    if args.compare:
        print_comparison(args.compare)
        return

    use_api = args.api is not None

    if not use_api and not args.adapter:
        parser.error("--adapter is required (unless using --compare or --api)")

    adapter_path = Path(args.adapter) if args.adapter else None
    if adapter_path and not adapter_path.exists():
        print(f"ERROR: adapter path does not exist: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    # Load benchmark
    if not args.benchmark.exists():
        print(f"ERROR: benchmark file not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    tasks = load_benchmark(args.benchmark)
    if not tasks:
        print("ERROR: no tasks loaded from benchmark file", file=sys.stderr)
        sys.exit(1)

    # Filter categories
    if args.categories:
        cats = set(args.categories)
        tasks = [t for t in tasks if t.get("category") in cats]
        if not tasks:
            print(f"ERROR: no tasks match categories: {args.categories}", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(tasks)} benchmark tasks from {args.benchmark}")

    model = tokenizer = None
    model_label = ""

    if use_api:
        # API mode — use OpenAI-compatible endpoint
        api_base = args.api
        api_model = args.api_model
        if not api_model:
            # Auto-detect from /v1/models
            from api import detect_model as _detect_model

            try:
                api_model = _detect_model(api_base=api_base)
                print(f"Auto-detected API model: {api_model}")
            except Exception:
                api_model = "default"
        model_label = f"api:{api_model}"
        print(f"Using API: {api_base} model={api_model}")
    else:
        # Local mlx_lm mode
        model_name = args.model
        if adapter_path:
            adapter_config = adapter_path / "adapter_config.json"
            if adapter_config.exists():
                try:
                    cfg = json.loads(adapter_config.read_text())
                    detected_model = cfg.get("model")
                    if detected_model and model_name == DEFAULT_MODEL and detected_model != DEFAULT_MODEL:
                        model_name = detected_model
                        print(f"Auto-detected base model from adapter config: {model_name}")
                except Exception:
                    pass

        print(f"Loading model: {model_name}")
        if adapter_path:
            print(f"Loading adapter: {adapter_path}")
        try:
            from mlx_lm import load

            model, tokenizer = load(
                model_name,
                adapter_path=str(adapter_path) if adapter_path else None,
            )
        except ImportError:
            print("ERROR: mlx_lm not installed. Install with: uv pip install mlx-lm", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: failed to load model/adapter: {e}", file=sys.stderr)
            sys.exit(1)
        model_label = model_name
        if adapter_path:
            model_label += f"+{adapter_path.name}"

    print(f"Running benchmark ({model_label})...\n")

    # Run benchmark
    results = []
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        category = task.get("category", "unknown")
        difficulty = task.get("difficulty", 0)

        print(f"  [{i}/{len(tasks)}] {task_id} ({category}, d={difficulty}) ...", end=" ", flush=True)

        messages = build_messages(task)
        tools = task.get("tools")

        try:
            if use_api:
                response = generate_response_api(api_base, api_model, messages, tools)
            else:
                response = generate_response(model, tokenizer, messages, tools)
        except TimeoutError:
            response = "[TIMEOUT]"
            print("TIMEOUT")
        except Exception as e:
            response = f"[ERROR: {e}]"
            print(f"ERROR: {e}")
        else:
            scores = score_task(response, task)
            print(f"overall={scores['overall']:.1%}")

        scores = score_task(response, task)
        scores["benchmark_version"] = BENCHMARK_VERSION
        scores["scoring_method"] = SCORING_METHOD

        result = {
            "id": task_id,
            "category": category,
            "difficulty": difficulty,
            "prompt": task["prompt"],
            "response": response,
            "scores": scores,
            "adapter": str(adapter_path) if adapter_path else "none",
            "model": model_label,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

    # Print report
    print_results_table(results)
    print_category_summary(results)

    # Save results
    output_path = resolve_output_path(args.output)
    atomic_write_jsonl(output_path, results)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
