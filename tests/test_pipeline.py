"""Tests for critical LoRA pipeline logic: promotion gates, scoring, dataset families, schemas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_adapter import BASE_SCORES, check_promotion
from merge_dataset import _classify_family, DATASET_FAMILIES
from scoring import (
    score_correctness,
    score_convention,
    score_tool_selection,
    score_code_presence,
    score_task,
    score_failure_penalty,
    CATEGORY_CONVENTIONS,
)
from schemas import (
    TrainingExample,
    ChatExample,
    Message,
    Role,
    ToolCall,
    FunctionCall,
    Tool,
    ToolFunction,
    ProvenanceMetadata,
    RightsBasis,
)
from curate_dataset import count_code_lines
from presplit_sequences import estimate_tokens


# ---------------------------------------------------------------------------
# 1. Promotion gate tests
# ---------------------------------------------------------------------------


def _registry(prod_overall: float = 80.0, minimum: float = 76.9, threshold: float = 2.0):
    return {
        "minimum_score": minimum,
        "regression_threshold": threshold,
        "production": {"adapter": "old-v1", "overall_score": prod_overall},
    }


def test_promotion_rejects_below_minimum():
    """Adapter scoring below minimum_score (76.9%) must be rejected."""
    scores = {k: 60.0 for k in BASE_SCORES}
    passed, reasons = check_promotion(scores, _registry(prod_overall=50.0))
    assert not passed
    assert any("below minimum" in r for r in reasons)


def test_promotion_rejects_regression():
    """Adapter where any category drops >threshold below BASE must be rejected."""
    scores = {k: 90.0 for k in BASE_SCORES}
    scores["coding"] = BASE_SCORES["coding"] - 5.0  # well below threshold of 2.0
    passed, reasons = check_promotion(scores, _registry(prod_overall=85.0))
    assert not passed
    assert any("coding" in r and "regressed" in r for r in reasons)


def test_promotion_accepts_improvement():
    """Adapter beating production on overall with no regressions must be accepted."""
    scores = {k: v + 5.0 for k, v in BASE_SCORES.items()}
    passed, reasons = check_promotion(scores, _registry(prod_overall=78.0))
    assert passed
    assert reasons == []


def test_promotion_rejects_tie():
    """Same score as production does not promote."""
    scores = {k: v + 5.0 for k, v in BASE_SCORES.items()}
    scores["overall"] = 80.0
    passed, reasons = check_promotion(scores, _registry(prod_overall=80.0))
    assert not passed
    assert any("does not beat" in r for r in reasons)


def test_promotion_force_overrides():
    """--force flag promotes even when checks fail (tested at gate level; force is CLI-only)."""
    scores = {k: 50.0 for k in BASE_SCORES}
    passed, reasons = check_promotion(scores, _registry(prod_overall=80.0))
    # check_promotion itself always returns the truth; --force is handled by main()
    assert not passed
    assert len(reasons) > 0


def test_promotion_category_at_exact_threshold():
    """Category at exactly base - threshold should pass (not reject)."""
    scores = {k: v + 5.0 for k, v in BASE_SCORES.items()}
    # Set coding to exactly base - threshold (the floor value)
    scores["coding"] = BASE_SCORES["coding"] - 2.0  # exactly at floor
    passed, reasons = check_promotion(scores, _registry(prod_overall=78.0))
    # At exactly the floor, new_val < floor is False, so it should pass
    assert passed
    assert reasons == []


def test_promotion_empty_results():
    """Empty scores dict should be rejected (below minimum)."""
    scores: dict[str, float] = {}
    passed, reasons = check_promotion(scores, _registry(prod_overall=80.0))
    assert not passed


def test_promotion_missing_category():
    """Results missing a category should still work (skip missing in regression)."""
    scores = {k: v + 5.0 for k, v in BASE_SCORES.items()}
    del scores["embodiment"]
    passed, reasons = check_promotion(scores, _registry(prod_overall=78.0))
    # Missing categories are skipped in regression check (new_val is None -> continue)
    assert passed


# ---------------------------------------------------------------------------
# 2. Scoring function tests
# ---------------------------------------------------------------------------


def test_score_correctness_all_patterns():
    """All expected patterns found -> 1.0"""
    assert score_correctness("foo bar baz", ["foo", "bar", "baz"]) == 1.0


def test_score_correctness_no_patterns():
    """No patterns found -> 0.0"""
    assert score_correctness("nothing here", ["foo", "bar"]) == 0.0


def test_score_correctness_partial():
    """Half patterns found -> 0.5"""
    assert score_correctness("foo only", ["foo", "bar"]) == 0.5


def test_score_correctness_regex_special_chars():
    """Patterns with regex special chars don't crash."""
    # These are valid regex patterns with special chars
    result = score_correctness("Init() and Start()", [r":Init\(\)", r":Start\(\)"])
    assert isinstance(result, float)
    # Also test with patterns that could be problematic
    result2 = score_correctness("test [bracket]", [r"\[bracket\]"])
    assert result2 == 1.0


def test_score_correctness_empty_response():
    """Empty response scores 0."""
    assert score_correctness("", ["foo", "bar"]) == 0.0


def test_score_correctness_empty_patterns():
    """Empty pattern list scores 1.0 (vacuous truth)."""
    assert score_correctness("anything", []) == 1.0


def test_score_convention_detects_strict():
    """--!strict in response gets credit."""
    result = score_convention("--!strict\nlocal M = {}", "coding")
    assert result is not None
    assert result > 0.0


def test_score_convention_skips_embodiment():
    """Embodiment tasks skip convention scoring."""
    result = score_convention("--!strict\nlocal M = {}", "embodiment")
    assert result is None


def test_score_convention_all_patterns():
    """Response with all coding convention patterns scores 1.0."""
    response = "--!strict\n@native\nfunction M:Init()\nend\nfunction M:Start()\nend"
    result = score_convention(response, "coding")
    assert result == 1.0


def test_score_convention_coding_category():
    """Coding tasks check strict_mode, native, init, start patterns."""
    expected_keys = CATEGORY_CONVENTIONS["coding"]
    assert "strict_mode" in expected_keys
    assert "native_annotation" in expected_keys
    assert "init_lifecycle" in expected_keys
    assert "start_lifecycle" in expected_keys


def test_score_convention_mcp_returns_none():
    """MCP tasks return None for convention (not applicable)."""
    result = score_convention("anything here", "mcp_tool_calling")
    assert result is None


def test_score_convention_bugfix_only_strict():
    """Bugfix category only checks strict_mode."""
    result = score_convention("--!strict\nlocal x = 1", "bugfix")
    assert result == 1.0
    result2 = score_convention("local x = 1", "bugfix")
    assert result2 == 0.0


def test_score_tool_selection_with_real_tools():
    """Response mentioning run_code gets credit for run_code tool tasks."""
    task = {"tools": [{"function": {"name": "run_code"}}]}
    result = score_tool_selection("I will use run_code to execute this", task)
    assert result == 1.0


def test_score_tool_selection_no_tools():
    """Non-tool task returns None."""
    task = {"tools": None}
    result = score_tool_selection("any response", task)
    assert result is None
    # Also test missing tools key
    result2 = score_tool_selection("any response", {})
    assert result2 is None


def test_score_tool_selection_mentions_tool():
    """Response mentioning the tool name gets credit."""
    task = {
        "tools": [
            {"function": {"name": "run_code"}},
            {"function": {"name": "get_console_output"}},
        ]
    }
    result = score_tool_selection("I used run_code and get_console_output", task)
    assert result == 1.0


def test_score_tool_selection_partial():
    """Response mentioning only some tools gets partial credit."""
    task = {
        "tools": [
            {"function": {"name": "run_code"}},
            {"function": {"name": "insert_model"}},
        ]
    }
    result = score_tool_selection("I used run_code to do it", task)
    assert result == 0.5


def test_score_code_presence_with_luau_block():
    """Response with ```luau block scores 1.0."""
    response = "Here is code:\n```luau\nlocal x = 1\n```"
    result = score_code_presence(response, "coding")
    assert result == 1.0


def test_score_code_presence_prose_only():
    """Response without code block scores 0.0."""
    result = score_code_presence("Just some prose without any code.", "coding")
    assert result == 0.0


def test_score_code_presence_non_code_category():
    """Non-code category returns None."""
    result = score_code_presence("anything", "embodiment")
    assert result is None


def test_score_code_presence_luau_keywords():
    """Luau keywords without backtick blocks still detected."""
    result = score_code_presence("local MyService = {}\nfunction MyService:Init()\nend", "coding")
    assert result == 1.0


def test_score_failure_penalty_empty():
    """Empty/short response gets 0.0 penalty."""
    assert score_failure_penalty("") == 0.0
    assert score_failure_penalty("short") == 0.0


def test_score_failure_penalty_error():
    """Error responses get 0.0."""
    assert score_failure_penalty("[TIMEOUT] something") == 0.0
    assert score_failure_penalty("[ERROR] something went wrong") == 0.0


def test_score_failure_penalty_medium():
    """Medium response (10-50 chars) gets 0.3."""
    assert score_failure_penalty("a" * 30) == 0.3


def test_score_failure_penalty_normal():
    """Normal length response gets 1.0."""
    assert score_failure_penalty("a" * 100) == 1.0


def test_score_task_composite_weights():
    """Verify overall score is weighted correctly."""
    task = {
        "category": "coding",
        "expected_patterns": ["--!strict"],
        "tools": None,
    }
    response = "--!strict\nlocal M = {}\nfunction M:Init()\nend\nfunction M:Start()\nend"
    result = score_task(response, task)
    assert "correctness" in result
    assert "overall" in result
    assert "applicable_dimensions" in result
    assert "correctness" in result["applicable_dimensions"]
    assert 0 <= result["overall"] <= 1


def test_score_task_tool_task():
    """Tool task includes tool_selection dimension."""
    task = {
        "category": "mcp_tool_calling",
        "expected_patterns": ["run_code"],
        "tools": [{"function": {"name": "run_code"}}],
    }
    response = "I will use run_code to execute this code block"
    result = score_task(response, task)
    assert result["tool_selection"] == 1.0
    assert "tool_selection" in result["applicable_dimensions"]


# ---------------------------------------------------------------------------
# 3. Dataset family classification tests
# ---------------------------------------------------------------------------


def test_classify_sft_files():
    """codebase*, oss_roblox -> sft family"""
    assert _classify_family("codebase_services.jsonl") == "sft"
    assert _classify_family("oss_roblox_libs.jsonl") == "sft"
    assert _classify_family("evolved_hard.jsonl") == "sft"


def test_classify_trajectory_files():
    """mcp_*, gameplay_*, studio_trajectory* -> trajectory family"""
    assert _classify_family("mcp_traces.jsonl") == "trajectory"
    assert _classify_family("gameplay_sessions.jsonl") == "trajectory"
    assert _classify_family("studio_trajectory_v2.jsonl") == "trajectory"


def test_classify_critic_files():
    """bugfix_*, critic_* -> critic family"""
    assert _classify_family("bugfix_pairs.jsonl") == "critic"
    assert _classify_family("critic_rewrites.jsonl") == "critic"
    assert _classify_family("preference_data.jsonl") == "critic"


def test_classify_unknown_file():
    """Unknown filename -> None"""
    assert _classify_family("totally_random_name.jsonl") is None


def test_classify_additional_sft_patterns():
    """Additional SFT family patterns are classified correctly."""
    assert _classify_family("api_docs_v2.jsonl") == "sft"
    assert _classify_family("devforum_qa.jsonl") == "sft"
    assert _classify_family("synthetic_hard.jsonl") == "sft"
    assert _classify_family("roblox_creator_guides.jsonl") == "sft"
    assert _classify_family("luau_stdlib.jsonl") == "sft"
    assert _classify_family("rojo_ecosystem.jsonl") == "sft"


def test_classify_additional_trajectory_patterns():
    """Additional trajectory patterns are classified correctly."""
    assert _classify_family("embodiment_data.jsonl") == "trajectory"


def test_classify_family_with_path():
    """Classification works with full paths too."""
    assert _classify_family("/data/raw/codebase_services.jsonl") == "sft"
    assert _classify_family("some/dir/bugfix_pairs.jsonl") == "critic"


def test_dataset_families_ratios_sum_to_one():
    """Target ratios across all families should sum to 1.0."""
    total = sum(cfg["target_ratio"] for cfg in DATASET_FAMILIES.values())
    assert abs(total - 1.0) < 0.001


# ---------------------------------------------------------------------------
# 4. Schema validation tests
# ---------------------------------------------------------------------------


def _msg(role: str, content: str = "hello"):
    return Message(role=Role(role), content=content)


def test_training_example_valid():
    """Valid TrainingExample passes validation."""
    ex = TrainingExample(messages=[_msg("user"), _msg("assistant", "response")])
    assert len(ex.messages) == 2


def test_training_example_rejects_bad_difficulty():
    """Difficulty > 5 rejected."""
    with pytest.raises(Exception):
        TrainingExample(
            messages=[_msg("user"), _msg("assistant")],
            difficulty=6,
        )


def test_message_allows_tool_calls():
    """Assistant message with tool_calls is valid."""
    tc = ToolCall(id="call_1", function=FunctionCall(name="run_code", arguments="{}"))
    msg = Message(role=Role.assistant, content=None, tool_calls=[tc])
    assert msg.tool_calls[0].function.name == "run_code"


def test_provenance_metadata_optional():
    """TrainingExample without provenance is valid."""
    ex = TrainingExample(messages=[_msg("user"), _msg("assistant")])
    assert ex.provenance is None


def test_training_example_difficulty_zero_rejected():
    """Difficulty below 1 is rejected."""
    with pytest.raises(ValidationError):
        TrainingExample(
            messages=[_msg("user"), _msg("assistant")],
            difficulty=0,
        )


def test_training_example_negative_difficulty_rejected():
    """Negative difficulty is rejected."""
    with pytest.raises(ValidationError):
        TrainingExample(
            messages=[_msg("user"), _msg("assistant")],
            difficulty=-1,
        )


def test_chat_example_requires_assistant():
    """ChatExample without an assistant message is rejected."""
    with pytest.raises(ValidationError):
        ChatExample(messages=[_msg("user"), _msg("user", "again")])


def test_chat_example_minimum_messages():
    """ChatExample requires at least 2 messages."""
    with pytest.raises(ValidationError):
        ChatExample(messages=[_msg("assistant")])


def test_chat_example_valid():
    """Valid ChatExample passes."""
    ex = ChatExample(messages=[_msg("user"), _msg("assistant", "response")])
    assert len(ex.messages) == 2


def test_provenance_with_rights_basis():
    """ProvenanceMetadata accepts valid rights basis."""
    p = ProvenanceMetadata(rights_basis=RightsBasis.open_source, license="MIT")
    assert p.rights_basis == RightsBasis.open_source


def test_tool_definition_valid():
    """Tool with function definition passes validation."""
    t = Tool(
        function=ToolFunction(
            name="run_code",
            description="Execute Luau code",
            parameters={"type": "object", "properties": {}},
        )
    )
    assert t.function.name == "run_code"
    assert t.type == "function"


def test_training_example_with_tools():
    """TrainingExample with tools is valid."""
    tool = Tool(
        function=ToolFunction(
            name="run_code",
            description="Execute code",
            parameters={"type": "object"},
        )
    )
    tc = ToolCall(id="call_1", function=FunctionCall(name="run_code", arguments='{"code": "print(1)"}'))
    ex = TrainingExample(
        messages=[
            _msg("user", "run some code"),
            Message(role=Role.assistant, content=None, tool_calls=[tc]),
        ],
        tools=[tool],
    )
    assert ex.tools is not None
    assert len(ex.tools) == 1


def test_all_roles_valid():
    """All Role enum values create valid messages."""
    for role in Role:
        msg = Message(role=role, content="test")
        assert msg.role == role


# ---------------------------------------------------------------------------
# 5. Curation logic tests
# ---------------------------------------------------------------------------


def test_count_code_lines_empty():
    """No code blocks -> 0 lines."""
    assert count_code_lines("just some prose text") == 0


def test_count_code_lines_luau_block():
    """```luau block with lines -> correct count."""
    code = "```luau\n" + "\n".join(f"local x{i} = {i}" for i in range(10)) + "\n```"
    assert count_code_lines(code) == 10


def test_count_code_lines_nested():
    """Multiple code blocks sum correctly."""
    text = (
        "First block:\n```lua\nlocal a = 1\nlocal b = 2\n```\n"
        "Second block:\n```luau\nlocal c = 3\nlocal d = 4\nlocal e = 5\n```"
    )
    result = count_code_lines(text)
    assert result == 5  # 2 + 3


def test_count_code_lines_skips_lang_tag():
    """Language tag line is not counted as code."""
    text = "```python\nprint('hello')\n```"
    # "python" doesn't start with --, local, {, [, return, function
    # so it's skipped as a lang tag
    result = count_code_lines(text)
    assert result == 1


def test_count_code_lines_empty_block():
    """Empty code block -> 0."""
    assert count_code_lines("```\n```") == 0


# ---------------------------------------------------------------------------
# 6. Presplit logic tests
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty():
    """Empty string -> 0 tokens."""
    assert estimate_tokens([]) == 0
    assert estimate_tokens([{"role": "user", "content": ""}]) == 0


def test_estimate_tokens_scales():
    """Longer text -> more tokens."""
    short = estimate_tokens([{"role": "user", "content": "hello"}])
    long = estimate_tokens([{"role": "user", "content": "hello " * 100}])
    assert long > short


def test_estimate_tokens_multiple_messages():
    """Token count aggregates across messages."""
    single = estimate_tokens([{"role": "user", "content": "hello world"}])
    double = estimate_tokens(
        [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hello world"},
        ]
    )
    assert double > single


def test_estimate_tokens_none_content():
    """Messages with None content don't crash."""
    result = estimate_tokens([{"role": "assistant"}])
    assert result == 0


# ---------------------------------------------------------------------------
# 7. Property-based fuzzing with Hypothesis
# ---------------------------------------------------------------------------


@given(st.text(min_size=0, max_size=1000))
@settings(max_examples=200)
def test_score_correctness_never_crashes(response):
    """score_correctness should never raise, regardless of input."""
    result = score_correctness(response, ["pattern1", "pattern2"])
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=200)
def test_score_convention_never_crashes(response):
    """score_convention should never raise, regardless of input."""
    for cat in ["coding", "bugfix", "architecture", "mcp_tool_calling", "embodiment"]:
        result = score_convention(response, cat)
        assert result is None or (0.0 <= result <= 1.0)


@given(st.integers(min_value=-100, max_value=100))
def test_difficulty_boundary(difficulty):
    """Difficulty outside 1-5 should be rejected."""
    try:
        TrainingExample(
            messages=[_msg("user"), _msg("assistant")],
            difficulty=difficulty,
        )
        assert 1 <= difficulty <= 5
    except ValidationError:
        assert difficulty < 1 or difficulty > 5


@given(st.text(min_size=0, max_size=2000))
@settings(max_examples=100)
def test_score_task_never_crashes(response):
    """score_task should handle any response string without crashing."""
    task = {"category": "coding", "expected_patterns": ["test"], "tools": None}
    result = score_task(response, task)
    assert isinstance(result, dict)
    assert "overall" in result
    assert 0 <= result["overall"] <= 1


@given(
    st.sampled_from(
        [
            "codebase.jsonl",
            "codebase_controllers.jsonl",
            "oss_roblox.jsonl",
            "mcp_tools.jsonl",
            "gameplay_sessions.jsonl",
            "studio_trajectory.jsonl",
            "bugfix_pairs.jsonl",
            "critic_repair.jsonl",
            "unknown_file.jsonl",
            "evolved.jsonl",
            "roblox_creator_docs.jsonl",
        ]
    )
)
def test_classify_family_deterministic(filename):
    """Family classification should be deterministic."""
    result1 = _classify_family(filename)
    result2 = _classify_family(filename)
    assert result1 == result2


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=100)
def test_score_code_presence_never_crashes(response):
    """score_code_presence should never raise, regardless of input."""
    for cat in ["coding", "bugfix", "architecture", "mcp_tool_calling", "embodiment", None]:
        result = score_code_presence(response, cat)
        assert result is None or (0.0 <= result <= 1.0)


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=100)
def test_score_failure_penalty_never_crashes(response):
    """score_failure_penalty should never raise."""
    result = score_failure_penalty(response)
    assert 0.0 <= result <= 1.0


@given(st.text(min_size=0, max_size=1000))
@settings(max_examples=100)
def test_count_code_lines_never_crashes(text):
    """count_code_lines should never raise."""
    result = count_code_lines(text)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# 8. Integration test
# ---------------------------------------------------------------------------


def test_full_scoring_pipeline():
    """End-to-end: create a benchmark task, score a real-looking response."""
    task = {
        "id": "test_01",
        "category": "coding",
        "difficulty": 3,
        "prompt": "Write a service",
        "expected_patterns": ["--!strict", r":Init\(\)", "return"],
        "tools": None,
    }
    response = """--!strict
local MyService = {}
function MyService:Init()
    self._data = {}
end
function MyService:Start()
    print("started")
end
return MyService"""
    result = score_task(response, task)
    assert result["correctness"] == 1.0
    assert result["overall"] > 0.5
    assert result["failure_penalty"] == 1.0
    assert result["code_presence"] == 1.0


def test_full_scoring_pipeline_empty_response():
    """Empty response should score very low overall."""
    task = {
        "category": "coding",
        "expected_patterns": ["--!strict", "return"],
        "tools": None,
    }
    result = score_task("", task)
    assert result["correctness"] == 0.0
    assert result["overall"] == 0.0
    assert result["failure_penalty"] == 0.0


def test_full_scoring_pipeline_tool_task():
    """Tool-calling task scores all dimensions."""
    task = {
        "category": "mcp_tool_calling",
        "expected_patterns": ["run_code"],
        "tools": [{"function": {"name": "run_code"}}],
    }
    response = "I will use run_code to execute the Luau script:\n```lua\nlocal x = 1\n```"
    result = score_task(response, task)
    assert result["correctness"] == 1.0
    assert result["tool_selection"] == 1.0
    assert result["overall"] > 0.5


# ---------------------------------------------------------------------------
# 9. Merge dataset integration test
# ---------------------------------------------------------------------------

import json as _json
import subprocess


def test_merge_dataset_integration(tmp_path):
    """End-to-end: create fixtures, run merge, verify output format."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"

    # Create 5 minimal valid JSONL examples in a file matching "codebase*" pattern
    fixture_file = raw_dir / "codebase_test.jsonl"
    examples = []
    for i in range(5):
        ex = {
            "messages": [
                {"role": "system", "content": "You are a Roblox Luau coding assistant."},
                {"role": "user", "content": f"Write a service that tracks player count (variant {i})."},
                {
                    "role": "assistant",
                    "content": (
                        f"--!strict\n"
                        f"local PlayerTracker{i} = {{}}\n"
                        f"function PlayerTracker{i}:Init()\n"
                        f"    self._count = 0\n"
                        f"end\n"
                        f"function PlayerTracker{i}:Start()\n"
                        f'    game:GetService("Players").PlayerAdded:Connect(function()\n'
                        f"        self._count += 1\n"
                        f"    end)\n"
                        f"end\n"
                        f"return PlayerTracker{i}"
                    ),
                },
            ],
            "category": "coding",
            "source_file": "codebase_test.jsonl",
        }
        examples.append(ex)

    with open(fixture_file, "w") as f:
        for ex in examples:
            f.write(_json.dumps(ex) + "\n")

    # Run merge_dataset.py with --raw-dir and --out-dir
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    result = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "merge_dataset.py"),
            "--raw-dir",
            str(raw_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"merge_dataset.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Check output files exist
    for split_name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        split_path = out_dir / split_name
        assert split_path.exists(), f"{split_name} not created"

    # Check all output lines are valid JSON with correct MLX format
    total_lines = 0
    for split_name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        split_path = out_dir / split_name
        with open(split_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                assert "messages" in row, f"{split_name}:{line_num} missing 'messages' key"
                assert isinstance(row["messages"], list), f"{split_name}:{line_num} 'messages' is not a list"
                assert len(row["messages"]) >= 2, f"{split_name}:{line_num} too few messages"
                for msg in row["messages"]:
                    assert isinstance(msg, dict), f"{split_name}:{line_num} message is not a dict"
                    assert "role" in msg, f"{split_name}:{line_num} message missing 'role'"
                    assert "content" in msg, f"{split_name}:{line_num} message missing 'content'"
                total_lines += 1

    # With 5 examples, we should get at least 1 line across all splits
    assert total_lines >= 1, f"Expected at least 1 output line, got {total_lines}"
