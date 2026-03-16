"""
Pydantic schemas for validating all training data.

Every example passes through these schemas before entering the dataset.
Invalid examples are logged and rejected — quality over quantity.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class RightsBasis(str, Enum):
    open_source = "open_source"
    own_codebase = "own_codebase"
    permissioned = "permissioned"
    generated = "generated"
    public_domain = "public_domain"


class TaskFamily(str, Enum):
    sft_builder = "sft_builder"  # Dataset A: core builder/scripter
    sft_scripter = "sft_scripter"  # Dataset A: core builder/scripter
    trajectory = "trajectory"  # Dataset B: studio action traces
    playtest = "playtest"  # Dataset B: playtest traces
    critic = "critic"  # Dataset C: critique/repair
    preference = "preference"  # Dataset C: pairwise preference
    eval_task = "eval_task"  # Dataset C: eval/benchmark


class Modality(str, Enum):
    code = "code"
    code_with_hierarchy = "code_with_hierarchy"
    tool_trajectory = "tool_trajectory"
    observation_action = "observation_action"
    critique_rewrite = "critique_rewrite"
    pairwise = "pairwise"


class DataSource(str, Enum):
    vertigo_codebase = "vertigo_codebase"
    roblox_api_docs = "roblox_api_docs"
    mcp_tool_calling = "mcp_tool_calling"
    devforum_qa = "devforum_qa"
    synthetic_evolved = "synthetic_evolved"
    bug_fix_pairs = "bug_fix_pairs"
    refactoring = "refactoring"
    roblox_creator_docs = "roblox_creator_docs"
    luau_repo = "luau_repo"
    rojo_ecosystem = "rojo_ecosystem"
    open_game_eval = "open_game_eval"
    studio_trajectory = "studio_trajectory"
    teacher_distillation = "teacher_distillation"
    critic_repair = "critic_repair"
    gameplay_session = "gameplay_session"


class Category(str, Enum):
    service = "service"
    controller = "controller"
    config = "config"
    builder = "builder"
    physics = "physics"
    types = "types"
    networking = "networking"
    mcp_tool_call = "mcp_tool_call"
    mcp_multi_step = "mcp_multi_step"
    api_usage = "api_usage"
    debugging = "debugging"
    refactoring = "refactoring"
    general_luau = "general_luau"
    trajectory = "trajectory"
    playtest = "playtest"
    critic = "critic"
    preference = "preference"
    scene_building = "scene_building"


# ---------------------------------------------------------------------------
# Message schemas (matching MLX chat + tools format)
# ---------------------------------------------------------------------------


class ToolFunction(BaseModel):
    """A single tool/function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class Tool(BaseModel):
    """Tool wrapper matching OpenAI format."""

    type: str = "function"
    function: ToolFunction


class FunctionCall(BaseModel):
    """A function call within a tool_call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    id: str
    type: str = "function"
    function: FunctionCall


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def content_or_tool_calls(cls, v, info):
        """Assistant messages may have content, tool_calls, or both."""
        return v


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------


class ProvenanceMetadata(BaseModel):
    """Rights-clean provenance tracking for every training example."""

    source_id: str | None = None  # unique source identifier
    rights_basis: RightsBasis | None = None
    license: str | None = None  # e.g. "MIT", "MPL-2.0", "proprietary-own"
    consent_id: str | None = None  # for permissioned data
    creator_id_hash: str | None = None  # hashed creator identifier
    project_id: str | None = None  # which project/repo
    task_family: TaskFamily | None = None
    modality: Modality | None = None
    scene_hash: str | None = None  # hash of scene/hierarchy context
    ast_hash: str | None = None  # normalized Luau AST hash for dedup
    eval_score: float | None = None  # auto-grading score if available
    pii_flag: bool = False
    teacher_model: str | None = None  # if generated, which model
    human_reviewed: bool = False
    freshness_date: str | None = None  # ISO date of source data


# ---------------------------------------------------------------------------
# Training example schemas
# ---------------------------------------------------------------------------


class ChatExample(BaseModel):
    """Standard chat training example (MLX chat format)."""

    messages: list[Message] = Field(min_length=2)

    @field_validator("messages")
    @classmethod
    def must_have_assistant(cls, v):
        if not any(m.role == Role.assistant for m in v):
            raise ValueError("Must contain at least one assistant message")
        return v


class ToolExample(BaseModel):
    """Tool-calling training example (MLX tools format)."""

    tools: list[Tool] = Field(min_length=1)
    messages: list[Message] = Field(min_length=2)


class TrainingExample(BaseModel):
    """Full training example with metadata (metadata stripped before MLX)."""

    messages: list[Message]
    tools: list[Tool] | None = None

    # Metadata (not passed to MLX)
    source: DataSource | None = None
    category: Category | None = None
    file_path: str | None = None
    difficulty: int | None = Field(None, ge=1, le=5)
    has_reasoning: bool = False
    verified: bool = False  # True if Luau code was verified by CLI
    provenance: ProvenanceMetadata | None = None


class MLXChatRow(BaseModel):
    """Final format written to JSONL for MLX (chat examples)."""

    messages: list[dict[str, Any]]


class MLXToolRow(BaseModel):
    """Final format written to JSONL for MLX (tool examples)."""

    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
