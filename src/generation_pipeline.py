"""Validation and persistence helpers for contrast-pair generation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


EXPECTED_RELATION = "positive_higher_than_negative"


class ValidationError(ValueError):
    """A model response does not satisfy the contrast-pair contract."""


def build_prompt_registry(prompts_root: Path) -> list[dict[str, Any]]:
    """Build the fixed general and metric-specific prompt registry."""
    definitions = [
        ("general_validation", "general_validation", "multiple"),
        ("size_compliance", "metric_specific", "Size Compliance"),
        ("intrachunk_cohesion", "metric_specific", "Intrachunk Cohesion"),
        ("contextual_coherence", "metric_specific", "Contextual Coherence"),
        ("boundary_clarity", "metric_specific", "Boundary Clarity"),
        ("chunk_score", "metric_specific", "ChunkScore"),
        ("hope_concept_unity", "metric_specific", "HOPE Concept Unity"),
        (
            "hope_semantic_independence",
            "metric_specific",
            "HOPE Semantic Independence",
        ),
        (
            "hope_information_preservation",
            "metric_specific",
            "HOPE Information Preservation",
        ),
    ]
    registry = []
    for prompt_name, dataset_type, target_metric in definitions:
        parent = prompts_root if dataset_type == "general_validation" else prompts_root / "metrics"
        registry.append(
            {
                "prompt_name": prompt_name,
                "dataset_type": dataset_type,
                "target_metric": target_metric,
                "path": parent / f"{prompt_name}.md",
            }
        )
    return registry


def select_prompts(
    registry: list[dict[str, Any]], names: list[str] | None
) -> list[dict[str, Any]]:
    """Select prompts by name while preserving registry order."""
    if names is None:
        return registry.copy()
    requested = set(names)
    known = {entry["prompt_name"] for entry in registry}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown prompt names: {', '.join(unknown)}")
    return [entry for entry in registry if entry["prompt_name"] in requested]


def load_prompt_text(
    prompts_root: Path, entry: dict[str, Any]
) -> tuple[str, str]:
    """Load the shared system prompt and one registered user prompt."""
    system_path = prompts_root / "system.md"
    missing = [str(path) for path in (system_path, entry["path"]) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prompt files: {', '.join(missing)}")
    return (
        system_path.read_text(encoding="utf-8"),
        entry["path"].read_text(encoding="utf-8"),
    )


def _normalized_text(parts: list[str] | str) -> str:
    text = "".join(parts) if isinstance(parts, list) else parts
    return re.sub(r"\|\||\s+", "", text)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def _require_indices(
    focus: dict[str, Any],
    key: str,
    upper: int,
    *,
    allow_empty: bool = False,
) -> list[int]:
    indices = focus.get(key)
    if not isinstance(indices, list) or (not indices and not allow_empty):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValidationError(f"focus.{key} must be a {qualifier} list")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= upper
        for index in indices
    ):
        raise ValidationError(f"focus.{key} contains an invalid zero-based index")
    return indices


def _validate_focus(prompt_name: str, variant: dict[str, Any]) -> None:
    focus = variant.get("focus")
    if not isinstance(focus, dict):
        raise ValidationError("focus must be an object")
    chunk_count = len(variant["chunks"])

    if prompt_name == "general_validation":
        _require_indices(focus, "target_chunk_indices", chunk_count)
        _require_indices(
            focus,
            "target_boundary_indices",
            max(chunk_count - 1, 0),
            allow_empty=True,
        )
    elif prompt_name in {
        "size_compliance",
        "intrachunk_cohesion",
        "contextual_coherence",
        "hope_concept_unity",
    }:
        _require_indices(focus, "target_chunk_indices", chunk_count)
    elif prompt_name == "boundary_clarity":
        _require_indices(focus, "target_boundary_indices", chunk_count - 1)
    elif prompt_name == "chunk_score":
        _require_indices(focus, "target_chunk_indices", chunk_count)
        _require_indices(focus, "target_boundary_indices", chunk_count - 1)
    elif prompt_name == "hope_semantic_independence":
        indices = _require_indices(
            focus, "dependent_chunk_indices", chunk_count
        )
        if len(set(indices)) < 2:
            raise ValidationError(
                "Semantic Independence requires at least two dependent chunks"
            )
        _require_string(focus.get("cue_question"), "focus.cue_question")
    elif prompt_name == "hope_information_preservation":
        _require_indices(focus, "affected_chunk_indices", chunk_count)
        _require_string(focus.get("fact"), "focus.fact")


def _validate_variants(payload: dict[str, Any], prompt_name: str) -> None:
    for variant_name in ("positive", "negative"):
        variant = payload.get(variant_name)
        if not isinstance(variant, dict):
            raise ValidationError(f"{variant_name} must be an object")
        chunks = variant.get("chunks")
        if (
            not isinstance(chunks, list)
            or not chunks
            or any(
                not isinstance(chunk, str) or not chunk.strip()
                for chunk in chunks
            )
        ):
            raise ValidationError(
                f"{variant_name}.chunks must contain non-empty strings"
            )
        _require_string(variant.get("rationale"), f"{variant_name}.rationale")
        _validate_focus(prompt_name, variant)

    if payload["positive"]["chunks"] == payload["negative"]["chunks"]:
        raise ValidationError("positive and negative chunks must differ")


def _validate_source_restoration(
    payload: dict[str, Any], prompt_name: str
) -> None:
    source = _normalized_text(payload["source_document"])
    variants = (
        ("positive",)
        if prompt_name == "hope_information_preservation"
        else ("positive", "negative")
    )
    for variant_name in variants:
        if _normalized_text(payload[variant_name]["chunks"]) != source:
            raise ValidationError(
                f"{variant_name} chunks do not restore source_document"
            )


def _validate_size_compliance(payload: dict[str, Any]) -> None:
    positive = payload["positive"]
    negative = payload["negative"]
    positive_range = positive["focus"].get("length_range_chars")
    negative_range = negative["focus"].get("length_range_chars")
    if not isinstance(positive_range, dict) or positive_range != negative_range:
        raise ValidationError("SC variants must use the same length_range_chars")

    minimum = positive_range.get("min")
    maximum = positive_range.get("max")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (minimum, maximum)
        )
        or minimum < 0
        or minimum > maximum
    ):
        raise ValidationError("SC length range must contain valid integer min/max")
    if any(
        not minimum <= len(chunk) <= maximum for chunk in positive["chunks"]
    ):
        raise ValidationError("all positive SC chunks must satisfy the length range")

    violating = {
        index
        for index, chunk in enumerate(negative["chunks"])
        if not minimum <= len(chunk) <= maximum
    }
    if not violating:
        raise ValidationError(
            "at least one negative SC chunk must violate the length range"
        )
    if violating.isdisjoint(negative["focus"]["target_chunk_indices"]):
        raise ValidationError("negative SC focus must identify a violating chunk")


def _validate_information_preservation(payload: dict[str, Any]) -> None:
    positive = payload["positive"]
    negative = payload["negative"]
    if _normalized_text(negative["chunks"]) == _normalized_text(
        payload["source_document"]
    ):
        raise ValidationError(
            "Information Preservation negative must lose or alter source information"
        )
    positive_fact = positive["focus"]["fact"]
    if (
        positive_fact != negative["focus"]["fact"]
        or _normalized_text(positive_fact)
        not in _normalized_text(payload["source_document"])
    ):
        raise ValidationError(
            "Information Preservation focus must name one source fact in both variants"
        )


def validate_pair(payload: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """Validate one model response against its prompt-specific contract."""
    if not isinstance(payload, dict):
        raise ValidationError("response must be a JSON object")
    for field in ("document_title", "source_document", "controlled_change"):
        _require_string(payload.get(field), field)
    if payload.get("expected_relation") != EXPECTED_RELATION:
        raise ValidationError(
            f"expected_relation must equal {EXPECTED_RELATION!r}"
        )

    prompt_name = entry["prompt_name"]
    _validate_variants(payload, prompt_name)
    if entry["dataset_type"] == "metric_specific":
        _validate_source_restoration(payload, prompt_name)
    if prompt_name == "size_compliance":
        _validate_size_compliance(payload)
    if prompt_name == "hope_information_preservation":
        _validate_information_preservation(payload)
    return payload


def make_record(
    payload: dict[str, Any],
    entry: dict[str, Any],
    *,
    run_id: str,
    model: str,
) -> dict[str, Any]:
    """Add provenance and review metadata to a validated model response."""
    return {
        "record_id": str(uuid4()),
        "run_id": run_id,
        "dataset_type": entry["dataset_type"],
        "target_metric": entry["target_metric"],
        "prompt_name": entry["prompt_name"],
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "unreviewed",
        **payload,
    }


def append_record(record: dict[str, Any], output_root: Path) -> Path:
    """Immediately append one successful record to its run JSONL file."""
    run_dir = output_root / record["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{record['dataset_type']}.jsonl"
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def generate_pair(
    client: Any,
    entry: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    max_attempts: int = 3,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Request and validate one pair, retrying invalid responses."""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValidationError("response was truncated")
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise ValidationError("response content is empty")
            return validate_pair(json.loads(content), entry), attempt, None
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
    return None, max_attempts, last_error
