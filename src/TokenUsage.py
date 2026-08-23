import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Convert an SDK usage object to JSON-compatible data."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    return json.loads(json.dumps(usage, default=vars))


def append_token_usage(
    response: Any,
    path: Path,
    *,
    run_id: str,
    prompt: str,
    pair_number: int,
    role: str,
    generator_attempt: int,
    judge_attempt: int | None,
    used_judge_feedback: bool,
    model: str,
    endpoint: str,
    temperature: float,
    reasoning: bool,
    reasoning_effort: str,
    max_tokens: int,
    result: str,
) -> dict[str, Any]:
    """Append usage metadata for one received API response."""
    usage = _usage_to_dict(getattr(response, "usage", None))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "prompt": prompt,
        "pair_number": pair_number,
        "role": role,
        "generator_attempt": generator_attempt,
        "judge_attempt": judge_attempt,
        "used_judge_feedback": used_judge_feedback,
        "model": model,
        "endpoint": endpoint,
        "temperature": temperature,
        "reasoning": reasoning,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "result": result,
        "usage_available": usage is not None,
        "usage": usage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_token_usage(path: Path) -> pd.DataFrame:
    """Load append-only token usage history from JSONL."""
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as stream:
        return pd.DataFrame(json.loads(line) for line in stream if line.strip())


def _usage_value(usage: Any, *paths: tuple[str, ...]) -> int | None:
    if not isinstance(usage, dict):
        return None
    for path in paths:
        value: Any = usage
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def summarize_token_usage(usage_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each run by role and configuration."""
    group_columns = [
        "run_id",
        "generator_model",
        "role",
        "model",
        "endpoint",
        "temperature",
        "reasoning",
        "reasoning_effort",
        "max_tokens",
    ]
    summary_columns = [
        *group_columns,
        "runs",
        "requests",
        "responses_with_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
    ]
    if usage_df.empty:
        return pd.DataFrame(columns=summary_columns)

    data = usage_df.copy()
    generator_models = (
        data.loc[data["role"].eq("generator"), ["run_id", "model"]]
        .drop_duplicates("run_id")
        .set_index("run_id")["model"]
    )
    data["generator_model"] = data["run_id"].map(generator_models)
    token_paths = {
        "prompt_tokens": (("prompt_tokens",),),
        "completion_tokens": (("completion_tokens",),),
        "total_tokens": (("total_tokens",),),
        "cached_tokens": (
            ("prompt_tokens_details", "cached_tokens"),
            ("cached_tokens",),
        ),
        "reasoning_tokens": (
            ("completion_tokens_details", "reasoning_tokens"),
            ("reasoning_tokens",),
        ),
    }
    for column, paths in token_paths.items():
        values = data["usage"].map(
            lambda usage, paths=paths: _usage_value(usage, *paths)
        )
        data[column] = pd.to_numeric(values, errors="coerce").astype("Int64")
    data["usage_available"] = data["usage_available"].fillna(False).astype(int)

    summary = (
        data.groupby(group_columns, dropna=False)
        .agg(
            runs=("run_id", "nunique"),
            requests=("role", "size"),
            responses_with_usage=("usage_available", "sum"),
            prompt_tokens=("prompt_tokens", lambda values: values.sum(min_count=1)),
            completion_tokens=(
                "completion_tokens",
                lambda values: values.sum(min_count=1),
            ),
            total_tokens=("total_tokens", lambda values: values.sum(min_count=1)),
            cached_tokens=("cached_tokens", lambda values: values.sum(min_count=1)),
            reasoning_tokens=(
                "reasoning_tokens",
                lambda values: values.sum(min_count=1),
            ),
        )
        .reset_index()
    )
    return summary[summary_columns]
