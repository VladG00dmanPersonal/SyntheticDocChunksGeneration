"""Minimal helpers for generating and saving JSON responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_json(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    regeneration_attempts: int = 0,
    log_context: str | None = None,
) -> dict[str, Any]:
    """Generate one JSON object with an OpenAI-compatible client."""
    context = f" for {log_context}" if log_context else ""
    for attempt in range(regeneration_attempts + 1):
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
                # reasoning_effort="high",
                # extra_body={"thinking": {"type": "enabled"}},
            )
            # print(response)
            return json.loads(response.choices[0].message.content)
        except Exception as error:
            remaining_attempts = regeneration_attempts - attempt
            print(
                f"Error generating JSON{context}: "
                f"{type(error).__name__}: {error}. Request "
                f"{attempt + 1}/{regeneration_attempts + 1}. "
                f"Regeneration attempts remaining: {remaining_attempts}"
            )
            if remaining_attempts == 0:
                raise

    raise RuntimeError("JSON generation attempts exhausted")


def save_json(
    data: dict[str, Any] | list[dict[str, Any]], path: Path
) -> Path:
    """Save JSON objects in a human-readable UTF-8 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
