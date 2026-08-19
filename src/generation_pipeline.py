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
) -> dict[str, Any]:
    """Generate one JSON object with an OpenAI-compatible client."""
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
    print(response)
    return json.loads(response.choices[0].message.content)


def save_json(data: dict[str, Any], path: Path) -> Path:
    """Save one JSON object in a human-readable UTF-8 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
