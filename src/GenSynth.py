import json
import logging
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import pydantic
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    from .PydanticContracts import (
        BoundaryClarityJudgeResult,
        ChunkScoreJudgeResult,
        ContextualCoherenceJudgeResult,
        GeneralJudgeResult,
        HopeConceptUnityJudgeResult,
        HopeInformationPreservationJudgeResult,
        HopeSemanticIndependenceJudgeResult,
        IntrachunkCohesionJudgeResult,
        SyntheticChunkingExample,
    )
    from .TokenUsage import append_token_usage
except ImportError:
    from PydanticContracts import (
        BoundaryClarityJudgeResult,
        ChunkScoreJudgeResult,
        ContextualCoherenceJudgeResult,
        GeneralJudgeResult,
        HopeConceptUnityJudgeResult,
        HopeInformationPreservationJudgeResult,
        HopeSemanticIndependenceJudgeResult,
        IntrachunkCohesionJudgeResult,
        SyntheticChunkingExample,
    )
    from TokenUsage import append_token_usage


ResultT = TypeVar("ResultT", bound=pydantic.BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "generated"
TOKEN_USAGE_PATH = OUTPUT_ROOT / "token_usage.jsonl"

# Generator
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 1.0
REASONING = False
REASONING_EFFORT = "medium"
MAX_TOKENS = (8192, 10000)[REASONING]
TIMEOUT_SECONDS = 240.0
PAIRS_PER_PROMPT = 30
REGENERATION_ATTEMPTS = 20
USE_JUDGE_FEEDBACK_ON_EVEN_ATTEMPTS = True

# Judge
JUDGE_MODEL_NAME = "deepseek-v4-pro"
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING = True
JUDGE_REASONING_EFFORT = "high"
JUDGE_MAX_TOKENS = (4096, 24000)[JUDGE_REASONING]
JUDGE_REGENERATION_ATTEMPTS = 20

SELECTED_PROMPTS = [
    # (Path("general_validation.md"), GeneralJudgeResult),
    # (Path("metrics/size_compliance.md"), SizeComplianceJudgeResult),
    # (Path("metrics/intrachunk_cohesion.md"), IntrachunkCohesionJudgeResult),
    # (Path("metrics/contextual_coherence.md"), ContextualCoherenceJudgeResult),
    # (Path("metrics/boundary_clarity.md"), BoundaryClarityJudgeResult),
    # (Path("metrics/chunk_score.md"), ChunkScoreJudgeResult),
    # (Path("metrics/hope_concept_unity.md"), HopeConceptUnityJudgeResult),
    (
        Path("metrics/hope_semantic_independence.md"),
        HopeSemanticIndependenceJudgeResult,
    ),
    (
        Path("metrics/hope_information_preservation.md"),
        HopeInformationPreservationJudgeResult,
    ),
]

LOGGER = logging.getLogger(__name__)
RUN_ID = uuid4().hex
client: OpenAI | None = None


def save_json(data: list[dict[str, Any]], path: Path) -> Path:
    """Save generated examples in a human-readable UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_saved_results(path: Path) -> list[dict[str, Any]]:
    """Load examples saved by an earlier run, if any."""
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return data


def pending_pair_numbers(saved_results: list[dict[str, Any]]) -> range:
    """Return the pair numbers that remain after the saved prefix."""
    if len(saved_results) > PAIRS_PER_PROMPT:
        raise ValueError(
            f"{len(saved_results)} examples in output exceed PAIRS_PER_PROMPT="
            f"{PAIRS_PER_PROMPT}"
        )
    return range(len(saved_results) + 1, PAIRS_PER_PROMPT + 1)


def with_json_schema(
    prompt: str, result_model: type[pydantic.BaseModel]
) -> str:
    """Append a compact Pydantic JSON schema to a system prompt."""
    schema = json.dumps(
        result_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{prompt.rstrip()}\n\nJSON schema ответа:\n{schema}"


def get_client() -> OpenAI:
    """Create the API client only when a generation run starts."""
    global client
    if client is None:
        client = OpenAI(
            api_key=os.environ["API_KEY"],
            base_url=BASE_URL,
            timeout=TIMEOUT_SECONDS,
        )
    return client


def llm_judge(
    example: SyntheticChunkingExample,
    system_prompt: str,
    metric_prompt: str,
    result_model: type[ResultT],
    prompt: str,
    pair_number: int,
    generator_attempt: int,
    used_judge_feedback: bool,
) -> ResultT:
    messages = [
        {
            "role": "system",
            "content": with_json_schema(system_prompt, result_model),
        },
        {
            "role": "user",
            "content": (
                f"{metric_prompt}\n\n"
                "Проверь следующий синтетический пример:\n\n"
                f"{example.model_dump_json(indent=2)}"
            ),
        },
    ]

    for judge_attempt in range(1, JUDGE_REGENERATION_ATTEMPTS + 1):
        response = get_client().chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body={
                "thinking": {"type": ("disabled", "enabled")[JUDGE_REASONING]}
            },
            reasoning_effort=JUDGE_REASONING_EFFORT,
        )
        try:
            content = response.choices[0].message.content
            verdict = result_model.model_validate_json(content)
        except (
            pydantic.ValidationError,
            json.JSONDecodeError,
            IndexError,
            AttributeError,
            TypeError,
        ):
            append_token_usage(
                response,
                TOKEN_USAGE_PATH,
                run_id=RUN_ID,
                prompt=prompt,
                pair_number=pair_number,
                role="judge",
                generator_attempt=generator_attempt,
                judge_attempt=judge_attempt,
                used_judge_feedback=used_judge_feedback,
                model=JUDGE_MODEL_NAME,
                endpoint=JUDGE_BASE_URL,
                temperature=JUDGE_TEMPERATURE,
                reasoning=JUDGE_REASONING,
                reasoning_effort=JUDGE_REASONING_EFFORT,
                max_tokens=JUDGE_MAX_TOKENS,
                result="invalid_json",
            )
            LOGGER.warning(
                "Pair %s: judge returned invalid JSON (attempt %s/%s)",
                pair_number,
                judge_attempt,
                JUDGE_REGENERATION_ATTEMPTS,
            )
            continue

        append_token_usage(
            response,
            TOKEN_USAGE_PATH,
            run_id=RUN_ID,
            prompt=prompt,
            pair_number=pair_number,
            role="judge",
            generator_attempt=generator_attempt,
            judge_attempt=judge_attempt,
            used_judge_feedback=used_judge_feedback,
            model=JUDGE_MODEL_NAME,
            endpoint=JUDGE_BASE_URL,
            temperature=JUDGE_TEMPERATURE,
            reasoning=JUDGE_REASONING,
            reasoning_effort=JUDGE_REASONING_EFFORT,
            max_tokens=JUDGE_MAX_TOKENS,
            result=("accepted" if verdict.valid else "judge_rejected"),
        )
        return verdict

    raise RuntimeError(
        f"Judge did not return valid {result_model.__name__} JSON after "
        f"{JUDGE_REGENERATION_ATTEMPTS} attempts"
    )


def generate(
    system_prompt: str,
    judge_system_prompt: str,
    user_prompt: str,
    judge_metric_prompt: str,
    judge_feedback_prompt: str,
    judge_result_model: type[ResultT],
    prompt: str,
    pair_number: int,
) -> dict[str, Any]:
    last_rejected_example = None
    last_judge_verdict = None

    def log_generator_response(
        response: Any, result: str, attempt: int, used_judge_feedback: bool
    ) -> None:
        append_token_usage(
            response,
            TOKEN_USAGE_PATH,
            run_id=RUN_ID,
            prompt=prompt,
            pair_number=pair_number,
            role="generator",
            generator_attempt=attempt,
            judge_attempt=None,
            used_judge_feedback=used_judge_feedback,
            model=MODEL_NAME,
            endpoint=BASE_URL,
            temperature=TEMPERATURE,
            reasoning=REASONING,
            reasoning_effort=REASONING_EFFORT,
            max_tokens=MAX_TOKENS,
            result=result,
        )

    for attempt in range(1, REGENERATION_ATTEMPTS + 1):
        messages = [
            {
                "role": "system",
                "content": with_json_schema(
                    system_prompt, SyntheticChunkingExample
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        used_judge_feedback = (
            USE_JUDGE_FEEDBACK_ON_EVEN_ATTEMPTS
            and attempt % 2 == 0
            and last_rejected_example is not None
            and last_judge_verdict is not None
        )
        if used_judge_feedback:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": last_rejected_example.model_dump_json(indent=2),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{judge_feedback_prompt.rstrip()}\n\n"
                            "Полный verdict судьи:\n"
                            f"{last_judge_verdict.model_dump_json(indent=2)}"
                        ),
                    },
                ]
            )

        last_rejected_example = None
        last_judge_verdict = None

        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": ("disabled", "enabled")[REASONING]}},
            reasoning_effort=REASONING_EFFORT,
        )
        try:
            content = response.choices[0].message.content
            result = SyntheticChunkingExample.model_validate_json(content)
        except (
            pydantic.ValidationError,
            json.JSONDecodeError,
            IndexError,
            AttributeError,
            TypeError,
        ):
            log_generator_response(response, "invalid_json", attempt, used_judge_feedback)
            LOGGER.warning(
                "Pair %s: generator returned invalid JSON (attempt %s/%s)",
                pair_number,
                attempt,
                REGENERATION_ATTEMPTS,
            )
            continue

        try:
            judge_verdict = llm_judge(
                example=result,
                system_prompt=judge_system_prompt,
                metric_prompt=judge_metric_prompt,
                result_model=judge_result_model,
                prompt=prompt,
                pair_number=pair_number,
                generator_attempt=attempt,
                used_judge_feedback=used_judge_feedback,
            )
        except Exception:
            log_generator_response(response, "judge_error", attempt, used_judge_feedback)
            raise

        if not judge_verdict.valid:
            log_generator_response(
                response, "judge_rejected", attempt, used_judge_feedback
            )
            LOGGER.info(
                "Pair %s: judge rejected generator attempt %s/%s",
                pair_number,
                attempt,
                REGENERATION_ATTEMPTS,
            )
            last_rejected_example = result
            last_judge_verdict = judge_verdict
            continue

        log_generator_response(response, "accepted", attempt, used_judge_feedback)
        LOGGER.info("Pair %s: judge accepted", pair_number)
        return result.model_dump()

    raise RuntimeError(
        f"Generator did not produce a judge-approved "
        f"{judge_result_model.__name__} example after "
        f"{REGENERATION_ATTEMPTS} attempts"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(PROJECT_ROOT / ".env")
    LOGGER.info("Generation run: %s", RUN_ID)

    system_prompt = (PROMPTS_ROOT / "system.md").read_text(encoding="utf-8")
    judge_system_prompt = (PROMPTS_ROOT / "judge" / "system.md").read_text(
        encoding="utf-8"
    )
    judge_feedback_prompt = (PROMPTS_ROOT / "judge_feedback.md").read_text(
        encoding="utf-8"
    )

    for prompt_path, judge_result_model in tqdm(
        SELECTED_PROMPTS, desc="Prompts", position=0
    ):
        prompt_name = prompt_path.stem
        output_path = OUTPUT_ROOT / f"{prompt_name}.json"
        results = load_saved_results(output_path)
        pending_pairs = pending_pair_numbers(results)

        if not pending_pairs:
            LOGGER.info("%s: already contains %s examples", prompt_name, len(results))
            continue

        LOGGER.info(
            "%s: continuing from pair %s of %s",
            prompt_name,
            pending_pairs.start,
            PAIRS_PER_PROMPT,
        )
        user_prompt = (PROMPTS_ROOT / prompt_path).read_text(encoding="utf-8")
        judge_metric_prompt = (PROMPTS_ROOT / "judge" / prompt_path).read_text(
            encoding="utf-8"
        )
        for pair_number in tqdm(
            pending_pairs,
            total=PAIRS_PER_PROMPT,
            initial=len(results),
            desc=prompt_name,
            position=1,
            leave=False,
        ):
            result = generate(
                system_prompt=system_prompt,
                judge_system_prompt=judge_system_prompt,
                user_prompt=user_prompt,
                judge_metric_prompt=judge_metric_prompt,
                judge_feedback_prompt=judge_feedback_prompt,
                judge_result_model=judge_result_model,
                prompt=str(prompt_path),
                pair_number=pair_number,
            )
            results.append(result)
            save_json(results, output_path)

        LOGGER.info("%s: saved %s examples to %s", prompt_name, len(results), output_path)


if __name__ == "__main__":
    main()
