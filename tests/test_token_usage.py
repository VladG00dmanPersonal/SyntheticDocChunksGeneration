import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_token_usage_module():
    try:
        return importlib.import_module("src.TokenUsage")
    except ModuleNotFoundError as error:
        raise AssertionError("src.TokenUsage is not implemented") from error


class FakeUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
    ):
        self.data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
                "provider_extra": 11,
            },
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
        }

    def model_dump(self, mode="python"):
        if mode != "json":
            raise AssertionError("usage must be serialized in JSON mode")
        return self.data


def append_record(
    module,
    path: Path,
    *,
    role: str,
    run_id: str,
    usage,
    model: str,
):
    return module.append_token_usage(
        SimpleNamespace(usage=usage),
        path,
        run_id=run_id,
        prompt="metrics/example.md",
        pair_number=1,
        role=role,
        generator_attempt=1,
        judge_attempt=1 if role == "judge" else None,
        used_judge_feedback=False,
        model=model,
        endpoint="https://example.test",
        temperature=0.0 if role == "judge" else 1.0,
        reasoning=role == "judge",
        reasoning_effort="high" if role == "judge" else "medium",
        max_tokens=200 if role == "judge" else 100,
        result="accepted",
    )


class TokenUsageTests(unittest.TestCase):
    def test_append_preserves_nested_usage_details(self):
        module = load_token_usage_module()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token_usage.jsonl"
            append_record(
                module,
                path,
                role="generator",
                run_id="run-1",
                usage=FakeUsage(100, 20, 10, 5),
                model="generator-model",
            )

            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(record["usage_available"])
        self.assertEqual(
            record["usage"]["prompt_tokens_details"],
            {"cached_tokens": 10, "provider_extra": 11},
        )
        self.assertEqual(
            record["usage"]["completion_tokens_details"],
            {"reasoning_tokens": 5},
        )

    def test_append_marks_missing_usage_explicitly(self):
        module = load_token_usage_module()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token_usage.jsonl"
            append_record(
                module,
                path,
                role="judge",
                run_id="run-1",
                usage=None,
                model="judge-model",
            )

            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(record["usage_available"])
        self.assertIsNone(record["usage"])

    def test_summary_separates_judge_usage_by_generator_run(self):
        module = load_token_usage_module()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token_usage.jsonl"
            append_record(
                module,
                path,
                role="generator",
                run_id="run-1",
                usage=FakeUsage(100, 20, 10, 5),
                model="generator-model-a",
            )
            append_record(
                module,
                path,
                role="generator",
                run_id="run-2",
                usage=FakeUsage(50, 10, 5, 2),
                model="generator-model-b",
            )
            append_record(
                module,
                path,
                role="judge",
                run_id="run-1",
                usage=FakeUsage(30, 6, 3, 4),
                model="judge-model",
            )
            append_record(
                module,
                path,
                role="judge",
                run_id="run-1",
                usage=None,
                model="judge-model",
            )
            append_record(
                module,
                path,
                role="judge",
                run_id="run-2",
                usage=FakeUsage(40, 8, 4, 6),
                model="judge-model",
            )

            usage_df = module.load_token_usage(path)
            summary = module.summarize_token_usage(usage_df).set_index(
                ["run_id", "role"]
            )

        run_1_generator = summary.loc[("run-1", "generator")]
        self.assertEqual(run_1_generator["generator_model"], "generator-model-a")
        self.assertEqual(run_1_generator["requests"], 1)
        self.assertEqual(run_1_generator["total_tokens"], 120)

        run_1_judge = summary.loc[("run-1", "judge")]
        self.assertEqual(run_1_judge["generator_model"], "generator-model-a")
        self.assertEqual(run_1_judge["requests"], 2)
        self.assertEqual(run_1_judge["responses_with_usage"], 1)
        self.assertEqual(run_1_judge["total_tokens"], 36)

        run_2_generator = summary.loc[("run-2", "generator")]
        self.assertEqual(run_2_generator["generator_model"], "generator-model-b")
        self.assertEqual(run_2_generator["requests"], 1)
        self.assertEqual(run_2_generator["total_tokens"], 60)

        run_2_judge = summary.loc[("run-2", "judge")]
        self.assertEqual(run_2_judge["generator_model"], "generator-model-b")
        self.assertEqual(run_2_judge["requests"], 1)
        self.assertEqual(run_2_judge["total_tokens"], 48)


if __name__ == "__main__":
    unittest.main()
