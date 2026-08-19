import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import src.generation_pipeline as pipeline


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        choice = SimpleNamespace(
            message=SimpleNamespace(
                content=json.dumps(self.payload, ensure_ascii=False)
            )
        )
        return SimpleNamespace(choices=[choice])


class FlakyCompletions(FakeCompletions):
    def __init__(self, payload, failures):
        super().__init__(payload)
        self.failures = failures
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError(f"temporary failure {self.attempts}")
        return super().create(**kwargs)


class GenerationPipelineTests(unittest.TestCase):
    def test_generate_json_returns_model_payload(self):
        payload = {"document_title": "Устав фонда «Маяк»"}
        completions = FakeCompletions(payload)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result = pipeline.generate_json(
            client,
            "system prompt",
            "user prompt",
            model="deepseek-v4-pro",
        )

        self.assertEqual(result, payload)
        self.assertEqual(
            completions.request["messages"],
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
        )
        self.assertEqual(
            completions.request["response_format"], {"type": "json_object"}
        )
        self.assertEqual(
            completions.request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_generate_json_retries_until_success_and_logs_errors(self):
        payload = {"document_title": "Устав фонда «Маяк»"}
        completions = FlakyCompletions(payload, failures=2)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        output = StringIO()

        with redirect_stdout(output):
            result = pipeline.generate_json(
                client,
                "system prompt",
                "user prompt",
                model="deepseek-v4-pro",
                regeneration_attempts=3,
                log_context="intrachunk_cohesion pair 2/10",
            )

        self.assertEqual(result, payload)
        self.assertEqual(completions.attempts, 3)
        self.assertIn(
            "Error generating JSON for intrachunk_cohesion pair 2/10: "
            "RuntimeError: temporary failure 1",
            output.getvalue(),
        )
        self.assertIn(
            "Error generating JSON for intrachunk_cohesion pair 2/10: "
            "RuntimeError: temporary failure 2",
            output.getvalue(),
        )

    def test_generate_json_stops_after_regeneration_attempts_are_exhausted(self):
        completions = FlakyCompletions({}, failures=4)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "temporary failure 4"):
                pipeline.generate_json(
                    client,
                    "system prompt",
                    "user prompt",
                    model="deepseek-v4-pro",
                    regeneration_attempts=3,
                    log_context="intrachunk_cohesion pair 2/10",
                )

        self.assertEqual(completions.attempts, 4)

    def test_save_json_writes_readable_utf8(self):
        payload = [
            {"document_title": "Устав фонда «Маяк»"},
            {"document_title": "Положение ассоциации «Орион»"},
        ]

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "result.json"
            result = pipeline.save_json(payload, path)

            self.assertEqual(result, path)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "[\n"
                '  {\n    "document_title": "Устав фонда «Маяк»"\n  },\n'
                '  {\n    "document_title": "Положение ассоциации «Орион»"\n  }\n'
                "]\n",
            )


if __name__ == "__main__":
    unittest.main()
