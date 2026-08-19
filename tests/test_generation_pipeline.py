import json
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
