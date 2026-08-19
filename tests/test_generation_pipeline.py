from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.generation_pipeline import (
    EXPECTED_RELATION,
    ValidationError,
    append_record,
    build_prompt_registry,
    generate_pair,
    load_prompt_text,
    make_record,
    select_prompts,
    validate_pair,
)


SC_ENTRY = {
    "prompt_name": "size_compliance",
    "dataset_type": "metric_specific",
    "target_metric": "Size Compliance",
}


def make_sc_pair():
    chunks = [
        "1.1. Фонд содействует образовательным проектам.",
        "1.2. Решение принимает попечительский совет.",
        "1.3. Отчёт публикуется ежегодно до 1 марта.",
    ]
    focus = {
        "target_chunk_indices": [0],
        "length_range_chars": {"min": 40, "max": 65},
    }
    return {
        "document_title": "Устав фонда «Северный маяк»",
        "source_document": "".join(chunks),
        "positive": {
            "chunks": chunks,
            "rationale": "Все чанки входят в диапазон.",
            "focus": deepcopy(focus),
        },
        "negative": {
            "chunks": ["".join(chunks)],
            "rationale": "Единый чанк превышает диапазон.",
            "focus": deepcopy(focus),
        },
        "controlled_change": "Удалены две границы.",
        "expected_relation": EXPECTED_RELATION,
    }


class FakeCompletions:
    def __init__(self, valid_payload, valid_on_attempt):
        self.valid_payload = valid_payload
        self.valid_on_attempt = valid_on_attempt
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        content = (
            json.dumps(self.valid_payload, ensure_ascii=False)
            if self.calls == self.valid_on_attempt
            else "{}"
        )
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=content),
        )
        return SimpleNamespace(choices=[choice])


class GenerationPipelineTests(unittest.TestCase):
    def test_prompt_registry_loads_nine_selectable_prompts(self):
        with TemporaryDirectory() as temporary_directory:
            prompts_root = Path(temporary_directory)
            (prompts_root / "metrics").mkdir()
            (prompts_root / "system.md").write_text("system", encoding="utf-8")
            registry = build_prompt_registry(prompts_root)
            for entry in registry:
                entry["path"].write_text(entry["prompt_name"], encoding="utf-8")

            selected = select_prompts(registry, ["size_compliance"])
            system, user = load_prompt_text(prompts_root, selected[0])

            self.assertEqual(len(registry), 9)
            self.assertEqual([item["prompt_name"] for item in selected], ["size_compliance"])
            self.assertEqual((system, user), ("system", "size_compliance"))
            with self.assertRaises(ValueError):
                select_prompts(registry, ["unknown"])

    def test_validate_pair_accepts_valid_size_compliance_pair(self):
        pair = make_sc_pair()
        self.assertIs(validate_pair(pair, SC_ENTRY), pair)

    def test_validate_pair_rejects_broken_contract(self):
        mutators = [
            lambda pair: pair.update(expected_relation="wrong"),
            lambda pair: pair["negative"]["focus"].update(
                target_chunk_indices=[1]
            ),
            lambda pair: pair["negative"]["chunks"].append("лишний текст"),
            lambda pair: (
                pair["positive"]["focus"].update(
                    length_range_chars={"min": 40, "max": 200}
                ),
                pair["negative"]["focus"].update(
                    length_range_chars={"min": 40, "max": 200}
                ),
            ),
        ]
        for mutator in mutators:
            with self.subTest(mutator=mutator):
                pair = make_sc_pair()
                mutator(pair)
                with self.assertRaises(ValidationError):
                    validate_pair(pair, SC_ENTRY)

    def test_information_preservation_requires_loss_or_distortion(self):
        first = "1.1. Отчёт подаётся до 1 марта."
        second = "1.2. Совет утверждает отчёт."
        focus = {
            "affected_chunk_indices": [0],
            "fact": "Отчёт подаётся до 1 марта.",
        }
        pair = {
            "document_title": "Устав вымышленного фонда",
            "source_document": first + second,
            "positive": {
                "chunks": [first, second],
                "rationale": "Факт сохранён.",
                "focus": deepcopy(focus),
            },
            "negative": {
                "chunks": [first + second],
                "rationale": "Изменена только группировка.",
                "focus": deepcopy(focus),
            },
            "controlled_change": "Изменена только граница.",
            "expected_relation": EXPECTED_RELATION,
        }
        entry = {
            "prompt_name": "hope_information_preservation",
            "dataset_type": "metric_specific",
            "target_metric": "HOPE Information Preservation",
        }
        with self.assertRaises(ValidationError):
            validate_pair(pair, entry)

    def test_generate_pair_retries_invalid_contract_three_times(self):
        pair = make_sc_pair()
        completions = FakeCompletions(pair, valid_on_attempt=3)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        payload, attempts, error = generate_pair(
            client,
            SC_ENTRY,
            "system",
            "user",
            model="deepseek-v4-pro",
        )
        self.assertEqual(payload, pair)
        self.assertEqual(attempts, 3)
        self.assertIsNone(error)
        self.assertEqual(completions.calls, 3)
        self.assertEqual(
            completions.last_kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            completions.last_kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_make_and_append_record_preserve_metadata(self):
        record = make_record(
            make_sc_pair(),
            SC_ENTRY,
            run_id="offline-smoke",
            model="deepseek-v4-pro",
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = append_record(record, root)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                output_path,
                root / "offline-smoke" / "metric_specific.jsonl",
            )
            self.assertEqual(saved["record_id"], record["record_id"])
            self.assertEqual(saved["review_status"], "unreviewed")
            self.assertEqual(saved["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
