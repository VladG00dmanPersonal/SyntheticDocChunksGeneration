import unittest

from pydantic import ValidationError

from src.PydanticContracts import LengthRangeChars, SyntheticChunkingExample


class SyntheticChunkingExampleContractTests(unittest.TestCase):
    def test_accepts_pair_level_rationale_and_evaluation_context(self):
        payload = {
            "document_title": "Устав автономной организации «Горизонт»",
            "source_document": "1.1. Организация проводит собрание ежегодно.",
            "positive": {
                "chunks": ["1.1. Организация проводит собрание ежегодно."],
            },
            "negative": {
                "chunks": ["1.1. Организация проводит", "собрание ежегодно."],
            },
            "controlled_change": "Условие отделено от основного положения.",
            "contrast_rationale": (
                "Negative хуже сохраняет смысловую завершённость положения."
            ),
            "evaluation_context": {
                "cue_question": "Как часто организация проводит собрание?",
            },
        }

        try:
            example = SyntheticChunkingExample.model_validate(payload)
        except ValidationError as error:
            self.fail(f"Новый контракт пары был отклонён: {error}")

        self.assertEqual(
            example.evaluation_context.cue_question,
            "Как часто организация проводит собрание?",
        )

    def test_rejects_multiple_evaluation_context_kinds(self):
        payload = {
            "document_title": "Устав автономной организации «Горизонт»",
            "source_document": "1.1. Организация проводит собрание ежегодно.",
            "positive": {
                "chunks": ["1.1. Организация проводит собрание ежегодно."],
            },
            "negative": {
                "chunks": ["1.1. Организация проводит", "собрание ежегодно."],
            },
            "controlled_change": "Условие отделено от основного положения.",
            "contrast_rationale": (
                "Negative хуже сохраняет смысловую завершённость положения."
            ),
            "evaluation_context": {
                "cue_question": "Как часто организация проводит собрание?",
                "fact": "Организация проводит собрание ежегодно.",
            },
        }

        with self.assertRaises(ValidationError):
            SyntheticChunkingExample.model_validate(payload)

    def test_rejects_reversed_length_range(self):
        with self.assertRaises(ValidationError):
            LengthRangeChars.model_validate({"min": 500, "max": 120})


if __name__ == "__main__":
    unittest.main()
