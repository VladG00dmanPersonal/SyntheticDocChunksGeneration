import importlib
import json
import tempfile
import unittest
from pathlib import Path


def load_generator_module():
    try:
        return importlib.import_module("src.GenSynth")
    except ModuleNotFoundError as error:
        raise AssertionError("src.GenSynth is not implemented") from error


class GenSynthResumeTests(unittest.TestCase):
    def test_rejects_a_saved_json_object_instead_of_a_result_list(self):
        """A corrupted output must not be treated as an empty result list."""
        module = load_generator_module()

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "hope_semantic_independence.json"
            output_path.write_text('{"pair": 1}', encoding="utf-8")

            with self.assertRaises(TypeError):
                module.load_saved_results(output_path)

    def test_resumes_with_the_pair_after_saved_examples(self):
        """A restart must not generate the first 16 pairs a second time."""
        module = load_generator_module()

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "hope_semantic_independence.json"
            output_path.write_text(
                json.dumps([{"pair": number} for number in range(1, 17)]),
                encoding="utf-8",
            )

            saved_results = module.load_saved_results(output_path)

        self.assertEqual(list(module.pending_pair_numbers(saved_results)), list(range(17, 31)))


if __name__ == "__main__":
    unittest.main()
