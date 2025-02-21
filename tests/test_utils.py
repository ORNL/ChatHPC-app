import json
import unittest
from pathlib import Path

from chathpc.app.app import App
from chathpc.app.utils.common_utils import load_json_arg


class TestLoadJson(unittest.TestCase):
    def test_json_dict(self):
        j = load_json_arg({"name": "test"})
        assert j == {"name": "test"}

    def test_json_str(self):
        j = load_json_arg('{"name": "test"}')
        assert j == {"name": "test"}

    def test_json_none(self):
        j = load_json_arg(None)
        assert j == {}

    def test_json_file(self):
        filename = "tests/files/config.json"
        j = load_json_arg(filename)
        with open(filename) as f:
            jj = json.loads(f.read())
        assert j == jj

    def test_json_path(self):
        path = Path("tests/files/config.json")
        j = load_json_arg(path)
        with open(path) as f:
            jj = json.loads(f.read())
        assert j == jj


class TestExtractAnswer(unittest.TestCase):
    def test_extract_answer_simple(self):
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "question": "Question",
            "context": "Context",
            "answer": "Answer",
        }
        tinput = app.training_prompt(**kwinput)

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."

    def test_extract_answer_simple2(self):
        """Test with different keywords."""
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "Goal\n\n### Question:\n{{question}}\n\n### Additional Info:\n{{context}}\n\n### Answer:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "user": "Question",
            "context": "Context",
            "assistant": "Answer",
        }
        tinput = app.training_prompt(**kwinput)

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."

    def test_extract_answer_simple3(self):
        """Test with bos tag."""
        app = App.from_json(
            "tests/files/config.json",
            extra_params={
                "prompt_template": "Goal\n\n### Question:\n{{question}}\n\n### Additional Info:\n{{context}}\n\n### Answer:\n{{answer}}\n\n",
                "prompt_template_file": None,
            },
        )
        kwinput = {
            "user": "Question",
            "context": "Context",
            "assistant": "Answer",
        }
        tinput = app.training_prompt(**kwinput)
        tinput = "<s> " + tinput + "</s>"

        expected = "Answer"
        result = app.extract_answer(tinput, **kwinput)
        assert result == expected, "Extraction result is not as expected."


if __name__ == "__main__":
    unittest.main()
