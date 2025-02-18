import json
import unittest
from pathlib import Path

from chathpc.app.app import App
from chathpc.app.utils.common_utils import extract_answer, load_json_arg


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
        app = App.from_json("tests/files/config.json")
        app.config.inference_prompt = "System\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n"
        app.config.training_prompt = (
            "System\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n{answer}\n\n"
        )
        tinput_prompt = app.chat_prompt("Question", "Context")
        tinput = app.training_prompt("Question", "Context", "Answer")

        expected = "Answer\n\n"
        result = extract_answer(tinput, tinput_prompt)
        assert result == expected, "Extraction result is not as expected."

    def test_extract_answer_simple2(self):
        app = App.from_json("tests/files/config.json")
        app.config.inference_prompt = (
            "Goal\n\n### Question:\n{question}\n\n### Additional Info:\n{context}\n\n### Answer:\n"
        )
        app.config.training_prompt = (
            "Goal\n\n### Question:\n{question}\n\n### Additional Info:\n{context}\n\n### Answer:\n{answer}\n\n"
        )
        tinput_prompt = app.chat_prompt("Question", "Context")
        tinput = app.training_prompt("Question", "Context", "Answer")

        expected = "Answer"
        result = extract_answer(tinput, prompt=tinput_prompt, stop="\n\n")
        assert result == expected, "Extraction result is not as expected."


if __name__ == "__main__":
    unittest.main()
