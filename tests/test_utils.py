import json
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
