import json
import os
import unittest
from pathlib import Path

from chathpc.app.app import AppConfig
from chathpc.app.utils.common_utils import load_json_arg


class TestAppConfig(unittest.TestCase):
    def setUp(self):
        os.environ["CHATHPC_DATA_FILE"] = "files/data_file.json"
        os.environ["CHATHPC_BASE_MODEL_PATH"] = "files/base_model"
        os.environ["CHATHPC_FINETUNED_MODEL_PATH"] = "files/finetuned_model"
        os.environ["CHATHPC_MERGED_MODEL_PATH"] = "files/merged_model"
        os.environ["CHATHPC_TRAINING_PROMPT"] = "train_prompt"
        os.environ["CHATHPC_INFERENCE_PROMPT"] = "inference_prompt"

    def test_test_config(self):
        preferences = AppConfig()
        assert preferences.data_file == Path("files/data_file.json")

    def tearDown(self):
        os.environ.pop("CHATHPC_DATA_FILE")
        os.environ.pop("CHATHPC_BASE_MODEL_PATH")
        os.environ.pop("CHATHPC_FINETUNED_MODEL_PATH")
        os.environ.pop("CHATHPC_MERGED_MODEL_PATH")
        os.environ.pop("CHATHPC_TRAINING_PROMPT")
        os.environ.pop("CHATHPC_INFERENCE_PROMPT")


class TestAppPreferences(unittest.TestCase):
    def setUp(self):
        os.environ["CHATHPC_DATA_FILE"] = "files/data_file.json"
        os.environ["CHATHPC_TRAINING_PROMPT"] = "train_prompt"
        os.environ["CHATHPC_INFERENCE_PROMPT"] = "inference_prompt"

    def tearDown(self):
        os.environ.pop("CHATHPC_DATA_FILE")
        os.environ.pop("CHATHPC_TRAINING_PROMPT")
        os.environ.pop("CHATHPC_INFERENCE_PROMPT")

    def test_create(self):
        preferences = AppConfig()
        assert preferences.data_file == Path("files/data_file.json"), "incorrect default data_file"

    def test_update(self):
        preferences = AppConfig()
        preferences.data_file = "new_file.json"
        assert preferences.data_file == "new_file.json", "incorrect default data_file"

    def test_test_config(self):
        preferences = AppConfig(
            data_file="files/data_file.json",
            base_model_path="files/base_model",
            finetuned_model_path="files/finetuned_model",
            merged_model_path="files/merged_model",
        )
        assert preferences.data_file == Path("files/data_file.json")


class TestAppPreferencesJson(unittest.TestCase):
    def test_json(self):
        preferences = AppConfig.from_json("tests/files/config.json")
        json_preferences = json.loads(preferences.model_dump_json())
        json_default = load_json_arg("tests/files/config.json")
        assert json_preferences == json_default, "config missmatch."


if __name__ == "__main__":
    unittest.main()
