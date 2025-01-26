import json
import os
import unittest

from chathpc.app.app import DEFAULT_APP_CONFIG_FILE, AppConfig
from chathpc.app.utils.common_utils import load_json_arg


class TestAppConfig(unittest.TestCase):
    def setUp(self):
        os.environ["CHATHPC_DATA_FILE"] = "files/data_file.json"
        os.environ["CHATHPC_BASE_MODEL_PATH"] = "files/base_model"
        os.environ["CHATHPC_FINETUNED_MODEL_PATH"] = "files/finetuned_model"
        os.environ["CHATHPC_MERGED_MODEL_PATH"] = "files/merged_model"

    def test_test_config(self):
        preferences = AppConfig()
        assert preferences.data_file == "files/data_file.json"

    def tearDown(self):
        os.environ.pop("CHATHPC_DATA_FILE")
        os.environ.pop("CHATHPC_BASE_MODEL_PATH")
        os.environ.pop("CHATHPC_FINETUNED_MODEL_PATH")
        os.environ.pop("CHATHPC_MERGED_MODEL_PATH")


class TestAppPreferences(unittest.TestCase):
    def test_create(self):
        preferences = AppConfig()
        assert (
            preferences.data_file == "/home/7ry/Data/ellora/kokkos-data/kokkos_create_context.json"
        ), "incorrect default data_file"

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
        assert preferences.data_file == "files/data_file.json"

    def test_json(self):
        preferences = AppConfig()
        json_preferences = json.loads(preferences.model_dump_json())
        json_default = load_json_arg(DEFAULT_APP_CONFIG_FILE)
        assert json_preferences == json_default, "config missmatch."


if __name__ == "__main__":
    unittest.main()
