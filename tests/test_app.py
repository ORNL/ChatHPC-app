import unittest

from chathpc.app.app import App, AppConfig


def test_chat_prompt_json():
    """Test basice prompt fuction from reading JSON config."""
    app = App.from_json("tests/files/config.json")

    expected = "You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\n"
    result = app.chat_prompt(question="Question", context="Context")
    assert result == expected, "Prompt is not as expected."


def test_chat_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    config = AppConfig.from_json(
        "tests/files/config.json",
        extra_params={
            "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n",
            "prompt_template_file": None,
        },
    )
    app = App(config)
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\n"
    result = app.chat_prompt(prompt="Question", context="Context")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json(
        "tests/files/config.json",
        extra_params={
            "prompt_template": "System\n\n### Input:\n{{question}}\n\n### Context:\n{{context}}\n\n### Response:\n{{answer}}\n\n",
            "prompt_template_file": None,
        },
    )
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\nAnswer\n\n"
    result = app.training_prompt(user="Question", context="Context", assistant="Answer")
    assert result == expected, "Prompt is not as expected."


if __name__ == "__main__":
    unittest.main()
