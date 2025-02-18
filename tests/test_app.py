import unittest

from chathpc.app.app import App


def test_chat_prompt_json():
    """Test basice prompt fuction from reading JSON config."""
    app = App.from_json("tests/files/config.json")

    expected = "You are a powerful LLM model for Kokkos. Your job is to answer questions about Kokkos programming model. You are given a question and context regarding Kokkos programming model.\n\nYou must output the Kokkos question that answers the question.\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\n"
    result = app.chat_prompt("Question", "Context")
    assert result == expected, "Prompt is not as expected."


def test_chat_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json("tests/files/config.json")
    app.config.inference_prompt = "System\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n"
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\n"
    result = app.chat_prompt("Question", "Context")
    assert result == expected, "Prompt is not as expected."


def test_training_prompt_simple():
    """Test basice prompt fuction from simple template setting."""
    app = App.from_json("tests/files/config.json")
    app.config.training_prompt = (
        "System\n\n### Input:\n{question}\n\n### Context:\n{context}\n\n### Response:\n{answer}\n\n"
    )
    expected = "System\n\n### Input:\nQuestion\n\n### Context:\nContext\n\n### Response:\nAnswer\n\n"
    result = app.training_prompt("Question", "Context", "Answer")
    assert result == expected, "Prompt is not as expected."


if __name__ == "__main__":
    unittest.main()
