from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chathpc.app.app import AppConfig

import openai
from loguru import logger
from openai import OpenAI

from chathpc.app.utils import template_utils


class ChatHPCOpenAI:
    def __init__(self, config: AppConfig):
        try:
            logger.info("OPENAI_API_BASE_URL is {}", os.environ.get("OPENAI_API_BASE_URL"))
            logger.info("OPENAI_API_KEY set? {}", "OPENAI_API_KEY" in os.environ)
            self.client = OpenAI(
                base_url=os.environ.get("OPENAI_API_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        except openai.OpenAIError as e:
            print(
                "Error: Unable to connect to OpenAI API endpoint. Please check the settings of OPENAI_API_BASE_URL and OPENAI_API_KEY."
            )
            print(e)
            sys.exit(1)
        self.config = config

    def openai_chat_evaluate(self, model_name: str, **kwargs) -> str | None:
        """Evaluate a prompt using OpenAI's ChatCompletion API.

        Args:
            model_name (str): Name of the OpenAI chat model to use (e.g. 'gpt-3.5-turbo')
            **kwargs: Keyword arguments containing prompt and context
                - prompt: The input prompt to evaluate
                - context: System context/instructions

        Returns:
            str | None: The generated chat response content, stripped of whitespace,
                        or None if chat fails

        Example:
            response = openai_chat_evaluate("gpt-3.5-turbo",
                                            prompt="What is 2+2?",
                                            context="You are a math tutor")
        """

        kw = template_utils.map_keywords(kwargs)

        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": kw["context"]},
                {"role": "user", "content": kw["prompt"]},
            ],
            max_completion_tokens=self.config.max_response_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content
