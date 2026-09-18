"""
qwen_provider.py

Qwen provider implementation using Hugging Face Inference Providers.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .base_provider import BaseProvider

load_dotenv()


class QwenProvider(BaseProvider):
    """
    Qwen model provider through Hugging Face Inference Providers.
    """

    def __init__(self, config):

        self.config = config

        self.model = config["models"]["qwen"]["model"]

        self.temperature = config["generation"]["temperature"]

        self.max_tokens = config["generation"]["max_output_tokens"]

        self.timeout = config["generation"]["timeout"]

        self.retry_attempts = config["runtime"]["retry_attempts"]

        self.retry_delay = config["runtime"]["retry_delay"]

        self.api_key = os.getenv("HF_TOKEN")

        if not self.api_key:
            raise ValueError(
                "HF_TOKEN environment variable is not set."
            )

        self.client = InferenceClient(
            api_key=self.api_key,
            provider="auto",
        )

    def generate_response(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response using the configured Qwen model.
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content