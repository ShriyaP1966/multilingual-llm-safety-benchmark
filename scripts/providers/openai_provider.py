"""
openai_provider.py

OpenAI provider implementation.
"""

from .base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI model provider.
    """

    def __init__(self, config):

        self.config = config

        self.model = config["models"]["openai"]["model"]

        self.temperature = config["generation"]["temperature"]

        self.max_tokens = config["generation"]["max_output_tokens"]

        self.timeout = config["generation"]["timeout"]

        self.retry_attempts = config["runtime"]["retry_attempts"]

        self.retry_delay = config["runtime"]["retry_delay"]

    def generate_response(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response.

        API implementation will be added later.
        """

        raise NotImplementedError(
            "OpenAI API integration not completed."
        )