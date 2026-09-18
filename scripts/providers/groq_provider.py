"""
groq_provider.py

Groq provider implementation for Qwen and GPT-OSS models.
"""

import os
import time

from dotenv import load_dotenv

from groq import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    Groq,
    RateLimitError,
)

from .base_provider import BaseProvider


load_dotenv()


# Transient failures worth retrying. Anything else (bad model id,
# auth failure, malformed request) is a real error and must surface
# immediately rather than being retried into silence.
RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class GroqProvider(BaseProvider):
    """
    Groq model provider.

    The specific model is selected through config.json.
    """

    def __init__(self, config):

        self.config = config

        self.model = config["models"]["groq"]["model"]

        self.temperature = config["generation"]["temperature"]

        self.max_tokens = config["generation"]["max_output_tokens"]

        self.timeout = config["generation"]["timeout"]

        self.retry_attempts = config["runtime"]["retry_attempts"]

        self.retry_delay = config["runtime"]["retry_delay"]

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(
            api_key=self.api_key
        )

    def generate_response(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response using the configured Groq model.
        """

        request_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        # reasoning_effort is used for GPT-OSS,
        # but not for Qwen.
        if "gpt-oss" in self.model.lower():
            request_params["reasoning_effort"] = "low"

        # --------------------------------------------------------
        # Retry transient failures with exponential backoff.
        #
        # retry_attempts and retry_delay were previously read from
        # config and never used, so a single dropped connection
        # aborted the whole benchmark run. A 312-call run is long
        # enough that transient failures are expected, not unusual.
        # --------------------------------------------------------

        last_error = None

        for attempt in range(1, self.retry_attempts + 1):

            try:

                completion = self.client.chat.completions.create(
                    **request_params
                )

                break

            except RETRYABLE_ERRORS as error:

                last_error = error

                if attempt == self.retry_attempts:
                    raise

                wait = self.retry_delay * (2 ** (attempt - 1))

                print(
                    f"    {type(error).__name__} on attempt "
                    f"{attempt}/{self.retry_attempts}; "
                    f"retrying in {wait}s"
                )

                time.sleep(wait)

        else:  # pragma: no cover - loop always breaks or raises
            raise last_error

        return {
            "text": completion.choices[0].message.content,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }