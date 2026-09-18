"""
gemini_provider.py

Placeholder Gemini provider.
"""

from .base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Placeholder provider.
    """

    def generate_response(self, prompt: str) -> str:

        raise NotImplementedError(
            "Gemini provider not implemented yet."
        )