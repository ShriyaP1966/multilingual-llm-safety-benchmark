"""
mock_provider.py

Mock provider used to test the experiment pipeline without making
any real API calls.
"""

from .base_provider import BaseProvider


class MockProvider(BaseProvider):
    """
    Mock model provider for testing.
    """

    def __init__(self, config):

        self.config = config

    def generate_response(self, prompt: str) -> str:
        """
        Return a fixed response.
        """

        return "THIS IS A TEST RESPONSE"