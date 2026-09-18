"""
base_provider.py

Abstract base class for all model providers.
"""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Abstract base class that every model provider must inherit.
    """

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        str
            Model response.
        """
        pass