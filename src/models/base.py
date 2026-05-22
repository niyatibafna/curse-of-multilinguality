from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingModel(ABC):
    """Base interface for models that encode text inputs."""

    @abstractmethod
    def encode(self, inputs: list[str], **kwargs: Any) -> Any:
        pass
