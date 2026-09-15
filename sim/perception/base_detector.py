from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDetector(ABC):
    """Base class for all perception detectors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        raise NotImplementedError
