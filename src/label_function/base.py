import logging
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

class LFType(Enum):
    SURFACE = "surface"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"

class BaseLF(ABC):
    def __init__(self, heuristic_path: Path):
        self.heuristic_path = heuristic_path
        self.heuristic_path.mkdir(parents=True, exist_ok=True)
        self.saved_path: Optional[Path] = None

    def get_saved_path(self) -> Optional[Path]:
        if self.saved_path is None:
            raise ValueError("The heuristic is not saved.")
        
        return self.saved_path
    
    def _find_saved_path(self):
        idx = 0
        saved_path = self.heuristic_path / f"heuristic_{idx}.py"
        while saved_path.exists():
            idx += 1
            saved_path = self.heuristic_path / f"heuristic_{idx}.py"
        
        return saved_path

    @abstractmethod
    def save(self) -> Path:
        pass