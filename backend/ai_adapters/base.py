from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path


class BaseAIAdapter(ABC):
    """
    Abstract Base Class for all AI modules in TruthLens.
    Ensures modularity and plug-and-play architecture.
    """
    
    @abstractmethod
    async def analyze(self, file_path: Union[str, Path], evidence_id: str) -> Dict[str, Any]:
        """
        Analyzes the media file and returns a structured dictionary
        conforming to the module's frozen JSON contract.
        """
        pass
