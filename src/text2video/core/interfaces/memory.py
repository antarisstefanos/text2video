from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class IMemorySystem(ABC):
    """Interface for the memory system."""
    
    @abstractmethod
    async def store_experience(self, experience: Dict[str, Any]) -> str:
        """Store an experience in memory."""
        pass
    
    @abstractmethod
    async def retrieve_relevant_memories(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the given context."""
        pass
    
    @abstractmethod
    async def consolidate_memories(self) -> None:
        """Consolidate and optimize stored memories."""
        pass
    
    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        pass
    
    @abstractmethod
    async def clear_memory(self, memory_type: Optional[str] = None) -> None:
        """Clear specific or all memories."""
        pass
    
    @abstractmethod
    async def backup_memory(self, backup_path: str) -> None:
        """Create a backup of the memory system."""
        pass 