from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IAgent(ABC):
    """Interface for all agents in the system."""
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a given task."""
        pass
    
    @abstractmethod
    async def collaborate(self, other_agents: List['IAgent'], task: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents on a task."""
        pass
    
    @abstractmethod
    def get_expertise(self) -> Dict[str, float]:
        """Get agent's expertise levels in different areas."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with necessary resources."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        pass 