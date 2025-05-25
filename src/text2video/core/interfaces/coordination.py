from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..interfaces.agent import IAgent

class ICoordinationSystem(ABC):
    """Interface for the coordination system."""
    
    @abstractmethod
    async def coordinate_agents(self, agents: List[IAgent], task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for a task."""
        pass
    
    @abstractmethod
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between agents."""
        pass
    
    @abstractmethod
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get metrics about coordination performance."""
        pass
    
    @abstractmethod
    async def register_agent(self, agent: IAgent) -> None:
        """Register a new agent with the coordination system."""
        pass
    
    @abstractmethod
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordination system."""
        pass
    
    @abstractmethod
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the current status of an agent."""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Dict[str, Any], target_agents: Optional[List[str]] = None) -> None:
        """Broadcast a message to all or specific agents."""
        pass 