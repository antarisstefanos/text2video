from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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

class IRLSystem(ABC):
    """Interface for reinforcement learning system."""
    
    @abstractmethod
    async def update_model(self, experience: Dict[str, Any]) -> None:
        """Update the RL model with new experience."""
        pass
    
    @abstractmethod
    async def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best action for a given state."""
        pass
    
    @abstractmethod
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics about the RL model's performance."""
        pass

class IVideoGenerator(ABC):
    """Interface for video generation system."""
    
    @abstractmethod
    async def generate_video(self, 
                           input_content: Dict[str, Any],
                           user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a video from input content."""
        pass
    
    @abstractmethod
    async def process_feedback(self, 
                             video_id: str,
                             feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback for a generated video."""
        pass
    
    @abstractmethod
    def get_generation_metrics(self) -> Dict[str, Any]:
        """Get metrics about video generation performance."""
        pass 