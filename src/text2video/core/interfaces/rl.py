from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the model on a batch of data."""
        pass
    
    @abstractmethod
    async def save_model(self, path: str) -> None:
        """Save the current model state."""
        pass
    
    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load a model from disk."""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        pass 