from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
    
    @abstractmethod
    async def validate_input(self, input_content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input content before generation."""
        pass
    
    @abstractmethod
    async def optimize_video(self, video_id: str, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an existing video."""
        pass
    
    @abstractmethod
    async def export_video(self, video_id: str, format: str, quality: str) -> str:
        """Export video in specified format and quality."""
        pass 