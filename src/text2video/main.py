import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import os
import sys
from pathlib import Path

from .core.memory.memory_system import EnhancedMemorySystem
from .core.coordination.message_bus import MessageBus
from .core.coordination.collaboration import CollaborativeWorkspace
from .core.rl.rl_system import RLSystem
from .core.video.enhanced_video_system import EnhancedVideoGenerationSystem

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=os.getenv("TEXT2VIDEO_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemConfig:
    """System configuration."""
    def __init__(self):
        self.openai_api_key = os.getenv("TEXT2VIDEO_OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("TEXT2VIDEO_PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("TEXT2VIDEO_PINECONE_ENVIRONMENT")
        self.comfyui_server_url = os.getenv("TEXT2VIDEO_COMFYUI_SERVER_URL", "http://localhost:8188")
        self.redis_url = os.getenv("TEXT2VIDEO_REDIS_URL", "redis://localhost:6379/0")
        self.stable_diffusion_model = "runwayml/stable-diffusion-v1-5"
        self.max_video_duration = 300
        self.quality_threshold = 0.8
        self.ethical_threshold = 0.9
        self.memory_db_path = "agent_memory.db"
        self.enable_rl = True
        self.enable_coordination = True
        self.environment = os.getenv("TEXT2VIDEO_ENVIRONMENT", "development")

class SystemHealth:
    """System health monitoring."""
    def __init__(self):
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()
        self.components_status = {
            'memory_system': False,
            'coordination_system': False,
            'rl_system': False,
            'video_system': False
        }
        self.error_count = 0
        self.warning_count = 0

    def update_component_status(self, component: str, status: bool):
        """Update component health status."""
        self.components_status[component] = status
        self.last_health_check = datetime.now()

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'status': 'healthy' if all(self.components_status.values()) else 'unhealthy',
            'uptime': str(datetime.now() - self.start_time),
            'last_check': self.last_health_check.isoformat(),
            'components': self.components_status,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }

async def main():
    """Enhanced example usage with all features."""
    
    # Initialize health monitoring
    health = SystemHealth()
    
    try:
        # Configuration
        config = SystemConfig()
        
        # Initialize core components
        logger.info("Initializing core components...")
        memory_system = EnhancedMemorySystem(config)
        health.update_component_status('memory_system', True)
        
        message_bus = MessageBus()
        workspace = CollaborativeWorkspace()
        health.update_component_status('coordination_system', True)
        
        rl_system = RLSystem(config)
        health.update_component_status('rl_system', True)
        
        # Initialize enhanced system
        system = EnhancedVideoGenerationSystem(
            memory_system=memory_system,
            coordination_system=None,  # Will be initialized by the system
            rl_system=rl_system,
            message_bus=message_bus,
            workspace=workspace
        )
        health.update_component_status('video_system', True)
        
        logger.info("🚀 Enhanced Video Generation System Started")
        logger.info("Features: RL Learning ✓, Long-term Memory ✓, Multi-Agent Coordination ✓")
        
        # Example 1: Basic video generation
        logger.info("\n📹 Generating video...")
        result1 = await system.generate_video(
            user_id="user_123",
            input_text="Create a video about renewable energy solutions with focus on solar power",
            parameters={
                'storyboard': {'style': 'educational', 'duration': 120},
                'visualization': {'resolution': '1920x1080', 'style': 'realistic'},
                'animation': {'smoothness': 'high', 'transition_style': 'fade'},
                'sound': {'background_music': True, 'voice_over': True},
                'quality': {'min_quality_score': 0.8}
            },
            metadata={'project_type': 'educational', 'target_audience': 'general'}
        )
        
        logger.info(f"Generation Result: {result1}")
        
        # Get task status
        task_status = system.get_task_status(result1)
        logger.info(f"Task Status: {task_status}")
        
        # Example 2: Generate another video with different parameters
        logger.info("\n📹 Generating second video...")
        result2 = await system.generate_video(
            user_id="user_123",
            input_text="Create a video about sustainable transportation focusing on electric vehicles",
            parameters={
                'storyboard': {'style': 'documentary', 'duration': 180},
                'visualization': {'resolution': '4k', 'style': 'cinematic'},
                'animation': {'smoothness': 'ultra', 'transition_style': 'dynamic'},
                'sound': {'background_music': True, 'voice_over': True, 'ambient_sounds': True},
                'quality': {'min_quality_score': 0.9}
            },
            metadata={'project_type': 'documentary', 'target_audience': 'professionals'}
        )
        
        logger.info(f"Second Generation Result: {result2}")
        
        # Example 3: Get system metrics
        logger.info("\n📊 System Performance Insights:")
        metrics = system.get_system_metrics()
        
        logger.info(f"Total Tasks: {metrics.get('total_tasks', 0)}")
        logger.info(f"Tasks Completed: {metrics.get('tasks_completed', 0)}")
        logger.info(f"Tasks Failed: {metrics.get('tasks_failed', 0)}")
        
        # Example 4: Cancel a task
        logger.info("\n❌ Cancelling task...")
        system.cancel_task(result2)
        
        # Example 5: Test complex task
        logger.info("\n🤝 Testing complex coordinated task...")
        complex_result = await system.generate_video(
            user_id="user_456",
            input_text="Create a comprehensive video series about climate change including data visualization, expert interviews, and actionable solutions",
            parameters={
                'storyboard': {'style': 'series', 'duration': 600, 'episodes': 3},
                'visualization': {'resolution': '4k', 'style': 'professional'},
                'animation': {'smoothness': 'ultra', 'transition_style': 'dynamic'},
                'sound': {'background_music': True, 'voice_over': True, 'ambient_sounds': True},
                'quality': {'min_quality_score': 0.95}
            },
            metadata={
                'project_type': 'series',
                'target_audience': 'professionals',
                'complexity': 'high',
                'requires_coordination': True
            }
        )
        
        logger.info(f"Complex Task Result: {complex_result}")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        health.error_count += 1
        # Update component status based on error
        if "memory" in str(e).lower():
            health.update_component_status('memory_system', False)
        elif "coordination" in str(e).lower():
            health.update_component_status('coordination_system', False)
        elif "rl" in str(e).lower():
            health.update_component_status('rl_system', False)
        elif "video" in str(e).lower():
            health.update_component_status('video_system', False)
    
    finally:
        # Shutdown system
        logger.info("\n🔄 Shutting down system...")
        # Add cleanup code here if needed
        logger.info("✅ System shutdown complete")
        
        # Return health status
        return health.get_health_status()

if __name__ == "__main__":
    # Run demonstration
    health_status = asyncio.run(main())
    print("\nSystem Health Status:")
    print(health_status) 