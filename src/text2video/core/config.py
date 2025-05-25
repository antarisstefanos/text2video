from pydantic import BaseModel, Field
from typing import Optional

class SystemConfig(BaseModel):
    """System configuration using Pydantic."""
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_environment: str = Field(..., description="Pinecone environment")
    
    # Service URLs
    comfyui_server_url: str = Field(
        default="http://localhost:8188",
        description="ComfyUI server URL"
    )
    
    # Model Settings
    stable_diffusion_model: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="Stable Diffusion model to use"
    )
    
    # System Limits
    max_video_duration: int = Field(
        default=300,
        description="Maximum video duration in seconds"
    )
    
    # Quality Thresholds
    quality_threshold: float = Field(
        default=0.8,
        description="Minimum quality threshold for generated content"
    )
    ethical_threshold: float = Field(
        default=0.9,
        description="Minimum ethical compliance threshold"
    )
    
    # Storage Settings
    memory_db_path: str = Field(
        default="agent_memory.db",
        description="Path to SQLite memory database"
    )
    
    # Feature Flags
    enable_rl: bool = Field(
        default=True,
        description="Enable reinforcement learning"
    )
    enable_coordination: bool = Field(
        default=True,
        description="Enable multi-agent coordination"
    )
    
    # Optional Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    max_agents: int = Field(
        default=10,
        description="Maximum number of concurrent agents"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching models and data"
    )
    
    class Config:
        env_prefix = "TEXT2VIDEO_"
        case_sensitive = False 