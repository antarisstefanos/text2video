from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from ..interfaces.agent import IAgent
from ..interfaces.memory import IMemorySystem
from ..interfaces.coordination import ICoordinationSystem
from ..interfaces.rl import IRLSystem
from ..coordination.message_bus import MessageBus, AgentMessage, MessageType
from ..coordination.collaboration import CollaborativeWorkspace, WorkItemStatus
from ..agents.coordinated_agent import CoordinatedAgent, AgentRole, AgentCapability

class VideoGenerationStatus(Enum):
    """Status of a video generation task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class VideoGenerationTask:
    """A video generation task."""
    task_id: str
    user_id: str
    input_text: str
    parameters: Dict[str, Any]
    status: VideoGenerationStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, user_id: str, input_text: str,
               parameters: Dict[str, Any], metadata: Dict[str, Any] = None) -> 'VideoGenerationTask':
        """Create a new video generation task."""
        return cls(
            task_id=str(uuid.uuid4()),
            user_id=user_id,
            input_text=input_text,
            parameters=parameters,
            status=VideoGenerationStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {}
        )

class EnhancedVideoGenerationSystem:
    """Enhanced video generation system with RL, memory, and coordination."""
    
    def __init__(self, memory_system: IMemorySystem,
                 coordination_system: ICoordinationSystem,
                 rl_system: IRLSystem,
                 message_bus: MessageBus,
                 workspace: CollaborativeWorkspace):
        self.logger = logging.getLogger("enhanced_video_system")
        self.memory_system = memory_system
        self.coordination_system = coordination_system
        self.rl_system = rl_system
        self.message_bus = message_bus
        self.workspace = workspace
        
        # Initialize agents
        self.agents: Dict[str, CoordinatedAgent] = {}
        self._initialize_agents()
        
        # Initialize state
        self.tasks: Dict[str, VideoGenerationTask] = {}
        self.performance_metrics = defaultdict(float)
    
    def _initialize_agents(self) -> None:
        """Initialize the system's agents."""
        try:
            # Create storyboard agent
            self.agents['storyboard'] = CoordinatedAgent(
                agent_id='storyboard_agent',
                role=AgentRole.STORYBOARD,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.STORYBOARD,
                    expertise_level=1.0,
                    skills={'storyboarding', 'scripting', 'scene_planning'},
                    metadata={'model': 'gpt-4'}
                )
            )
            
            # Create visualization agent
            self.agents['visualization'] = CoordinatedAgent(
                agent_id='visualization_agent',
                role=AgentRole.VISUALIZATION,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.VISUALIZATION,
                    expertise_level=1.0,
                    skills={'scene_rendering', 'style_transfer', 'composition'},
                    metadata={'model': 'stable-diffusion-xl'}
                )
            )
            
            # Create animation agent
            self.agents['animation'] = CoordinatedAgent(
                agent_id='animation_agent',
                role=AgentRole.ANIMATION,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.ANIMATION,
                    expertise_level=1.0,
                    skills={'motion_generation', 'keyframing', 'interpolation'},
                    metadata={'model': 'animatediff'}
                )
            )
            
            # Create sound agent
            self.agents['sound'] = CoordinatedAgent(
                agent_id='sound_agent',
                role=AgentRole.SOUND,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.SOUND,
                    expertise_level=1.0,
                    skills={'audio_generation', 'mixing', 'synchronization'},
                    metadata={'model': 'audiogen'}
                )
            )
            
            # Create coordinator agent
            self.agents['coordinator'] = CoordinatedAgent(
                agent_id='coordinator_agent',
                role=AgentRole.COORDINATOR,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.COORDINATOR,
                    expertise_level=1.0,
                    skills={'task_planning', 'resource_allocation', 'conflict_resolution'},
                    metadata={'model': 'gpt-4'}
                )
            )
            
            # Create quality agent
            self.agents['quality'] = CoordinatedAgent(
                agent_id='quality_agent',
                role=AgentRole.QUALITY,
                memory_system=self.memory_system,
                coordination_system=self.coordination_system,
                rl_system=self.rl_system,
                message_bus=self.message_bus,
                workspace=self.workspace,
                capabilities=AgentCapability(
                    role=AgentRole.QUALITY,
                    expertise_level=1.0,
                    skills={'quality_assessment', 'feedback_analysis', 'improvement_suggestion'},
                    metadata={'model': 'gpt-4'}
                )
            )
            
            self.logger.info("Initialized all agents")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def generate_video(self, user_id: str, input_text: str,
                      parameters: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Generate a video from text input."""
        try:
            # Create task
            task = VideoGenerationTask.create(user_id, input_text, parameters, metadata)
            self.tasks[task.task_id] = task
            
            # Create collaboration session
            session_id = str(uuid.uuid4())
            self.workspace.create_session(
                session_id=session_id,
                description=f"Video generation for task {task.task_id}",
                participants=list(self.agents.keys()),
                metadata={'task_id': task.task_id}
            )
            
            # Add work items
            self._create_work_items(session_id, task)
            
            # Start coordination
            self._coordinate_generation(session_id, task)
            
            return task.task_id
            
        except Exception as e:
            self.logger.error(f"Error generating video: {str(e)}")
            raise
    
    def _create_work_items(self, session_id: str, task: VideoGenerationTask) -> None:
        """Create work items for the video generation task."""
        try:
            # Storyboard work item
            self.workspace.add_work_item(
                session_id=session_id,
                title="Generate Storyboard",
                description=task.input_text,
                metadata={
                    'task_type': AgentRole.STORYBOARD.value,
                    'parameters': task.parameters.get('storyboard', {}),
                    'task_id': task.task_id
                }
            )
            
            # Visualization work item
            self.workspace.add_work_item(
                session_id=session_id,
                title="Generate Visuals",
                description="Generate visual elements based on storyboard",
                metadata={
                    'task_type': AgentRole.VISUALIZATION.value,
                    'parameters': task.parameters.get('visualization', {}),
                    'task_id': task.task_id
                }
            )
            
            # Animation work item
            self.workspace.add_work_item(
                session_id=session_id,
                title="Generate Animation",
                description="Animate visual elements",
                metadata={
                    'task_type': AgentRole.ANIMATION.value,
                    'parameters': task.parameters.get('animation', {}),
                    'task_id': task.task_id
                }
            )
            
            # Sound work item
            self.workspace.add_work_item(
                session_id=session_id,
                title="Generate Sound",
                description="Generate audio elements",
                metadata={
                    'task_type': AgentRole.SOUND.value,
                    'parameters': task.parameters.get('sound', {}),
                    'task_id': task.task_id
                }
            )
            
            # Quality check work item
            self.workspace.add_work_item(
                session_id=session_id,
                title="Quality Check",
                description="Assess and improve video quality",
                metadata={
                    'task_type': AgentRole.QUALITY.value,
                    'parameters': task.parameters.get('quality', {}),
                    'task_id': task.task_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error creating work items: {str(e)}")
            raise
    
    def _coordinate_generation(self, session_id: str, task: VideoGenerationTask) -> None:
        """Coordinate the video generation process."""
        try:
            # Update task status
            task.status = VideoGenerationStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            
            # Get work items
            work_items = self.workspace.get_session_work_items(session_id)
            
            # Process work items in sequence
            for item in work_items:
                if item.status == WorkItemStatus.PENDING:
                    # Get appropriate agent
                    agent = self.agents[item.metadata['task_type']]
                    
                    # Process work item
                    agent._process_work_item(item)
                    
                    # Update metrics
                    self._update_metrics(item.metadata['task_type'], {
                        'work_items_processed': 1,
                        'success_rate': 1.0 if item.status == WorkItemStatus.COMPLETED else 0.0
                    })
            
            # Check if all work items are completed
            all_completed = all(item.status == WorkItemStatus.COMPLETED
                              for item in work_items)
            
            if all_completed:
                task.status = VideoGenerationStatus.COMPLETED
            else:
                task.status = VideoGenerationStatus.FAILED
            
            task.updated_at = datetime.now()
            
            # Close session
            self.workspace.close_session(session_id)
            
        except Exception as e:
            self.logger.error(f"Error coordinating generation: {str(e)}")
            task.status = VideoGenerationStatus.FAILED
            task.updated_at = datetime.now()
            raise
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a video generation task."""
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} does not exist")
            
            task = self.tasks[task_id]
            return {
                'task_id': task.task_id,
                'user_id': task.user_id,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'updated_at': task.updated_at.isoformat(),
                'metadata': task.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            return {}
    
    def cancel_task(self, task_id: str) -> None:
        """Cancel a video generation task."""
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} does not exist")
            
            task = self.tasks[task_id]
            if task.status == VideoGenerationStatus.COMPLETED:
                raise ValueError("Cannot cancel completed task")
            
            task.status = VideoGenerationStatus.CANCELLED
            task.updated_at = datetime.now()
            
            # Close any active sessions
            for session_id, session in self.workspace.sessions.items():
                if session['metadata'].get('task_id') == task_id:
                    self.workspace.close_session(session_id)
            
            self.logger.info(f"Cancelled task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling task: {str(e)}")
            raise
    
    def _update_metrics(self, agent_type: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        try:
            for metric, value in metrics.items():
                self.performance_metrics[f'{agent_type}_{metric}'] = value
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system-wide performance metrics."""
        try:
            metrics = dict(self.performance_metrics)
            
            # Add task statistics
            task_statuses = defaultdict(int)
            for task in self.tasks.values():
                task_statuses[task.status.value] += 1
            
            metrics['total_tasks'] = len(self.tasks)
            for status, count in task_statuses.items():
                metrics[f'tasks_{status}'] = count
            
            # Add agent statistics
            for agent_id, agent in self.agents.items():
                agent_metrics = agent.get_metrics()
                for metric, value in agent_metrics.items():
                    metrics[f'agent_{agent_id}_{metric}'] = value
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {str(e)}")
            return {} 