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

class AgentRole(Enum):
    """Roles that an agent can take in the system."""
    STORYBOARD = "storyboard"
    VISUALIZATION = "visualization"
    ANIMATION = "animation"
    SOUND = "sound"
    COORDINATOR = "coordinator"
    QUALITY = "quality"

@dataclass
class AgentCapability:
    """Capabilities of an agent."""
    role: AgentRole
    expertise_level: float
    skills: Set[str]
    metadata: Dict[str, Any]

class CoordinatedAgent(IAgent):
    """An agent that can coordinate with other agents."""
    
    def __init__(self, agent_id: str, role: AgentRole,
                 memory_system: IMemorySystem,
                 coordination_system: ICoordinationSystem,
                 rl_system: IRLSystem,
                 message_bus: MessageBus,
                 workspace: CollaborativeWorkspace,
                 capabilities: Optional[AgentCapability] = None):
        self.logger = logging.getLogger(f"coordinated_agent_{agent_id}")
        self.agent_id = agent_id
        self.role = role
        self.memory_system = memory_system
        self.coordination_system = coordination_system
        self.rl_system = rl_system
        self.message_bus = message_bus
        self.workspace = workspace
        self.capabilities = capabilities or AgentCapability(
            role=role,
            expertise_level=1.0,
            skills=set(),
            metadata={}
        )
        
        # Subscribe to message bus
        self.message_bus.subscribe(agent_id, self._handle_message)
        
        # Initialize state
        self.current_task = None
        self.current_session = None
        self.performance_metrics = defaultdict(float)
    
    def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming messages."""
        try:
            if message.message_type == MessageType.TASK_REQUEST:
                self._handle_task_request(message)
            elif message.message_type == MessageType.COLLABORATION_REQUEST:
                self._handle_collaboration_request(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                self._handle_status_update(message)
            elif message.message_type == MessageType.ERROR:
                self._handle_error(message)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self._send_error_message(str(e))
    
    def _handle_task_request(self, message: AgentMessage) -> None:
        """Handle a task request."""
        try:
            task = message.content.get('task')
            if not task:
                raise ValueError("No task provided in message")
            
            # Check if we can handle this task
            if not self._can_handle_task(task):
                self._send_error_message(f"Cannot handle task type: {task.get('type')}")
                return
            
            # Accept task
            self.current_task = task
            self._send_status_update("accepted_task", {
                'task_id': task.get('id'),
                'status': 'in_progress'
            })
            
            # Execute task
            result = self.execute_task(task)
            
            # Send response
            self._send_task_response(result)
            
        except Exception as e:
            self.logger.error(f"Error handling task request: {str(e)}")
            self._send_error_message(str(e))
    
    def _handle_collaboration_request(self, message: AgentMessage) -> None:
        """Handle a collaboration request."""
        try:
            session_id = message.content.get('session_id')
            if not session_id:
                raise ValueError("No session ID provided in message")
            
            # Join collaboration session
            self.current_session = session_id
            self._send_status_update("joined_session", {
                'session_id': session_id,
                'status': 'active'
            })
            
            # Process work items
            work_items = self.workspace.get_session_work_items(session_id)
            for item in work_items:
                if item.status == WorkItemStatus.PENDING:
                    self._process_work_item(item)
            
        except Exception as e:
            self.logger.error(f"Error handling collaboration request: {str(e)}")
            self._send_error_message(str(e))
    
    def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle a status update."""
        try:
            status = message.content.get('status')
            if not status:
                raise ValueError("No status provided in message")
            
            # Update our state based on status
            if status == 'task_completed':
                self.current_task = None
            elif status == 'session_closed':
                self.current_session = None
            
            # Update metrics
            self._update_metrics(message.content)
            
        except Exception as e:
            self.logger.error(f"Error handling status update: {str(e)}")
    
    def _handle_error(self, message: AgentMessage) -> None:
        """Handle an error message."""
        try:
            error = message.content.get('error')
            if not error:
                raise ValueError("No error provided in message")
            
            # Log error
            self.logger.error(f"Received error: {error}")
            
            # Update metrics
            self.performance_metrics['error_count'] += 1
            
            # Try to recover if possible
            self._attempt_recovery(error)
            
        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}")
    
    def _can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if this agent can handle a task."""
        try:
            task_type = task.get('type')
            required_skills = task.get('required_skills', set())
            
            # Check role
            if task_type != self.role.value:
                return False
            
            # Check skills
            if not required_skills.issubset(self.capabilities.skills):
                return False
            
            # Check expertise level
            if task.get('min_expertise', 0) > self.capabilities.expertise_level:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking task capability: {str(e)}")
            return False
    
    def _process_work_item(self, work_item: Any) -> None:
        """Process a work item in the collaboration session."""
        try:
            # Update status
            self.workspace.update_work_item(
                work_item.item_id,
                WorkItemStatus.IN_PROGRESS,
                assigned_agent=self.agent_id
            )
            
            # Process item
            result = self.execute_task({
                'type': work_item.metadata.get('task_type'),
                'content': work_item.description,
                'metadata': work_item.metadata
            })
            
            # Update status
            self.workspace.update_work_item(
                work_item.item_id,
                WorkItemStatus.COMPLETED,
                metadata={'result': result}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing work item: {str(e)}")
            self.workspace.update_work_item(
                work_item.item_id,
                WorkItemStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    def _attempt_recovery(self, error: str) -> None:
        """Attempt to recover from an error."""
        try:
            # Log recovery attempt
            self.logger.info(f"Attempting recovery from error: {error}")
            
            # Update metrics
            self.performance_metrics['recovery_attempts'] += 1
            
            # Try to get help from other agents
            self._send_collaboration_request({
                'error': error,
                'context': {
                    'current_task': self.current_task,
                    'current_session': self.current_session
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error in recovery attempt: {str(e)}")
    
    def _update_metrics(self, data: Dict[str, Any]) -> None:
        """Update performance metrics."""
        try:
            # Update task metrics
            if 'task_metrics' in data:
                for metric, value in data['task_metrics'].items():
                    self.performance_metrics[f'task_{metric}'] = value
            
            # Update collaboration metrics
            if 'collaboration_metrics' in data:
                for metric, value in data['collaboration_metrics'].items():
                    self.performance_metrics[f'collaboration_{metric}'] = value
            
            # Update quality metrics
            if 'quality_metrics' in data:
                for metric, value in data['quality_metrics'].items():
                    self.performance_metrics[f'quality_{metric}'] = value
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def _send_message(self, recipient_id: Optional[str], message_type: MessageType,
                     content: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        """Send a message through the message bus."""
        try:
            message = AgentMessage.create(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=content,
                metadata=metadata
            )
            self.message_bus.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
    
    def _send_task_response(self, result: Dict[str, Any]) -> None:
        """Send a task response."""
        self._send_message(
            recipient_id=self.current_task.get('requester_id'),
            message_type=MessageType.TASK_RESPONSE,
            content={'result': result}
        )
    
    def _send_status_update(self, status: str, data: Dict[str, Any]) -> None:
        """Send a status update."""
        self._send_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.STATUS_UPDATE,
            content={'status': status, **data}
        )
    
    def _send_error_message(self, error: str) -> None:
        """Send an error message."""
        self._send_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.ERROR,
            content={'error': error}
        )
    
    def _send_collaboration_request(self, data: Dict[str, Any]) -> None:
        """Send a collaboration request."""
        self._send_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.COLLABORATION_REQUEST,
            content=data
        )
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task."""
        try:
            # Validate task
            if not self._can_handle_task(task):
                raise ValueError(f"Cannot handle task type: {task.get('type')}")
            
            # Get task parameters
            task_type = task.get('type')
            content = task.get('content')
            metadata = task.get('metadata', {})
            
            # Execute based on role
            if task_type == AgentRole.STORYBOARD.value:
                result = self._execute_storyboard_task(content, metadata)
            elif task_type == AgentRole.VISUALIZATION.value:
                result = self._execute_visualization_task(content, metadata)
            elif task_type == AgentRole.ANIMATION.value:
                result = self._execute_animation_task(content, metadata)
            elif task_type == AgentRole.SOUND.value:
                result = self._execute_sound_task(content, metadata)
            elif task_type == AgentRole.COORDINATOR.value:
                result = self._execute_coordination_task(content, metadata)
            elif task_type == AgentRole.QUALITY.value:
                result = self._execute_quality_task(content, metadata)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Store experience
            self.memory_system.store_experience({
                'type': task_type,
                'content': content,
                'result': result,
                'metadata': metadata
            })
            
            # Update metrics
            self.performance_metrics['tasks_completed'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            self.performance_metrics['task_errors'] += 1
            raise
    
    def _execute_storyboard_task(self, content: Dict[str, Any],
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a storyboard task."""
        # Implementation specific to storyboard tasks
        raise NotImplementedError()
    
    def _execute_visualization_task(self, content: Dict[str, Any],
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a visualization task."""
        # Implementation specific to visualization tasks
        raise NotImplementedError()
    
    def _execute_animation_task(self, content: Dict[str, Any],
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an animation task."""
        # Implementation specific to animation tasks
        raise NotImplementedError()
    
    def _execute_sound_task(self, content: Dict[str, Any],
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sound task."""
        # Implementation specific to sound tasks
        raise NotImplementedError()
    
    def _execute_coordination_task(self, content: Dict[str, Any],
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coordination task."""
        # Implementation specific to coordination tasks
        raise NotImplementedError()
    
    def _execute_quality_task(self, content: Dict[str, Any],
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a quality task."""
        # Implementation specific to quality tasks
        raise NotImplementedError()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return dict(self.performance_metrics)
    
    def update_capabilities(self, capabilities: AgentCapability) -> None:
        """Update agent capabilities."""
        self.capabilities = capabilities
        self._send_status_update("capabilities_updated", {
            'capabilities': {
                'role': capabilities.role.value,
                'expertise_level': capabilities.expertise_level,
                'skills': list(capabilities.skills),
                'metadata': capabilities.metadata
            }
        }) 