from typing import Any, Dict, List, Optional
import logging
import uuid
from datetime import datetime

from ..core.interfaces import IAgent
from ..core.config import SystemConfig

class BaseAgent(IAgent):
    """Base implementation of an agent with common functionality."""
    
    def __init__(self, 
                 agent_id: str,
                 role: str,
                 config: SystemConfig,
                 expertise: Optional[Dict[str, float]] = None):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        self.expertise = expertise or {}
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.is_initialized = False
        self.current_tasks: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """Initialize the agent with necessary resources."""
        if self.is_initialized:
            return
        
        self.logger.info(f"Initializing agent {self.agent_id} with role {self.role}")
        # Add initialization logic here
        self.is_initialized = True
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        # Add cleanup logic here
        self.is_initialized = False
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a given task."""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing task {task_id}: {task.get('description', 'No description')}")
            self.current_tasks.append(task)
            
            result = await self._execute_task_impl(task)
            
            # Record performance
            execution_time = (datetime.now() - start_time).total_seconds()
            performance = {
                'task_id': task_id,
                'execution_time': execution_time,
                'success': result.get('success', False),
                'quality_score': result.get('quality_score', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            self.performance_history.append(performance)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id
            }
        finally:
            self.current_tasks.remove(task)
    
    async def collaborate(self, other_agents: List[IAgent], task: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents on a task."""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        
        self.logger.info(f"Starting collaboration with {len(other_agents)} agents")
        
        # Implement collaboration logic
        return {
            'success': True,
            'collaboration_id': str(uuid.uuid4()),
            'participants': [agent.agent_id for agent in other_agents],
            'result': {}
        }
    
    def get_expertise(self) -> Dict[str, float]:
        """Get agent's expertise levels in different areas."""
        return self.expertise
    
    async def _execute_task_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of task execution. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_task_impl")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent."""
        if not self.performance_history:
            return {
                'total_tasks': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'avg_quality_score': 0.0
            }
        
        total_tasks = len(self.performance_history)
        success_count = sum(1 for p in self.performance_history if p['success'])
        avg_execution_time = sum(p['execution_time'] for p in self.performance_history) / total_tasks
        avg_quality = sum(p['quality_score'] for p in self.performance_history) / total_tasks
        
        return {
            'total_tasks': total_tasks,
            'success_rate': success_count / total_tasks,
            'avg_execution_time': avg_execution_time,
            'avg_quality_score': avg_quality
        } 