from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime
import asyncio

from .base_agent import BaseAgent
from ..core.config import SystemConfig
from ..core.interfaces.coordination import ICoordinationSystem
from ..core.interfaces.agent import IAgent

class CoordinationAgent(BaseAgent):
    """Agent responsible for coordinating other agents in the system."""
    
    def __init__(self, 
                 agent_id: str,
                 config: SystemConfig,
                 coordination_system: ICoordinationSystem,
                 expertise: Optional[Dict[str, float]] = None):
        super().__init__(
            agent_id=agent_id,
            role="coordinator",
            config=config,
            expertise=expertise or {
                'task_coordination': 0.95,
                'conflict_resolution': 0.9,
                'resource_management': 0.85
            }
        )
        self.coordination_system = coordination_system
        self.registered_agents: Dict[str, IAgent] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue = asyncio.Queue()
        self.processing_tasks: Set[asyncio.Task] = set()
    
    async def initialize(self) -> None:
        """Initialize the coordination agent."""
        await super().initialize()
        # Start the background processing task
        self.processing_tasks.add(
            asyncio.create_task(self._process_task_queue())
        )
    
    async def shutdown(self) -> None:
        """Clean up resources and stop processing."""
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        self.processing_tasks.clear()
        
        # Unregister all agents
        for agent_id in list(self.registered_agents.keys()):
            await self.unregister_agent(agent_id)
        
        await super().shutdown()
    
    async def _execute_task_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement coordination task execution."""
        try:
            # Add task to coordination queue
            await self.task_queue.put(task)
            
            return {
                'success': True,
                'message': 'Task queued for coordination',
                'task_id': task.get('task_id', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in task coordination: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_type': 'coordination'
            }
    
    async def _process_task_queue(self) -> None:
        """Process tasks from the coordination queue."""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Coordinate task execution
                result = await self._coordinate_task(task)
                
                # Process result
                await self._handle_coordination_result(result)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing coordination queue: {str(e)}")
    
    async def _coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate the execution of a task across multiple agents."""
        try:
            task_id = task.get('task_id', '')
            self.active_tasks[task_id] = task
            
            # Get required agent types
            required_agents = task.get('required_agents', [])
            
            # Check agent availability
            available_agents = await self._get_available_agents(required_agents)
            if not available_agents:
                return {
                    'success': False,
                    'error': 'Required agents not available',
                    'task_id': task_id
                }
            
            # Coordinate task execution
            result = await self.coordination_system.coordinate_agents(
                agents=available_agents,
                task=task
            )
            
            return {
                'success': True,
                'result': result,
                'task_id': task_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating task: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task.get('task_id', '')
            }
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _get_available_agents(self, required_types: List[str]) -> List[IAgent]:
        """Get available agents of required types."""
        available_agents = []
        for agent_type in required_types:
            for agent in self.registered_agents.values():
                if agent.role == agent_type and await self._is_agent_available(agent):
                    available_agents.append(agent)
        return available_agents
    
    async def _is_agent_available(self, agent: IAgent) -> bool:
        """Check if an agent is available for new tasks."""
        status = await self.coordination_system.get_agent_status(agent.agent_id)
        return status.get('status') == 'available'
    
    async def _handle_coordination_result(self, result: Dict[str, Any]) -> None:
        """Handle the result of task coordination."""
        if result['success']:
            # Store metrics
            self.metrics['total_tasks'] += 1
            self.metrics['successful_tasks'] += 1
            
            # Get coordination metrics
            metrics = await self.coordination_system.get_coordination_metrics()
            self.metrics['coordination_metrics'] = metrics
            
        else:
            self.metrics['total_tasks'] += 1
            self.metrics['failed_tasks'] += 1
    
    async def register_agent(self, agent: IAgent) -> Dict[str, Any]:
        """Register a new agent with the coordination system."""
        try:
            await self.coordination_system.register_agent(agent)
            self.registered_agents[agent.agent_id] = agent
            return {
                'success': True,
                'message': f'Agent {agent.agent_id} registered successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error registering agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def unregister_agent(self, agent_id: str) -> Dict[str, Any]:
        """Unregister an agent from the coordination system."""
        try:
            if agent_id in self.registered_agents:
                await self.coordination_system.unregister_agent(agent_id)
                del self.registered_agents[agent_id]
                return {
                    'success': True,
                    'message': f'Agent {agent_id} unregistered successfully',
                    'timestamp': datetime.now().isoformat()
                }
            return {
                'success': False,
                'error': f'Agent {agent_id} not found'
            }
        except Exception as e:
            self.logger.error(f"Error unregistering agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def broadcast_message(self, message: Dict[str, Any], target_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Broadcast a message to all or specific agents."""
        try:
            result = await self.coordination_system.broadcast_message(
                message=message,
                target_agents=target_agents
            )
            return {
                'success': True,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 