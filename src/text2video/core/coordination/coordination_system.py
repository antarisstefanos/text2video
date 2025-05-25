from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime
import asyncio
from collections import defaultdict, deque
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

from ..interfaces.coordination import ICoordinationSystem
from ..interfaces.agent import IAgent

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    STATUS = "status"
    CONFLICT = "conflict"
    CONSENSUS = "consensus"
    BROADCAST = "broadcast"

@dataclass
class Message:
    """Message structure for the message bus."""
    type: MessageType
    sender: str
    content: Dict[str, Any]
    timestamp: datetime
    message_id: str
    recipients: Optional[Set[str]] = None
    requires_ack: bool = False
    acknowledged_by: Set[str] = None

class EnhancedCoordinationSystem(ICoordinationSystem):
    """Enhanced implementation of the coordination system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("coordination_system")
        self.config = config
        
        # Agent registry
        self.agents: Dict[str, IAgent] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        
        # Message bus
        self.message_bus = asyncio.Queue()
        self.message_history = deque(maxlen=1000)
        self.message_handlers = defaultdict(list)
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.task_history = deque(maxlen=1000)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Consensus management
        self.consensus_groups: Dict[str, Dict[str, Any]] = {}
        self.consensus_timeout = config.get('consensus_timeout', 30.0)
        
        # Collaborative workspace
        self.workspace = {
            'shared_state': {},
            'locks': defaultdict(asyncio.Lock),
            'version': 0
        }
        
        # Metrics
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'conflicts_resolved': 0,
            'consensus_achieved': 0,
            'messages_processed': 0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        asyncio.create_task(self._process_message_bus())
        asyncio.create_task(self._process_task_queue())
        asyncio.create_task(self._monitor_agent_status())
    
    async def coordinate_agents(self, task: Dict[str, Any], agents: List[IAgent]) -> Dict[str, Any]:
        """Coordinate multiple agents for a task."""
        try:
            # Generate task ID
            task_id = hashlib.md5(json.dumps(task, sort_keys=True).encode()).hexdigest()
            
            # Create task entry
            task_entry = {
                'task_id': task_id,
                'task': task,
                'agents': [agent.agent_id for agent in agents],
                'status': 'pending',
                'start_time': datetime.now(),
                'results': {},
                'consensus_required': task.get('requires_consensus', False)
            }
            
            # Add to task queue
            await self.task_queue.put(task_entry)
            self.active_tasks[task_id] = task_entry
            
            # Update metrics
            self.metrics['total_tasks'] += 1
            
            # Notify agents
            message = Message(
                type=MessageType.TASK,
                sender='coordination_system',
                content={'task_id': task_id, 'task': task},
                timestamp=datetime.now(),
                message_id=hashlib.md5(f"{task_id}_{datetime.now().isoformat()}".encode()).hexdigest(),
                recipients={agent.agent_id for agent in agents},
                requires_ack=True,
                acknowledged_by=set()
            )
            await self.message_bus.put(message)
            
            return {
                'task_id': task_id,
                'status': 'pending',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating agents: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    async def resolve_conflicts(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between agents."""
        try:
            conflict_id = conflict.get('conflict_id', hashlib.md5(json.dumps(conflict, sort_keys=True).encode()).hexdigest())
            
            # Create conflict resolution group
            self.consensus_groups[conflict_id] = {
                'conflict': conflict,
                'agents': set(conflict.get('involved_agents', [])),
                'proposals': {},
                'votes': defaultdict(int),
                'start_time': datetime.now(),
                'status': 'pending'
            }
            
            # Request proposals from involved agents
            message = Message(
                type=MessageType.CONFLICT,
                sender='coordination_system',
                content={'conflict_id': conflict_id, 'conflict': conflict},
                timestamp=datetime.now(),
                message_id=hashlib.md5(f"{conflict_id}_{datetime.now().isoformat()}".encode()).hexdigest(),
                recipients=set(conflict.get('involved_agents', [])),
                requires_ack=True,
                acknowledged_by=set()
            )
            await self.message_bus.put(message)
            
            # Wait for consensus
            try:
                await asyncio.wait_for(
                    self._wait_for_consensus(conflict_id),
                    timeout=self.consensus_timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Consensus timeout for conflict {conflict_id}")
                return {
                    'conflict_id': conflict_id,
                    'status': 'timeout',
                    'resolution': 'default',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get consensus result
            consensus = self.consensus_groups[conflict_id]
            resolution = max(consensus['votes'].items(), key=lambda x: x[1])[0]
            
            # Update metrics
            self.metrics['conflicts_resolved'] += 1
            self.metrics['consensus_achieved'] += 1
            
            return {
                'conflict_id': conflict_id,
                'status': 'resolved',
                'resolution': resolution,
                'votes': dict(consensus['votes']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error resolving conflict: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get metrics about the coordination system."""
        return {
            **self.metrics,
            'active_tasks': len(self.active_tasks),
            'active_consensus_groups': len(self.consensus_groups),
            'registered_agents': len(self.agents),
            'timestamp': datetime.now().isoformat()
        }
    
    async def register_agent(self, agent: IAgent) -> None:
        """Register an agent with the coordination system."""
        try:
            agent_id = agent.agent_id
            self.agents[agent_id] = agent
            self.agent_status[agent_id] = {
                'status': 'registered',
                'last_heartbeat': datetime.now(),
                'capabilities': agent.get_expertise()
            }
            
            # Notify other agents
            message = Message(
                type=MessageType.STATUS,
                sender='coordination_system',
                content={'agent_id': agent_id, 'status': 'registered'},
                timestamp=datetime.now(),
                message_id=hashlib.md5(f"{agent_id}_register_{datetime.now().isoformat()}".encode()).hexdigest(),
                recipients=set(self.agents.keys()) - {agent_id}
            )
            await self.message_bus.put(message)
            
        except Exception as e:
            self.logger.error(f"Error registering agent: {str(e)}")
            raise
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordination system."""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                del self.agent_status[agent_id]
                
                # Notify other agents
                message = Message(
                    type=MessageType.STATUS,
                    sender='coordination_system',
                    content={'agent_id': agent_id, 'status': 'unregistered'},
                    timestamp=datetime.now(),
                    message_id=hashlib.md5(f"{agent_id}_unregister_{datetime.now().isoformat()}".encode()).hexdigest(),
                    recipients=set(self.agents.keys())
                )
                await self.message_bus.put(message)
                
        except Exception as e:
            self.logger.error(f"Error unregistering agent: {str(e)}")
            raise
    
    async def broadcast_message(self, message: Dict[str, Any], sender: str) -> None:
        """Broadcast a message to all agents."""
        try:
            broadcast_message = Message(
                type=MessageType.BROADCAST,
                sender=sender,
                content=message,
                timestamp=datetime.now(),
                message_id=hashlib.md5(f"{sender}_broadcast_{datetime.now().isoformat()}".encode()).hexdigest(),
                recipients=set(self.agents.keys())
            )
            await self.message_bus.put(broadcast_message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {str(e)}")
            raise
    
    async def _process_message_bus(self) -> None:
        """Process messages from the message bus."""
        while True:
            try:
                message = await self.message_bus.get()
                
                # Store in history
                self.message_history.append(message)
                
                # Process message
                if message.recipients:
                    for agent_id in message.recipients:
                        if agent_id in self.agents:
                            agent = self.agents[agent_id]
                            await agent.handle_message(message)
                            
                            if message.requires_ack:
                                message.acknowledged_by.add(agent_id)
                
                # Update metrics
                self.metrics['messages_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
    
    async def _process_task_queue(self) -> None:
        """Process tasks from the task queue."""
        while True:
            try:
                task_entry = await self.task_queue.get()
                task_id = task_entry['task_id']
                
                # Execute task with agents
                results = await asyncio.gather(*[
                    self.agents[agent_id].execute_task(task_entry['task'])
                    for agent_id in task_entry['agents']
                    if agent_id in self.agents
                ])
                
                # Update task status
                task_entry['results'] = {
                    agent_id: result
                    for agent_id, result in zip(task_entry['agents'], results)
                }
                task_entry['status'] = 'completed'
                task_entry['end_time'] = datetime.now()
                
                # Check if consensus is required
                if task_entry['consensus_required']:
                    consensus_result = await self._achieve_consensus(task_id, results)
                    task_entry['consensus'] = consensus_result
                
                # Update metrics
                if all(result.get('success', False) for result in results):
                    self.metrics['successful_tasks'] += 1
                else:
                    self.metrics['failed_tasks'] += 1
                
                # Remove from active tasks
                del self.active_tasks[task_id]
                
            except Exception as e:
                self.logger.error(f"Error processing task: {str(e)}")
    
    async def _monitor_agent_status(self) -> None:
        """Monitor agent status and handle timeouts."""
        while True:
            try:
                current_time = datetime.now()
                
                # Check agent heartbeats
                for agent_id, status in self.agent_status.items():
                    if (current_time - status['last_heartbeat']).total_seconds() > self.config.get('agent_timeout', 60):
                        self.logger.warning(f"Agent {agent_id} timeout")
                        await self.unregister_agent(agent_id)
                
                await asyncio.sleep(self.config.get('monitor_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Error monitoring agent status: {str(e)}")
    
    async def _wait_for_consensus(self, conflict_id: str) -> None:
        """Wait for consensus to be achieved in a conflict resolution group."""
        while True:
            consensus = self.consensus_groups[conflict_id]
            
            # Check if all agents have voted
            if len(consensus['votes']) == len(consensus['agents']):
                # Check if there's a clear winner
                max_votes = max(consensus['votes'].values())
                winners = [k for k, v in consensus['votes'].items() if v == max_votes]
                
                if len(winners) == 1:
                    consensus['status'] = 'achieved'
                    return
            
            await asyncio.sleep(0.1)
    
    async def _achieve_consensus(self, task_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Achieve consensus among agents for a task result."""
        try:
            # Group results by type
            result_groups = defaultdict(list)
            for result in results:
                result_type = result.get('type', 'unknown')
                result_groups[result_type].append(result)
            
            # Find most common result type
            most_common_type = max(result_groups.items(), key=lambda x: len(x[1]))[0]
            
            # Calculate average for numeric values
            consensus_result = {
                'type': most_common_type,
                'values': {}
            }
            
            # Get all numeric fields
            numeric_fields = set()
            for result in result_groups[most_common_type]:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        numeric_fields.add(key)
            
            # Calculate averages
            for field in numeric_fields:
                values = [r[field] for r in result_groups[most_common_type] if field in r]
                if values:
                    consensus_result['values'][field] = sum(values) / len(values)
            
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"Error achieving consensus: {str(e)}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    async def _update_workspace(self, update: Dict[str, Any], agent_id: str) -> None:
        """Update the collaborative workspace."""
        try:
            async with self.workspace['locks'][agent_id]:
                # Apply update
                for key, value in update.items():
                    self.workspace['shared_state'][key] = value
                
                # Increment version
                self.workspace['version'] += 1
                
                # Notify other agents
                message = Message(
                    type=MessageType.STATUS,
                    sender='coordination_system',
                    content={
                        'workspace_update': True,
                        'version': self.workspace['version'],
                        'changes': update
                    },
                    timestamp=datetime.now(),
                    message_id=hashlib.md5(f"workspace_update_{self.workspace['version']}".encode()).hexdigest(),
                    recipients=set(self.agents.keys()) - {agent_id}
                )
                await self.message_bus.put(message)
                
        except Exception as e:
            self.logger.error(f"Error updating workspace: {str(e)}")
            raise 