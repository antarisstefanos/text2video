from typing import Any, Dict, List, Optional, Set, Callable
import logging
from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    BROADCAST = "broadcast"

@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, sender_id: str, recipient_id: Optional[str],
               message_type: MessageType, content: Dict[str, Any],
               metadata: Dict[str, Any] = None) -> 'AgentMessage':
        """Create a new message."""
        return cls(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            message_type=MessageType(data['message_type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data['metadata']
        )

class MessageBus:
    """Centralized message bus for agent communication."""
    
    def __init__(self):
        self.logger = logging.getLogger("message_bus")
        self.subscribers: Dict[str, Set[Callable[[AgentMessage], None]]] = defaultdict(set)
        self.message_history: List[AgentMessage] = []
        self.conversation_tracking: Dict[str, List[AgentMessage]] = defaultdict(list)
        self.max_history_size = 1000
    
    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe an agent to receive messages."""
        self.subscribers[agent_id].add(callback)
        self.logger.info(f"Agent {agent_id} subscribed to message bus")
    
    def unsubscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]) -> None:
        """Unsubscribe an agent from receiving messages."""
        if agent_id in self.subscribers:
            self.subscribers[agent_id].discard(callback)
            if not self.subscribers[agent_id]:
                del self.subscribers[agent_id]
            self.logger.info(f"Agent {agent_id} unsubscribed from message bus")
    
    def send_message(self, message: AgentMessage) -> None:
        """Send a message to the specified recipient or broadcast to all subscribers."""
        try:
            # Add to message history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)
            
            # Track conversation
            if message.recipient_id:
                self.conversation_tracking[message.recipient_id].append(message)
            
            # Deliver message
            if message.recipient_id:
                # Direct message
                if message.recipient_id in self.subscribers:
                    for callback in self.subscribers[message.recipient_id]:
                        try:
                            callback(message)
                        except Exception as e:
                            self.logger.error(f"Error in message callback: {str(e)}")
            else:
                # Broadcast message
                for agent_id, callbacks in self.subscribers.items():
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            self.logger.error(f"Error in broadcast callback: {str(e)}")
            
            self.logger.info(f"Message {message.message_id} sent from {message.sender_id} "
                           f"to {message.recipient_id or 'all'}")
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            raise
    
    def get_message_history(self, agent_id: Optional[str] = None,
                           message_type: Optional[MessageType] = None,
                           limit: int = 100) -> List[AgentMessage]:
        """Get message history filtered by agent and/or message type."""
        try:
            messages = self.message_history
            
            if agent_id:
                messages = [m for m in messages if m.sender_id == agent_id or
                          m.recipient_id == agent_id]
            
            if message_type:
                messages = [m for m in messages if m.message_type == message_type]
            
            return messages[-limit:]
            
        except Exception as e:
            self.logger.error(f"Error getting message history: {str(e)}")
            return []
    
    def get_conversation_history(self, agent_id: str,
                               limit: int = 100) -> List[AgentMessage]:
        """Get conversation history for a specific agent."""
        try:
            return self.conversation_tracking[agent_id][-limit:]
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_history(self, agent_id: Optional[str] = None) -> None:
        """Clear message history for a specific agent or all agents."""
        try:
            if agent_id:
                self.message_history = [m for m in self.message_history
                                      if m.sender_id != agent_id and
                                      m.recipient_id != agent_id]
                if agent_id in self.conversation_tracking:
                    del self.conversation_tracking[agent_id]
            else:
                self.message_history.clear()
                self.conversation_tracking.clear()
            
            self.logger.info(f"Message history cleared for {agent_id or 'all agents'}")
            
        except Exception as e:
            self.logger.error(f"Error clearing message history: {str(e)}")
            raise
    
    def get_active_agents(self) -> List[str]:
        """Get list of agents currently subscribed to the message bus."""
        return list(self.subscribers.keys())
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get statistics about message traffic."""
        try:
            stats = {
                'total_messages': len(self.message_history),
                'active_agents': len(self.subscribers),
                'message_types': defaultdict(int),
                'messages_per_agent': defaultdict(int)
            }
            
            for message in self.message_history:
                stats['message_types'][message.message_type.value] += 1
                stats['messages_per_agent'][message.sender_id] += 1
                if message.recipient_id:
                    stats['messages_per_agent'][message.recipient_id] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting message stats: {str(e)}")
            return {} 