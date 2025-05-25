from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class WorkItemStatus(Enum):
    """Status of a work item in the collaborative workspace."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class WorkItem:
    """A work item in the collaborative workspace."""
    item_id: str
    title: str
    description: str
    assigned_agent: Optional[str]
    status: WorkItemStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, title: str, description: str,
               metadata: Dict[str, Any] = None) -> 'WorkItem':
        """Create a new work item."""
        return cls(
            item_id=str(uuid.uuid4()),
            title=title,
            description=description,
            assigned_agent=None,
            status=WorkItemStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {}
        )

class ConsensusManager:
    """Manages consensus among agents for collaborative decisions."""
    
    def __init__(self):
        self.logger = logging.getLogger("consensus_manager")
        self.decisions: Dict[str, Dict[str, Any]] = {}
        self.votes: Dict[str, Dict[str, Any]] = {}
        self.consensus_threshold = 0.75  # 75% agreement required
    
    def create_decision(self, decision_id: str, description: str,
                       options: List[str], metadata: Dict[str, Any] = None) -> None:
        """Create a new decision to be voted on."""
        try:
            self.decisions[decision_id] = {
                'description': description,
                'options': options,
                'votes': {},
                'status': 'pending',
                'created_at': datetime.now(),
                'metadata': metadata or {}
            }
            self.logger.info(f"Created decision {decision_id}: {description}")
            
        except Exception as e:
            self.logger.error(f"Error creating decision: {str(e)}")
            raise
    
    def cast_vote(self, decision_id: str, agent_id: str,
                  option: str, confidence: float = 1.0) -> None:
        """Cast a vote for a decision."""
        try:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision {decision_id} does not exist")
            
            if option not in self.decisions[decision_id]['options']:
                raise ValueError(f"Option {option} is not valid for decision {decision_id}")
            
            self.decisions[decision_id]['votes'][agent_id] = {
                'option': option,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Agent {agent_id} voted for {option} in decision {decision_id}")
            
        except Exception as e:
            self.logger.error(f"Error casting vote: {str(e)}")
            raise
    
    def get_consensus(self, decision_id: str) -> Optional[str]:
        """Get the consensus result for a decision."""
        try:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision {decision_id} does not exist")
            
            decision = self.decisions[decision_id]
            if not decision['votes']:
                return None
            
            # Count weighted votes
            vote_counts = defaultdict(float)
            total_confidence = 0.0
            
            for vote in decision['votes'].values():
                vote_counts[vote['option']] += vote['confidence']
                total_confidence += vote['confidence']
            
            if total_confidence == 0:
                return None
            
            # Calculate percentages
            for option in vote_counts:
                vote_counts[option] /= total_confidence
            
            # Find option with highest percentage
            best_option = max(vote_counts.items(), key=lambda x: x[1])
            
            # Check if consensus threshold is met
            if best_option[1] >= self.consensus_threshold:
                decision['status'] = 'resolved'
                decision['result'] = best_option[0]
                return best_option[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting consensus: {str(e)}")
            return None
    
    def get_decision_status(self, decision_id: str) -> Dict[str, Any]:
        """Get the current status of a decision."""
        try:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision {decision_id} does not exist")
            
            decision = self.decisions[decision_id]
            consensus = self.get_consensus(decision_id)
            
            return {
                'decision_id': decision_id,
                'description': decision['description'],
                'options': decision['options'],
                'status': decision['status'],
                'consensus': consensus,
                'votes': decision['votes'],
                'created_at': decision['created_at'],
                'metadata': decision['metadata']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting decision status: {str(e)}")
            return {}

class CollaborativeWorkspace:
    """Shared workspace for agent collaboration."""
    
    def __init__(self):
        self.logger = logging.getLogger("collaborative_workspace")
        self.work_items: Dict[str, WorkItem] = {}
        self.shared_data: Dict[str, Any] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.consensus_manager = ConsensusManager()
    
    def create_session(self, session_id: str, description: str,
                      participants: List[str], metadata: Dict[str, Any] = None) -> None:
        """Create a new collaboration session."""
        try:
            self.sessions[session_id] = {
                'description': description,
                'participants': participants,
                'work_items': [],
                'shared_data': {},
                'created_at': datetime.now(),
                'metadata': metadata or {}
            }
            self.logger.info(f"Created collaboration session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            raise
    
    def add_work_item(self, session_id: str, title: str, description: str,
                     metadata: Dict[str, Any] = None) -> str:
        """Add a work item to a session."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} does not exist")
            
            work_item = WorkItem.create(title, description, metadata)
            self.work_items[work_item.item_id] = work_item
            self.sessions[session_id]['work_items'].append(work_item.item_id)
            
            self.logger.info(f"Added work item {work_item.item_id} to session {session_id}")
            return work_item.item_id
            
        except Exception as e:
            self.logger.error(f"Error adding work item: {str(e)}")
            raise
    
    def update_work_item(self, item_id: str, status: WorkItemStatus,
                        assigned_agent: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> None:
        """Update a work item's status and metadata."""
        try:
            if item_id not in self.work_items:
                raise ValueError(f"Work item {item_id} does not exist")
            
            work_item = self.work_items[item_id]
            work_item.status = status
            work_item.updated_at = datetime.now()
            
            if assigned_agent is not None:
                work_item.assigned_agent = assigned_agent
            
            if metadata is not None:
                work_item.metadata.update(metadata)
            
            self.logger.info(f"Updated work item {item_id} status to {status.value}")
            
        except Exception as e:
            self.logger.error(f"Error updating work item: {str(e)}")
            raise
    
    def share_data(self, session_id: str, key: str, value: Any) -> None:
        """Share data in a session."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} does not exist")
            
            self.sessions[session_id]['shared_data'][key] = {
                'value': value,
                'updated_at': datetime.now()
            }
            
            self.logger.info(f"Shared data {key} in session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error sharing data: {str(e)}")
            raise
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get all data shared in a session."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} does not exist")
            
            return {
                'session_info': self.sessions[session_id],
                'work_items': [self.work_items[item_id] for item_id in
                             self.sessions[session_id]['work_items']],
                'shared_data': self.sessions[session_id]['shared_data']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session data: {str(e)}")
            return {}
    
    def get_work_item(self, item_id: str) -> Optional[WorkItem]:
        """Get a work item by ID."""
        return self.work_items.get(item_id)
    
    def get_session_work_items(self, session_id: str,
                             status: Optional[WorkItemStatus] = None) -> List[WorkItem]:
        """Get work items in a session, optionally filtered by status."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} does not exist")
            
            items = [self.work_items[item_id] for item_id in
                    self.sessions[session_id]['work_items']]
            
            if status:
                items = [item for item in items if item.status == status]
            
            return items
            
        except Exception as e:
            self.logger.error(f"Error getting session work items: {str(e)}")
            return []
    
    def close_session(self, session_id: str) -> None:
        """Close a collaboration session."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} does not exist")
            
            # Update all work items to completed
            for item_id in self.sessions[session_id]['work_items']:
                if self.work_items[item_id].status == WorkItemStatus.IN_PROGRESS:
                    self.update_work_item(item_id, WorkItemStatus.COMPLETED)
            
            # Archive session data
            self.sessions[session_id]['closed_at'] = datetime.now()
            self.sessions[session_id]['status'] = 'closed'
            
            self.logger.info(f"Closed session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")
            raise 