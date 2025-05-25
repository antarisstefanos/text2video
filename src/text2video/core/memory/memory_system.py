from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import json
import os
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import hashlib
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch

from ..interfaces.memory import IMemorySystem

class EnhancedMemorySystem(IMemorySystem):
    """Enhanced implementation of the memory system with vector store integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("memory_system")
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'memory_storage'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        pinecone.init(
            api_key=config.get('pinecone_api_key'),
            environment=config.get('pinecone_environment')
        )
        self.vector_store = pinecone.Index(config.get('pinecone_index_name'))
        
        # Initialize feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Working memory
        self.working_memory = deque(maxlen=100)
        self.working_memory_ttl = config.get('working_memory_ttl', 3600)  # 1 hour
        
        # User preferences
        self.user_preferences = defaultdict(dict)
        self.preference_weights = defaultdict(lambda: 1.0)
        
        # Memory metrics
        self.metrics = {
            'total_memories': 0,
            'memory_types': defaultdict(int),
            'last_consolidation': None,
            'consolidation_count': 0
        }
        
        # Load existing memories
        self._load_memories()
    
    async def store_experience(self, experience: Dict[str, Any]) -> str:
        """Store a new experience in memory."""
        try:
            # Generate memory ID
            memory_id = hashlib.md5(json.dumps(experience, sort_keys=True).encode()).hexdigest()
            
            # Add metadata
            experience['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'memory_id': memory_id,
                'type': experience.get('type', 'unknown')
            }
            
            # Extract features
            features = self._extract_features(experience)
            
            # Store in vector store
            self.vector_store.upsert(
                vectors=[{
                    'id': memory_id,
                    'values': features.tolist(),
                    'metadata': experience['metadata']
                }]
            )
            
            # Store in working memory
            self.working_memory.append({
                'memory_id': memory_id,
                'experience': experience,
                'timestamp': datetime.now()
            })
            
            # Update user preferences if applicable
            if 'user_id' in experience:
                self._update_user_preferences(experience)
            
            # Save to disk
            self._save_memory(memory_id, experience)
            
            # Update metrics
            self.metrics['total_memories'] += 1
            self.metrics['memory_types'][experience.get('type', 'unknown')] += 1
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing experience: {str(e)}")
            raise
    
    async def retrieve_relevant_memories(self, context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the given context."""
        try:
            # Extract context features
            context_features = self._extract_features(context)
            
            # Query vector store
            results = self.vector_store.query(
                vector=context_features.tolist(),
                top_k=limit,
                include_metadata=True
            )
            
            # Load full memories
            memories = []
            for match in results.matches:
                memory_id = match.id
                memory = self._load_memory(memory_id)
                if memory:
                    memory['relevance_score'] = match.score
                    memories.append(memory)
            
            # Sort by relevance
            memories.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {str(e)}")
            return []
    
    async def consolidate_memories(self, quality_threshold: float = 0.8) -> Dict[str, Any]:
        """Consolidate and optimize stored memories."""
        try:
            # Get all memories
            all_memories = self._load_all_memories()
            
            # Group by type
            memories_by_type = defaultdict(list)
            for memory in all_memories:
                memories_by_type[memory.get('type', 'unknown')].append(memory)
            
            # Consolidate each type
            consolidated = {}
            for mem_type, memories in memories_by_type.items():
                # Calculate quality scores
                quality_scores = []
                for memory in memories:
                    score = self._calculate_memory_quality(memory)
                    quality_scores.append(score)
                
                # Filter by quality
                high_quality_memories = [
                    mem for mem, score in zip(memories, quality_scores)
                    if score >= quality_threshold
                ]
                
                # Store consolidated memories
                consolidated[mem_type] = high_quality_memories
            
            # Update vector store
            for mem_type, memories in consolidated.items():
                for memory in memories:
                    features = self._extract_features(memory)
                    self.vector_store.upsert(
                        vectors=[{
                            'id': memory['metadata']['memory_id'],
                            'values': features.tolist(),
                            'metadata': memory['metadata']
                        }]
                    )
            
            # Update metrics
            self.metrics['last_consolidation'] = datetime.now().isoformat()
            self.metrics['consolidation_count'] += 1
            
            return {
                'success': True,
                'consolidated_types': list(consolidated.keys()),
                'total_memories': sum(len(mems) for mems in consolidated.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            **self.metrics,
            'working_memory_size': len(self.working_memory),
            'user_preferences_count': len(self.user_preferences),
            'timestamp': datetime.now().isoformat()
        }
    
    async def clear_memory(self, memory_type: Optional[str] = None) -> None:
        """Clear specific or all memories."""
        try:
            if memory_type:
                # Clear specific type
                memories = self._load_all_memories()
                for memory in memories:
                    if memory.get('type') == memory_type:
                        memory_id = memory['metadata']['memory_id']
                        self._delete_memory(memory_id)
                        self.vector_store.delete(ids=[memory_id])
                
                # Update metrics
                self.metrics['total_memories'] -= self.metrics['memory_types'].get(memory_type, 0)
                self.metrics['memory_types'][memory_type] = 0
            else:
                # Clear all memories
                self.vector_store.delete(delete_all=True)
                shutil.rmtree(self.storage_path)
                self.storage_path.mkdir(parents=True, exist_ok=True)
                
                # Reset metrics
                self.metrics = {
                    'total_memories': 0,
                    'memory_types': defaultdict(int),
                    'last_consolidation': None,
                    'consolidation_count': 0
                }
            
        except Exception as e:
            self.logger.error(f"Error clearing memory: {str(e)}")
            raise
    
    async def backup_memory(self, backup_path: str) -> None:
        """Create a backup of the memory system."""
        try:
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup memories
            shutil.copytree(
                self.storage_path,
                backup_path / 'memories',
                dirs_exist_ok=True
            )
            
            # Backup metrics and preferences
            backup_data = {
                'metrics': self.metrics,
                'user_preferences': dict(self.user_preferences),
                'preference_weights': dict(self.preference_weights),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(backup_path / 'backup_metadata.json', 'w') as f:
                json.dump(backup_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error backing up memory: {str(e)}")
            raise
    
    def _extract_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract features from content for vector store."""
        # Combine text content
        text_content = f"{content.get('user_input', '')} {content.get('storyboard', {}).get('description', '')}"
        
        # Tokenize and encode
        inputs = self.tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.encoder(**inputs).last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze()
    
    def _calculate_memory_quality(self, memory: Dict[str, Any]) -> float:
        """Calculate quality score for a memory."""
        # Base quality factors
        factors = {
            'completeness': 0.0,
            'relevance': 0.0,
            'user_feedback': 0.0
        }
        
        # Check completeness
        required_fields = ['type', 'content', 'metadata']
        factors['completeness'] = sum(1 for field in required_fields if field in memory) / len(required_fields)
        
        # Check relevance (based on user preferences)
        if 'user_id' in memory:
            user_prefs = self.user_preferences.get(memory['user_id'], {})
            if user_prefs:
                factors['relevance'] = np.mean([
                    pref.get('rating', 0.5)
                    for pref in user_prefs.values()
                ])
        
        # Check user feedback
        if 'user_feedback' in memory:
            feedback_score = memory.get('user_rating', 0.5)
            factors['user_feedback'] = feedback_score
        
        # Calculate weighted average
        weights = {
            'completeness': 0.4,
            'relevance': 0.3,
            'user_feedback': 0.3
        }
        
        return sum(score * weights[factor] for factor, score in factors.items())
    
    def _update_user_preferences(self, experience: Dict[str, Any]) -> None:
        """Update user preferences based on experience."""
        user_id = experience['user_id']
        content_hash = hashlib.md5(json.dumps(experience, sort_keys=True).encode()).hexdigest()
        
        # Update preferences
        self.user_preferences[user_id][content_hash] = {
            'rating': experience.get('user_rating', 0.5),
            'feedback': experience.get('user_feedback', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update preference weights
        self.preference_weights[user_id] = min(
            1.0,
            self.preference_weights[user_id] + 0.1
        )
    
    def _save_memory(self, memory_id: str, memory: Dict[str, Any]) -> None:
        """Save memory to disk."""
        memory_path = self.storage_path / f"{memory_id}.json"
        with open(memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
    
    def _load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load memory from disk."""
        memory_path = self.storage_path / f"{memory_id}.json"
        if memory_path.exists():
            with open(memory_path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_all_memories(self) -> List[Dict[str, Any]]:
        """Load all memories from disk."""
        memories = []
        for memory_file in self.storage_path.glob('*.json'):
            with open(memory_file, 'r') as f:
                memories.append(json.load(f))
        return memories
    
    def _delete_memory(self, memory_id: str) -> None:
        """Delete memory from disk."""
        memory_path = self.storage_path / f"{memory_id}.json"
        if memory_path.exists():
            memory_path.unlink()
    
    def _load_memories(self) -> None:
        """Load existing memories from disk."""
        try:
            memories = self._load_all_memories()
            
            # Update metrics
            self.metrics['total_memories'] = len(memories)
            for memory in memories:
                self.metrics['memory_types'][memory.get('type', 'unknown')] += 1
            
            # Load into vector store
            for memory in memories:
                features = self._extract_features(memory)
                self.vector_store.upsert(
                    vectors=[{
                        'id': memory['metadata']['memory_id'],
                        'values': features.tolist(),
                        'metadata': memory['metadata']
                    }]
                )
            
        except Exception as e:
            self.logger.error(f"Error loading memories: {str(e)}")
            # Initialize empty metrics if loading fails
            self.metrics = {
                'total_memories': 0,
                'memory_types': defaultdict(int),
                'last_consolidation': None,
                'consolidation_count': 0
            } 