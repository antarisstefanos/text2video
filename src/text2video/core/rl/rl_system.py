from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import hashlib
import json

from ..interfaces.rl import IRLSystem
from transformers import AutoTokenizer, AutoModel

class RewardModel(nn.Module):
    """Neural network for learning user preferences and quality assessment."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

class UserFeedbackRL:
    """Reinforcement Learning system for user feedback integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_model = RewardModel()
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.feedback_buffer = deque(maxlen=10000)
        self.user_preferences = defaultdict(dict)
        
        # Feature extractor for content embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    def extract_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract features from generated content for reward model."""
        # Combine text content
        text_content = f"{content.get('user_input', '')} {content.get('storyboard', {}).get('description', '')}"
        
        # Tokenize and encode
        inputs = self.tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.encoder(**inputs).last_hidden_state.mean(dim=1)
        
        # Add metadata features
        metadata_features = torch.tensor([
            content.get('quality_score', 0.0),
            content.get('generation_time', 0.0),
            len(content.get('visual_assets', [])),
            len(content.get('audio_assets', []))
        ]).unsqueeze(0)
        
        return torch.cat([embeddings, metadata_features], dim=1)
    
    def store_feedback(self, content: Dict[str, Any], user_rating: float, 
                      user_feedback: str, user_id: str = "default"):
        """Store user feedback for training."""
        features = self.extract_features(content)
        
        feedback_entry = {
            'features': features,
            'rating': user_rating,
            'feedback': user_feedback,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'content_hash': hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Update user preferences
        self.user_preferences[user_id][feedback_entry['content_hash']] = {
            'rating': user_rating,
            'feedback': user_feedback,
            'features': features.squeeze().tolist()
        }
    
    def train_reward_model(self, batch_size: int = 32):
        """Train the reward model on collected feedback."""
        if len(self.feedback_buffer) < batch_size:
            return
        
        # Sample batch
        batch = list(self.feedback_buffer)[-batch_size:]
        
        features = torch.cat([item['features'] for item in batch])
        ratings = torch.tensor([item['rating'] for item in batch]).float().unsqueeze(1)
        
        # Train
        self.optimizer.zero_grad()
        predictions = self.reward_model(features)
        loss = torch.nn.functional.mse_loss(predictions, ratings)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_reward(self, content: Dict[str, Any]) -> float:
        """Predict reward/quality for given content."""
        features = self.extract_features(content)
        with torch.no_grad():
            reward = self.reward_model(features).item()
        return reward
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for a specific user."""
        if user_id not in self.user_preferences:
            return {}
        
        preferences = self.user_preferences[user_id]
        
        # Analyze patterns in preferences
        high_rated = [p for p in preferences.values() if p['rating'] > 0.7]
        low_rated = [p for p in preferences.values() if p['rating'] < 0.3]
        
        return {
            'total_feedback': len(preferences),
            'high_rated_count': len(high_rated),
            'low_rated_count': len(low_rated),
            'average_rating': np.mean([p['rating'] for p in preferences.values()]) if preferences else 0.5,
            'preferred_features': self._extract_preferred_features(high_rated),
            'avoided_features': self._extract_preferred_features(low_rated)
        }
    
    def _extract_preferred_features(self, rated_items: List[Dict]) -> Dict[str, Any]:
        """Extract common features from highly rated items."""
        if not rated_items:
            return {}
        
        # Simple feature analysis - in practice, use more sophisticated methods
        avg_features = np.mean([item['features'] for item in rated_items], axis=0)
        return {'feature_vector': avg_features.tolist()}

class AgentOrchestrationRL:
    """RL for learning optimal agent coordination patterns."""
    
    def __init__(self):
        self.coordination_history = deque(maxlen=1000)
        self.agent_performance = defaultdict(lambda: {'success_rate': 0.5, 'avg_quality': 0.5, 'avg_time': 1.0})
        self.collaboration_patterns = defaultdict(lambda: {'success_rate': 0.5, 'synergy_score': 0.5})
        
        # Q-learning for agent selection
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate
        
    def record_coordination_outcome(self, agents_used: List[str], task_type: str, 
                                  success: bool, quality_score: float, execution_time: float):
        """Record the outcome of an agent coordination pattern."""
        coordination_key = f"{task_type}::{'+'.join(sorted(agents_used))}"
        
        outcome = {
            'agents': agents_used,
            'task_type': task_type,
            'success': success,
            'quality': quality_score,
            'time': execution_time,
            'timestamp': datetime.now(),
            'coordination_key': coordination_key
        }
        
        self.coordination_history.append(outcome)
        
        # Update agent performance
        for agent in agents_used:
            perf = self.agent_performance[agent]
            perf['success_rate'] = 0.9 * perf['success_rate'] + 0.1 * (1.0 if success else 0.0)
            perf['avg_quality'] = 0.9 * perf['avg_quality'] + 0.1 * quality_score
            perf['avg_time'] = 0.9 * perf['avg_time'] + 0.1 * execution_time
        
        # Update collaboration patterns
        collab = self.collaboration_patterns[coordination_key]
        collab['success_rate'] = 0.9 * collab['success_rate'] + 0.1 * (1.0 if success else 0.0)
        collab['synergy_score'] = 0.9 * collab['synergy_score'] + 0.1 * quality_score
        
        # Update Q-table
        reward = quality_score if success else -0.5
        state = f"{task_type}::{len(agents_used)}"
        action = coordination_key
        
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.learning_rate * (reward - old_q)
    
    def select_optimal_agents(self, task_type: str, available_agents: List[str], 
                            max_agents: int = 3) -> List[str]:
        """Select optimal agent combination for a task using learned patterns."""
        state = f"{task_type}::{max_agents}"
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            num_agents = min(np.random.randint(1, max_agents + 1), len(available_agents))
            return np.random.choice(available_agents, num_agents, replace=False).tolist()
        
        # Exploitation: use Q-table
        best_action = None
        best_q_value = float('-inf')
        
        # Try different agent combinations
        for num_agents in range(1, min(max_agents + 1, len(available_agents) + 1)):
            from itertools import combinations
            for agent_combo in combinations(available_agents, num_agents):
                action = f"{task_type}::{'+'.join(sorted(agent_combo))}"
                q_value = self.q_table[state][action]
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = list(agent_combo)
        
        return best_action if best_action else available_agents[:max_agents]
    
    def get_coordination_insights(self) -> Dict[str, Any]:
        """Get insights about learned coordination patterns."""
        if not self.coordination_history:
            return {}
        
        recent_history = list(self.coordination_history)[-100:]
        
        return {
            'total_coordinations': len(self.coordination_history),
            'recent_success_rate': np.mean([h['success'] for h in recent_history]),
            'recent_avg_quality': np.mean([h['quality'] for h in recent_history]),
            'best_agent_performances': dict(sorted(
                self.agent_performance.items(), 
                key=lambda x: x[1]['success_rate'] * x[1]['avg_quality'], 
                reverse=True
            )[:5]),
            'best_collaboration_patterns': dict(sorted(
                self.collaboration_patterns.items(),
                key=lambda x: x[1]['success_rate'] * x[1]['synergy_score'],
                reverse=True
            )[:5])
        }

class RLSystem(IRLSystem):
    """Implementation of the reinforcement learning system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("rl_system")
        self.config = config
        self.reward_model = RewardModel()
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.feedback_buffer = deque(maxlen=10000)
        self.user_preferences = defaultdict(dict)
        
        # Feature extractor for content embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Coordination learning
        self.coordination_history = deque(maxlen=1000)
        self.agent_performance = defaultdict(lambda: {'success_rate': 0.5, 'avg_quality': 0.5, 'avg_time': 1.0})
        self.collaboration_patterns = defaultdict(lambda: {'success_rate': 0.5, 'synergy_score': 0.5})
        
        # Q-learning for agent selection
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate
    
    async def update_model(self, experience: Dict[str, Any]) -> None:
        """Update the RL model with new experience."""
        try:
            # Extract features
            features = self._extract_features(experience)
            
            # Store feedback
            if 'user_rating' in experience:
                self._store_feedback(experience, features)
            
            # Update coordination patterns
            if 'agents_used' in experience:
                self._update_coordination_patterns(experience)
            
            # Train model
            await self.train([experience])
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            raise
    
    async def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best action for a given state."""
        try:
            # Extract state features
            features = self._extract_features(state)
            
            # Get model prediction
            with torch.no_grad():
                reward = self.reward_model(features).item()
            
            # Select action based on state type
            if state.get('type') == 'agent_selection':
                return self._select_optimal_agents(state)
            else:
                return {
                    'action': 'default',
                    'confidence': reward,
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error getting action: {str(e)}")
            return {
                'action': 'default',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get metrics about the RL model's performance."""
        return {
            'feedback_buffer_size': len(self.feedback_buffer),
            'user_preferences_count': len(self.user_preferences),
            'coordination_patterns': len(self.collaboration_patterns),
            'agent_performance': dict(self.agent_performance),
            'timestamp': datetime.now().isoformat()
        }
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the model on a batch of data."""
        try:
            if not training_data:
                return {'success': False, 'error': 'No training data provided'}
            
            # Prepare training batch
            features = torch.cat([self._extract_features(item) for item in training_data])
            targets = torch.tensor([item.get('user_rating', 0.5) for item in training_data]).float().unsqueeze(1)
            
            # Train
            self.optimizer.zero_grad()
            predictions = self.reward_model(features)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            return {
                'success': True,
                'loss': loss.item(),
                'batch_size': len(training_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def save_model(self, path: str) -> None:
        """Save the current model state."""
        try:
            torch.save({
                'model_state': self.reward_model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'q_table': dict(self.q_table),
                'user_preferences': dict(self.user_preferences),
                'collaboration_patterns': dict(self.collaboration_patterns)
            }, path)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    async def load_model(self, path: str) -> None:
        """Load a model from disk."""
        try:
            checkpoint = torch.load(path)
            self.reward_model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.q_table = defaultdict(lambda: defaultdict(float), checkpoint['q_table'])
            self.user_preferences = defaultdict(dict, checkpoint['user_preferences'])
            self.collaboration_patterns = defaultdict(lambda: {'success_rate': 0.5, 'synergy_score': 0.5},
                                                    checkpoint['collaboration_patterns'])
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        try:
            if not test_data:
                return {'success': False, 'error': 'No test data provided'}
            
            # Prepare test batch
            features = torch.cat([self._extract_features(item) for item in test_data])
            targets = torch.tensor([item.get('user_rating', 0.5) for item in test_data]).float().unsqueeze(1)
            
            # Evaluate
            with torch.no_grad():
                predictions = self.reward_model(features)
                loss = torch.nn.functional.mse_loss(predictions, targets)
                accuracy = torch.mean((torch.abs(predictions - targets) < 0.1).float())
            
            return {
                'success': True,
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'test_size': len(test_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_features(self, content: Dict[str, Any]) -> torch.Tensor:
        """Extract features from content for reward model."""
        # Combine text content
        text_content = f"{content.get('user_input', '')} {content.get('storyboard', {}).get('description', '')}"
        
        # Tokenize and encode
        inputs = self.tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.encoder(**inputs).last_hidden_state.mean(dim=1)
        
        # Add metadata features
        metadata_features = torch.tensor([
            content.get('quality_score', 0.0),
            content.get('generation_time', 0.0),
            len(content.get('visual_assets', [])),
            len(content.get('audio_assets', []))
        ]).unsqueeze(0)
        
        return torch.cat([embeddings, metadata_features], dim=1)
    
    def _store_feedback(self, content: Dict[str, Any], features: torch.Tensor) -> None:
        """Store user feedback for training."""
        feedback_entry = {
            'features': features,
            'rating': content.get('user_rating', 0.5),
            'feedback': content.get('user_feedback', ''),
            'user_id': content.get('user_id', 'default'),
            'timestamp': datetime.now(),
            'content_hash': hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Update user preferences
        self.user_preferences[feedback_entry['user_id']][feedback_entry['content_hash']] = {
            'rating': feedback_entry['rating'],
            'feedback': feedback_entry['feedback'],
            'features': features.squeeze().tolist()
        }
    
    def _update_coordination_patterns(self, experience: Dict[str, Any]) -> None:
        """Update coordination patterns from experience."""
        agents_used = experience.get('agents_used', [])
        task_type = experience.get('task_type', 'unknown')
        success = experience.get('success', False)
        quality_score = experience.get('quality_score', 0.5)
        execution_time = experience.get('execution_time', 1.0)
        
        # Update agent performance
        for agent in agents_used:
            perf = self.agent_performance[agent]
            perf['success_rate'] = 0.9 * perf['success_rate'] + 0.1 * (1.0 if success else 0.0)
            perf['avg_quality'] = 0.9 * perf['avg_quality'] + 0.1 * quality_score
            perf['avg_time'] = 0.9 * perf['avg_time'] + 0.1 * execution_time
        
        # Update collaboration patterns
        coordination_key = f"{task_type}::{'+'.join(sorted(agents_used))}"
        collab = self.collaboration_patterns[coordination_key]
        collab['success_rate'] = 0.9 * collab['success_rate'] + 0.1 * (1.0 if success else 0.0)
        collab['synergy_score'] = 0.9 * collab['synergy_score'] + 0.1 * quality_score
        
        # Update Q-table
        reward = quality_score if success else -0.5
        state = f"{task_type}::{len(agents_used)}"
        action = coordination_key
        
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.learning_rate * (reward - old_q)
    
    def _select_optimal_agents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal agent combination for a task."""
        task_type = state.get('task_type', 'unknown')
        available_agents = state.get('available_agents', [])
        max_agents = state.get('max_agents', 3)
        
        state_key = f"{task_type}::{max_agents}"
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            num_agents = min(np.random.randint(1, max_agents + 1), len(available_agents))
            selected_agents = np.random.choice(available_agents, num_agents, replace=False).tolist()
        else:
            # Exploitation: use Q-table
            best_action = None
            best_q_value = float('-inf')
            
            # Try different agent combinations
            for num_agents in range(1, min(max_agents + 1, len(available_agents) + 1)):
                from itertools import combinations
                for agent_combo in combinations(available_agents, num_agents):
                    action = f"{task_type}::{'+'.join(sorted(agent_combo))}"
                    q_value = self.q_table[state_key][action]
                    
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = list(agent_combo)
            
            selected_agents = best_action if best_action else available_agents[:max_agents]
        
        return {
            'selected_agents': selected_agents,
            'confidence': best_q_value if 'best_q_value' in locals() else 0.5,
            'timestamp': datetime.now().isoformat()
        } 