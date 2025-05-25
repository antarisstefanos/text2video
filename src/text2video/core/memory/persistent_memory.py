from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import sqlite3
import json
import uuid
import numpy as np
from pathlib import Path

class PersistentMemoryDatabase:
    """SQLite-based persistent memory storage."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.logger = logging.getLogger("persistent_memory")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Episodic memory: specific experiences and projects
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                user_id TEXT,
                task_type TEXT,
                input_content TEXT,
                output_content TEXT,
                quality_score REAL,
                user_rating REAL,
                success BOOLEAN,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        # Semantic memory: learned concepts and patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id TEXT PRIMARY KEY,
                concept_type TEXT,
                concept_name TEXT,
                description TEXT,
                embedding BLOB,
                confidence REAL,
                usage_count INTEGER,
                last_used DATETIME,
                created_at DATETIME
            )
        ''')
        
        # Procedural memory: learned workflows and procedures
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS procedural_memory (
                id TEXT PRIMARY KEY,
                procedure_name TEXT,
                task_type TEXT,
                agent_sequence TEXT,
                parameters TEXT,
                success_rate REAL,
                avg_quality REAL,
                usage_count INTEGER,
                last_updated DATETIME
            )
        ''')
        
        # User preferences and patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                interaction_patterns TEXT,
                quality_patterns TEXT,
                updated_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_episode(self, project_id: str, user_id: str, task_type: str,
                     input_content: Dict, output_content: Dict, quality_score: float,
                     user_rating: float = None, success: bool = True, metadata: Dict = None) -> str:
        """Store an episodic memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            episode_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO episodic_memory 
                (id, project_id, user_id, task_type, input_content, output_content, 
                 quality_score, user_rating, success, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                episode_id, project_id, user_id, task_type,
                json.dumps(input_content), json.dumps(output_content),
                quality_score, user_rating, success, datetime.now(),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            return episode_id
            
        except Exception as e:
            self.logger.error(f"Error storing episode: {str(e)}")
            raise
    
    def retrieve_similar_episodes(self, task_type: str, user_id: str = None, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar episodes based on task type and user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM episodic_memory 
                WHERE task_type = ?
            '''
            params = [task_type]
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY quality_score DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            episodes = []
            
            for row in cursor.fetchall():
                episodes.append({
                    'id': row[0],
                    'project_id': row[1],
                    'user_id': row[2],
                    'task_type': row[3],
                    'input_content': json.loads(row[4]),
                    'output_content': json.loads(row[5]),
                    'quality_score': row[6],
                    'user_rating': row[7],
                    'success': row[8],
                    'timestamp': row[9],
                    'metadata': json.loads(row[10])
                })
            
            conn.close()
            return episodes
            
        except Exception as e:
            self.logger.error(f"Error retrieving episodes: {str(e)}")
            return []
    
    def store_semantic_concept(self, concept_type: str, concept_name: str,
                             description: str, embedding: np.ndarray,
                             confidence: float = 1.0) -> str:
        """Store a semantic concept."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            concept_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO semantic_memory 
                (id, concept_type, concept_name, description, embedding,
                 confidence, usage_count, last_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                concept_id, concept_type, concept_name, description,
                embedding.tobytes(), confidence, 1, datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            return concept_id
            
        except Exception as e:
            self.logger.error(f"Error storing semantic concept: {str(e)}")
            raise
    
    def retrieve_semantic_concepts(self, concept_type: str = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve semantic concepts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM semantic_memory"
            params = []
            
            if concept_type:
                query += " WHERE concept_type = ?"
                params.append(concept_type)
            
            query += " ORDER BY confidence DESC, usage_count DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            concepts = []
            
            for row in cursor.fetchall():
                concepts.append({
                    'id': row[0],
                    'concept_type': row[1],
                    'concept_name': row[2],
                    'description': row[3],
                    'embedding': np.frombuffer(row[4]),
                    'confidence': row[5],
                    'usage_count': row[6],
                    'last_used': row[7],
                    'created_at': row[8]
                })
            
            conn.close()
            return concepts
            
        except Exception as e:
            self.logger.error(f"Error retrieving semantic concepts: {str(e)}")
            return []
    
    def store_procedure(self, procedure_name: str, task_type: str,
                       agent_sequence: List[str], parameters: Dict[str, Any],
                       success_rate: float = 1.0, avg_quality: float = 1.0) -> str:
        """Store a procedural memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            procedure_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO procedural_memory 
                (id, procedure_name, task_type, agent_sequence, parameters,
                 success_rate, avg_quality, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                procedure_id, procedure_name, task_type,
                json.dumps(agent_sequence), json.dumps(parameters),
                success_rate, avg_quality, 1, datetime.now()
            ))
            
            conn.commit()
            conn.close()
            return procedure_id
            
        except Exception as e:
            self.logger.error(f"Error storing procedure: {str(e)}")
            raise
    
    def retrieve_procedures(self, task_type: str = None,
                          min_success_rate: float = 0.0,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve procedures."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM procedural_memory WHERE success_rate >= ?"
            params = [min_success_rate]
            
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)
            
            query += " ORDER BY success_rate DESC, avg_quality DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            procedures = []
            
            for row in cursor.fetchall():
                procedures.append({
                    'id': row[0],
                    'procedure_name': row[1],
                    'task_type': row[2],
                    'agent_sequence': json.loads(row[3]),
                    'parameters': json.loads(row[4]),
                    'success_rate': row[5],
                    'avg_quality': row[6],
                    'usage_count': row[7],
                    'last_updated': row[8]
                })
            
            conn.close()
            return procedures
            
        except Exception as e:
            self.logger.error(f"Error retrieving procedures: {str(e)}")
            return []
    
    def update_user_profile(self, user_id: str, preferences: Dict[str, Any],
                          interaction_patterns: Dict[str, Any],
                          quality_patterns: Dict[str, Any]) -> None:
        """Update user profile."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preferences, interaction_patterns, quality_patterns, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                json.dumps(preferences),
                json.dumps(interaction_patterns),
                json.dumps(quality_patterns),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
            raise
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_profiles WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'user_id': row[0],
                    'preferences': json.loads(row[1]),
                    'interaction_patterns': json.loads(row[2]),
                    'quality_patterns': json.loads(row[3]),
                    'updated_at': row[4]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return None
    
    def cleanup_old_memories(self, max_age_days: int = 30) -> None:
        """Clean up old memories."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up episodic memories
            cursor.execute('''
                DELETE FROM episodic_memory 
                WHERE timestamp < ? AND quality_score < 0.7
            ''', (cutoff_date,))
            
            # Clean up semantic concepts
            cursor.execute('''
                DELETE FROM semantic_memory 
                WHERE last_used < ? AND confidence < 0.8
            ''', (cutoff_date,))
            
            # Clean up procedures
            cursor.execute('''
                DELETE FROM procedural_memory 
                WHERE last_updated < ? AND success_rate < 0.7
            ''', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old memories: {str(e)}")
            raise 