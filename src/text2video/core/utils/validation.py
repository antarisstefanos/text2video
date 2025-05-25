from typing import Any, Dict, List, Optional
import re
from datetime import datetime

def validate_task(task: Dict[str, Any]) -> bool:
    """Validate a task dictionary."""
    required_fields = ['type', 'content']
    
    # Check required fields
    if not all(field in task for field in required_fields):
        return False
    
    # Validate task type
    if not isinstance(task['type'], str) or not task['type']:
        return False
    
    # Validate content
    if not isinstance(task['content'], (dict, str)):
        return False
    
    return True

def validate_agent_config(config: Dict[str, Any]) -> bool:
    """Validate an agent configuration dictionary."""
    required_fields = ['agent_id', 'role']
    
    # Check required fields
    if not all(field in config for field in required_fields):
        return False
    
    # Validate agent_id
    if not isinstance(config['agent_id'], str) or not config['agent_id']:
        return False
    
    # Validate role
    if not isinstance(config['role'], str) or not config['role']:
        return False
    
    return True

def validate_memory_content(content: Dict[str, Any]) -> bool:
    """Validate memory content dictionary."""
    required_fields = ['type', 'data']
    
    # Check required fields
    if not all(field in content for field in required_fields):
        return False
    
    # Validate type
    if not isinstance(content['type'], str) or not content['type']:
        return False
    
    # Validate data
    if not isinstance(content['data'], (dict, str, list)):
        return False
    
    return True

def validate_video_params(params: Dict[str, Any]) -> bool:
    """Validate video generation parameters."""
    required_fields = ['resolution', 'format']
    
    # Check required fields
    if not all(field in params for field in required_fields):
        return False
    
    # Validate resolution
    if not isinstance(params['resolution'], str):
        return False
    if not re.match(r'^\d+x\d+$', params['resolution']):
        return False
    
    # Validate format
    if not isinstance(params['format'], str):
        return False
    if params['format'] not in ['mp4', 'avi', 'mov']:
        return False
    
    return True

def validate_timestamp(timestamp: str) -> bool:
    """Validate ISO format timestamp."""
    try:
        datetime.fromisoformat(timestamp)
        return True
    except ValueError:
        return False

def validate_expertise_levels(expertise: Dict[str, float]) -> bool:
    """Validate agent expertise levels."""
    if not isinstance(expertise, dict):
        return False
    
    for skill, level in expertise.items():
        if not isinstance(skill, str) or not skill:
            return False
        if not isinstance(level, (int, float)) or not 0 <= level <= 1:
            return False
    
    return True 