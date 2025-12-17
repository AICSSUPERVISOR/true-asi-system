"""Helper Utilities"""
from typing import Any, Dict
import hashlib
from datetime import datetime


def generate_id(data: str) -> str:
    """Generate unique ID from data"""
    return hashlib.md5(data.encode()).hexdigest()


def timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().isoformat()


def validate_entity(entity: Dict[str, Any]) -> bool:
    """Validate entity structure"""
    required_fields = ['name', 'type']
    return all(field in entity for field in required_fields)
