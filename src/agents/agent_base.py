"""Base Agent Class"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentBase:
    """Base class for all autonomous agents"""
    
    def __init__(self, agent_id: int, specialty: str):
        self.agent_id = agent_id
        self.specialty = specialty
        self.status = "idle"
        self.tasks_completed = 0
        self.success_rate = 1.0
        self.learning_rate = 0.01
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute()")
    
    async def learn(self, feedback: Dict[str, Any]):
        """Learn from feedback"""
        if feedback.get("success"):
            self.success_rate = min(1.0, self.success_rate * 1.01)
        else:
            self.success_rate *= 0.99
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'specialty': self.specialty,
            'status': self.status,
            'tasks_completed': self.tasks_completed,
            'success_rate': self.success_rate
        }
