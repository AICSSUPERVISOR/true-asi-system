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
        """Execute a task - default implementation with LLM"""
        try:
            import openai
            import os
            
            # Get task details
            task_type = task.get('type', 'general')
            task_description = task.get('description', '')
            task_input = task.get('input', {})
            
            # Use OpenAI to execute task
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    'success': False,
                    'error': 'No OpenAI API key configured',
                    'agent_id': self.agent_id
                }
            
            client = openai.OpenAI(api_key=api_key)
            
            # Create prompt based on task
            prompt = f"""Task Type: {task_type}
Description: {task_description}
Input: {task_input}

Please execute this task and provide a detailed result."""
            
            # Execute with LLM
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.choices[0].message.content
            
            # Update agent stats
            self.tasks_completed += 1
            self.status = "completed"
            
            return {
                'success': True,
                'result': result,
                'agent_id': self.agent_id,
                'task_type': task_type
            }
            
        except Exception as e:
            self.status = "error"
            return {
                'success': False,
                'error': str(e),
                'agent_id': self.agent_id
            }
    
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
