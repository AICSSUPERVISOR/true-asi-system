"""
Agent Manager - Orchestrates all 250 autonomous agents
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib
import sys
import os

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages all 250 autonomous agents in the TRUE ASI System"""
    
    def __init__(self):
        self.agents: List[Any] = []
        self.agent_status: Dict[int, str] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("Agent Manager initialized")
    
    async def initialize_agents(self, count: int = 250):
        """Initialize all agents dynamically"""
        logger.info(f"Initializing {count} agents...")
        
        # Add agents directory to path
        agents_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
        if agents_dir not in sys.path:
            sys.path.insert(0, agents_dir)
        
        for i in range(count):
            try:
                # Dynamically import agent module
                module_name = f"agent_{str(i).zfill(3)}"
                class_name = f"Agent{str(i).zfill(3)}"
                
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                
                # Instantiate agent
                agent = agent_class()
                self.agents.append(agent)
                self.agent_status[i] = 'idle'
                
                if (i + 1) % 50 == 0:
                    logger.info(f"  Initialized {i + 1}/{count} agents")
                    
            except Exception as e:
                logger.error(f"Failed to initialize agent {i}: {str(e)}")
        
        logger.info(f"✅ Successfully initialized {len(self.agents)} agents")
    
    async def assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign task to available agent"""
        # Find idle agent
        for i, agent in enumerate(self.agents):
            if self.agent_status[i] == 'idle':
                self.agent_status[i] = 'working'
                
                try:
                    result = await agent.execute(task)
                    self.agent_status[i] = 'idle'
                    return result
                except Exception as e:
                    logger.error(f"Agent {i} task failed: {str(e)}")
                    self.agent_status[i] = 'idle'
                    return {'status': 'failed', 'error': str(e)}
        
        # No idle agents, queue the task
        await self.task_queue.put(task)
        return {'status': 'queued'}
    
    async def get_pending_tasks(self) -> List[Dict]:
        """Get pending tasks from queue"""
        tasks = []
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                tasks.append(task)
            except asyncio.QueueEmpty:
                break
        return tasks
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        total = len(self.agents)
        idle = sum(1 for status in self.agent_status.values() if status == 'idle')
        working = sum(1 for status in self.agent_status.values() if status == 'working')
        
        return {
            'total_agents': total,
            'idle': idle,
            'working': working,
            'utilization': working / total if total > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
