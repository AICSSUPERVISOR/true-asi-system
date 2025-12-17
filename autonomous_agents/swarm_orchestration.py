#!/usr/bin/env python3
"""
SWARM INTELLIGENCE & AGENT ORCHESTRATION
=========================================
Coordinates multiple autonomous agents for complex tasks
Implements collective intelligence and distributed problem-solving
"""

import json
import os
import hashlib
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# SWARM CONFIGURATION
# ============================================================================

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SwarmTask:
    """Represents a task in the swarm"""
    id: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    assigned_agent: Optional[str]
    required_capabilities: List[str]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    parent_task_id: Optional[str]
    subtasks: List[str]

# ============================================================================
# SWARM ORCHESTRATOR
# ============================================================================

class SwarmOrchestrator:
    """
    Orchestrates multiple autonomous agents as a swarm
    Implements collective intelligence for complex problem-solving
    """
    
    def __init__(self, max_workers: int = 10):
        self.agents: Dict[str, Any] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.results: Dict[str, Any] = {}
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.collective_knowledge: Dict[str, Any] = {}
    
    def _generate_task_id(self, description: str) -> str:
        """Generate unique task ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{description}{timestamp}".encode()).hexdigest()[:12]
    
    # ========================================================================
    # AGENT MANAGEMENT
    # ========================================================================
    
    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent with the swarm"""
        with self.lock:
            self.agents[agent_id] = {
                "agent": agent,
                "status": "available",
                "current_task": None,
                "completed_tasks": 0,
                "performance_score": 1.0
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        with self.lock:
            return [
                agent_id for agent_id, info in self.agents.items()
                if info["status"] == "available"
            ]
    
    def get_best_agent_for_task(self, task: SwarmTask) -> Optional[str]:
        """Find the best agent for a task based on capabilities"""
        available = self.get_available_agents()
        if not available:
            return None
        
        best_agent = None
        best_score = 0
        
        for agent_id in available:
            agent_info = self.agents[agent_id]
            agent = agent_info["agent"]
            
            # Calculate capability match score
            if hasattr(agent, 'capabilities'):
                agent_caps = [c.value if hasattr(c, 'value') else str(c) for c in agent.capabilities]
                matches = sum(1 for cap in task.required_capabilities if cap in agent_caps)
                score = matches / max(len(task.required_capabilities), 1)
                
                # Factor in performance
                score *= agent_info["performance_score"]
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        
        return best_agent or (available[0] if available else None)
    
    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================
    
    def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        required_capabilities: List[str] = None,
        input_data: Dict[str, Any] = None,
        parent_task_id: Optional[str] = None
    ) -> SwarmTask:
        """Create a new task"""
        task_id = self._generate_task_id(description)
        
        task = SwarmTask(
            id=task_id,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            assigned_agent=None,
            required_capabilities=required_capabilities or [],
            input_data=input_data or {},
            output_data=None,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None,
            parent_task_id=parent_task_id,
            subtasks=[]
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.put((priority.value, task_id))
        
        return task
    
    def decompose_task(self, task: SwarmTask, subtask_specs: List[Dict]) -> List[SwarmTask]:
        """Decompose a complex task into subtasks"""
        subtasks = []
        
        for spec in subtask_specs:
            subtask = self.create_task(
                description=spec["description"],
                priority=spec.get("priority", task.priority),
                required_capabilities=spec.get("capabilities", []),
                input_data=spec.get("input", {}),
                parent_task_id=task.id
            )
            subtasks.append(subtask)
            task.subtasks.append(subtask.id)
        
        return subtasks
    
    def assign_task(self, task_id: str) -> bool:
        """Assign a task to the best available agent"""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        agent_id = self.get_best_agent_for_task(task)
        if not agent_id:
            return False
        
        with self.lock:
            task.assigned_agent = agent_id
            task.status = TaskStatus.ASSIGNED
            self.agents[agent_id]["status"] = "busy"
            self.agents[agent_id]["current_task"] = task_id
        
        return True
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task using the assigned agent"""
        task = self.tasks.get(task_id)
        if not task or not task.assigned_agent:
            return {"error": "Task not found or not assigned"}
        
        agent_info = self.agents.get(task.assigned_agent)
        if not agent_info:
            return {"error": "Assigned agent not found"}
        
        agent = agent_info["agent"]
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now().isoformat()
        
        try:
            # Execute task using agent
            if hasattr(agent, 'execute_task'):
                result = agent.execute_task(task.description)
            elif hasattr(agent, 'think'):
                result = agent.think(task.description)
            else:
                result = {"error": "Agent cannot execute tasks"}
            
            # Update task with results
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            
            # Update agent stats
            with self.lock:
                agent_info["completed_tasks"] += 1
                agent_info["status"] = "available"
                agent_info["current_task"] = None
                
                # Update performance based on success
                if result.get("success", True) and not result.get("error"):
                    agent_info["performance_score"] = min(1.0, agent_info["performance_score"] + 0.01)
            
            # Share knowledge with swarm
            self._share_knowledge(task, result)
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.output_data = {"error": str(e)}
            
            with self.lock:
                agent_info["status"] = "available"
                agent_info["current_task"] = None
                agent_info["performance_score"] = max(0.5, agent_info["performance_score"] - 0.05)
            
            return {"error": str(e)}
    
    # ========================================================================
    # COLLECTIVE INTELLIGENCE
    # ========================================================================
    
    def _share_knowledge(self, task: SwarmTask, result: Dict) -> None:
        """Share task results with the collective knowledge base"""
        with self.lock:
            task_type = task.description[:50]
            if task_type not in self.collective_knowledge:
                self.collective_knowledge[task_type] = []
            
            self.collective_knowledge[task_type].append({
                "task_id": task.id,
                "result": result,
                "agent": task.assigned_agent,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_collective_insights(self, topic: str) -> List[Dict]:
        """Get insights from collective knowledge on a topic"""
        insights = []
        for task_type, results in self.collective_knowledge.items():
            if topic.lower() in task_type.lower():
                insights.extend(results)
        return insights
    
    def consensus_decision(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Make a decision through agent consensus"""
        votes = {}
        reasoning = []
        
        for agent_id, agent_info in self.agents.items():
            agent = agent_info["agent"]
            if hasattr(agent, 'think'):
                prompt = f"Question: {question}\nOptions: {options}\nChoose the best option and explain why."
                response = agent.think(prompt)
                
                # Extract vote from response
                for option in options:
                    if option.lower() in str(response).lower():
                        votes[option] = votes.get(option, 0) + 1
                        reasoning.append({
                            "agent": agent_id,
                            "vote": option,
                            "reasoning": response
                        })
                        break
        
        # Determine winner
        winner = max(votes.items(), key=lambda x: x[1])[0] if votes else options[0]
        
        return {
            "decision": winner,
            "votes": votes,
            "reasoning": reasoning,
            "confidence": votes.get(winner, 0) / max(len(self.agents), 1)
        }
    
    # ========================================================================
    # PARALLEL EXECUTION
    # ========================================================================
    
    def execute_parallel(self, tasks: List[SwarmTask]) -> List[Dict]:
        """Execute multiple tasks in parallel"""
        futures = []
        results = []
        
        for task in tasks:
            if self.assign_task(task.id):
                future = self.executor.submit(self.execute_task, task.id)
                futures.append((task.id, future))
        
        for task_id, future in futures:
            try:
                result = future.result(timeout=300)
                results.append({"task_id": task_id, "result": result})
            except Exception as e:
                results.append({"task_id": task_id, "error": str(e)})
        
        return results
    
    def process_queue(self, max_tasks: int = 10) -> List[Dict]:
        """Process tasks from the queue"""
        results = []
        processed = 0
        
        while processed < max_tasks and not self.task_queue.empty():
            try:
                _, task_id = self.task_queue.get_nowait()
                task = self.tasks.get(task_id)
                
                if task and task.status == TaskStatus.PENDING:
                    if self.assign_task(task_id):
                        result = self.execute_task(task_id)
                        results.append({"task_id": task_id, "result": result})
                        processed += 1
            except queue.Empty:
                break
        
        return results
    
    # ========================================================================
    # SWARM COORDINATION
    # ========================================================================
    
    def coordinate_complex_task(
        self,
        description: str,
        decomposition_strategy: Callable[[str], List[Dict]]
    ) -> Dict[str, Any]:
        """
        Coordinate a complex task across the swarm
        1. Decompose into subtasks
        2. Assign to appropriate agents
        3. Execute in parallel where possible
        4. Aggregate results
        """
        # Create main task
        main_task = self.create_task(
            description=description,
            priority=TaskPriority.HIGH
        )
        
        # Decompose task
        subtask_specs = decomposition_strategy(description)
        subtasks = self.decompose_task(main_task, subtask_specs)
        
        # Execute subtasks in parallel
        subtask_results = self.execute_parallel(subtasks)
        
        # Aggregate results
        aggregated = {
            "main_task_id": main_task.id,
            "description": description,
            "subtasks": len(subtasks),
            "results": subtask_results,
            "success_rate": sum(1 for r in subtask_results if "error" not in r) / max(len(subtask_results), 1)
        }
        
        # Update main task
        main_task.output_data = aggregated
        main_task.status = TaskStatus.COMPLETED
        main_task.completed_at = datetime.now().isoformat()
        
        return aggregated
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        with self.lock:
            return {
                "total_agents": len(self.agents),
                "available_agents": len(self.get_available_agents()),
                "total_tasks": len(self.tasks),
                "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
                "failed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
                "collective_knowledge_entries": sum(len(v) for v in self.collective_knowledge.values())
            }
    
    def save_state(self, path: str) -> None:
        """Save swarm state to file"""
        state = {
            "status": self.get_swarm_status(),
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "collective_knowledge": self.collective_knowledge,
            "saved_at": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SWARM INTELLIGENCE ORCHESTRATION - INITIALIZATION")
    print("=" * 70)
    
    # Import agent framework
    import sys
    sys.path.insert(0, '/home/ubuntu/real-asi/autonomous_agents')
    from core_agent_framework import AgentFactory, Industry
    
    # Create orchestrator
    print("\n[1] Creating Swarm Orchestrator...")
    orchestrator = SwarmOrchestrator(max_workers=10)
    print("    ✅ Orchestrator created")
    
    # Create and register agents
    print("\n[2] Registering Agents with Swarm...")
    
    # Create master agent
    master = AgentFactory.create_master_agent()
    orchestrator.register_agent(master.id, master)
    print(f"    ✅ Registered: {master.name}")
    
    # Create industry agents
    for industry in Industry:
        agent = AgentFactory.create_agent(
            name=f"Swarm_{industry.value}_Agent",
            industry=industry
        )
        orchestrator.register_agent(agent.id, agent)
        print(f"    ✅ Registered: {agent.name}")
    
    # Create tasks
    print("\n[3] Creating Sample Tasks...")
    
    tasks = [
        orchestrator.create_task(
            "Analyze market trends for Q4 2024",
            priority=TaskPriority.HIGH,
            required_capabilities=["data_analysis", "research"]
        ),
        orchestrator.create_task(
            "Generate code for automated trading system",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["code_generation", "automation"]
        ),
        orchestrator.create_task(
            "Review legal compliance for new product",
            priority=TaskPriority.HIGH,
            required_capabilities=["research", "decision_making"]
        )
    ]
    
    for task in tasks:
        print(f"    ✅ Task: {task.description[:40]}...")
    
    # Get swarm status
    print("\n[4] Swarm Status:")
    status = orchestrator.get_swarm_status()
    for key, value in status.items():
        print(f"    {key}: {value}")
    
    # Save state
    print("\n[5] Saving Swarm State...")
    save_path = "/home/ubuntu/real-asi/autonomous_agents/swarm/swarm_state.json"
    orchestrator.save_state(save_path)
    print(f"    ✅ Saved to {save_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SWARM ORCHESTRATION SUMMARY")
    print("=" * 70)
    print(f"Total Agents: {status['total_agents']}")
    print(f"Available Agents: {status['available_agents']}")
    print(f"Total Tasks: {status['total_tasks']}")
    print(f"Max Workers: {orchestrator.max_workers}")
    print("Features: Parallel Execution, Collective Intelligence, Consensus")
    print("=" * 70)
