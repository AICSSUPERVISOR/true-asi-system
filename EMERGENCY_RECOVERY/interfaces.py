'''
Core Interfaces for the True ASI System

This file defines the shared data structures, enums, and abstract base classes
used across the entire system to prevent circular dependencies and ensure a clean architecture.
'''

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

class TaskType(Enum):
    """Task types for intelligent model routing"""
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    CHAT = "chat"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"

class ConsensusMethod(Enum):
    """Consensus methods for multi-model responses"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BEST_OF_N = "best_of_n"
    ENSEMBLE = "ensemble"

@dataclass
class UnifiedResponse:
    """Unified response from the entity"""
    response: str
    models_used: List[str]
    consensus_method: Optional[str]
    confidence: float
    total_latency: float
    metadata: Dict[str, Any]

class UnifiedEntityLayer:
    """Abstract base class for the Unified Entity Layer"""
    def process_task(self, task: str, task_type: TaskType, models: List[str]) -> UnifiedResponse:
        raise NotImplementedError
