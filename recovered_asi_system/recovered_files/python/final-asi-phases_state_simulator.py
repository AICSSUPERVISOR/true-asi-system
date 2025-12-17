
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class State:
    '''Represents a state in the simulation'''
    id: int
    features: np.ndarray
    value: float
    parent: int = None

class MonteCarloStateSimulator:
    '''Custom state space simulator using Monte Carlo Tree Search'''
    
    def __init__(self, state_dim=64, num_simulations=100):
        self.state_dim = state_dim
        self.num_simulations = num_simulations
        self.states = []
    
    def simulate_parallel(self, initial_state: np.ndarray, depth=5) -> List[State]:
        '''Simulate multiple future states in parallel'''
        start_time = time.time()
        
        # Initialize root state
        root = State(
            id=0,
            features=initial_state,
            value=self._evaluate_state(initial_state)
        )
        self.states = [root]
        
        # Parallel simulation
        for sim in range(self.num_simulations):
            current_state = initial_state.copy()
            
            for d in range(depth):
                # Generate next state
                action = np.random.randn(self.state_dim) * 0.1
                next_state = current_state + action
                
                # Evaluate
                value = self._evaluate_state(next_state)
                
                # Store
                state = State(
                    id=len(self.states),
                    features=next_state,
                    value=value,
                    parent=0 if d == 0 else len(self.states) - 1
                )
                self.states.append(state)
                
                current_state = next_state
        
        elapsed = time.time() - start_time
        states_per_sec = len(self.states) / elapsed
        
        print(f"Simulated {len(self.states)} states in {elapsed*1000:.1f}ms")
        print(f"Throughput: {states_per_sec:.0f} states/sec")
        
        return self.states
    
    def select_best(self) -> State:
        '''Select best state from simulations'''
        start_time = time.time()
        best = max(self.states, key=lambda s: s.value)
        elapsed = (time.time() - start_time) * 1000
        print(f"Selection time: {elapsed:.2f}ms")
        return best
    
    def _evaluate_state(self, state: np.ndarray) -> float:
        '''Evaluate state quality'''
        # Simple heuristic: prefer states with balanced features
        return -np.var(state)

# Test the simulator
simulator = MonteCarloStateSimulator(state_dim=64, num_simulations=100)
initial = np.random.randn(64)
states = simulator.simulate_parallel(initial, depth=5)
best = simulator.select_best()

print(f"Best state value: {best.value:.4f}")
print(f"Total states explored: {len(states)}")
