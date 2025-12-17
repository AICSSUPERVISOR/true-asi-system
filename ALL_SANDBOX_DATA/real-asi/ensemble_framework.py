#!/usr/bin/env python3
"""
ARC-AGI Ensemble Framework
Combines multiple solvers with weighted voting and confidence estimation

Features:
1. Multiple solver integration (LLM, TRM, CompressARC, etc.)
2. Weighted voting based on solver confidence
3. Consistency checking across solvers
4. Fallback strategies
5. Result aggregation and reporting
"""

import json
import os
import time
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import hashlib

Grid = List[List[int]]

@dataclass
class SolverResult:
    """Result from a single solver"""
    solver_name: str
    prediction: Optional[Grid]
    confidence: float
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleResult:
    """Result from ensemble voting"""
    task_id: str
    final_prediction: Optional[Grid]
    final_confidence: float
    voting_method: str
    solver_results: List[SolverResult]
    agreement_score: float
    execution_time: float

class Solver:
    """Base class for ARC solvers"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        """
        Solve an ARC task
        
        Args:
            input_grid: The input grid to transform
            train_examples: Training examples for this task
        
        Returns:
            (prediction, confidence) tuple
        """
        raise NotImplementedError

class IdentitySolver(Solver):
    """Baseline: return input unchanged"""
    
    def __init__(self):
        super().__init__("identity", weight=0.1)
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        return input_grid, 0.1

class CopyOutputSolver(Solver):
    """Baseline: return first training output"""
    
    def __init__(self):
        super().__init__("copy_output", weight=0.2)
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        if train_examples:
            return train_examples[0].get("output", input_grid), 0.2
        return input_grid, 0.1

class PatternMatchSolver(Solver):
    """Match input to training examples"""
    
    def __init__(self):
        super().__init__("pattern_match", weight=0.3)
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        for example in train_examples:
            if example.get("input") == input_grid:
                return example.get("output"), 0.9
        
        # No exact match, return most similar output
        if train_examples:
            return train_examples[0].get("output", input_grid), 0.2
        return input_grid, 0.1

class TransformationSolver(Solver):
    """Try common transformations"""
    
    def __init__(self):
        super().__init__("transformation", weight=0.4)
        self.transformations = [
            ("identity", lambda g: g),
            ("rotate_90", self._rotate_90),
            ("rotate_180", self._rotate_180),
            ("rotate_270", self._rotate_270),
            ("flip_h", self._flip_h),
            ("flip_v", self._flip_v),
            ("transpose", self._transpose),
        ]
    
    def _rotate_90(self, grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]
    
    def _rotate_180(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid[::-1]]
    
    def _rotate_270(self, grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0])
        return [[grid[j][cols-1-i] for j in range(rows)] for i in range(cols)]
    
    def _flip_h(self, grid: Grid) -> Grid:
        return [row[::-1] for row in grid]
    
    def _flip_v(self, grid: Grid) -> Grid:
        return grid[::-1]
    
    def _transpose(self, grid: Grid) -> Grid:
        return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        # Find which transformation works on training examples
        for name, transform in self.transformations:
            all_match = True
            for example in train_examples:
                try:
                    transformed = transform(example["input"])
                    if transformed != example["output"]:
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            
            if all_match:
                try:
                    return transform(input_grid), 0.8
                except:
                    pass
        
        return input_grid, 0.1

class LLMSolver(Solver):
    """LLM-based solver using API"""
    
    def __init__(self, api_key: str = "", model: str = "claude-3-5-sonnet-20241022"):
        super().__init__("llm", weight=0.6)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
    
    def solve(self, input_grid: Grid, train_examples: List[Dict]) -> Tuple[Optional[Grid], float]:
        if not self.api_key:
            return None, 0.0
        
        # Format prompt
        examples_str = ""
        for i, ex in enumerate(train_examples):
            examples_str += f"\nExample {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}\n"
        
        prompt = f"""Solve this ARC-AGI puzzle. Given the training examples, predict the output for the test input.

Training Examples:
{examples_str}

Test Input:
{input_grid}

Respond with ONLY the output grid as a Python list of lists. No explanation."""
        
        import urllib.request
        import urllib.error
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result.get("content", [{}])[0].get("text", "")
                
                # Parse the grid
                try:
                    # Find list in response
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    if start >= 0 and end > start:
                        grid_str = text[start:end]
                        grid = json.loads(grid_str)
                        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                            return grid, 0.5
                except:
                    pass
                
                return None, 0.0
        except Exception as e:
            return None, 0.0

class EnsembleFramework:
    """Framework for combining multiple ARC solvers"""
    
    def __init__(self, solvers: List[Solver] = None):
        self.solvers = solvers or [
            IdentitySolver(),
            CopyOutputSolver(),
            PatternMatchSolver(),
            TransformationSolver(),
            LLMSolver()
        ]
    
    def add_solver(self, solver: Solver):
        """Add a solver to the ensemble"""
        self.solvers.append(solver)
    
    def remove_solver(self, name: str):
        """Remove a solver by name"""
        self.solvers = [s for s in self.solvers if s.name != name]
    
    def _grid_to_hash(self, grid: Optional[Grid]) -> str:
        """Convert grid to hash for comparison"""
        if grid is None:
            return "none"
        return hashlib.md5(json.dumps(grid).encode()).hexdigest()
    
    def _weighted_vote(self, results: List[SolverResult]) -> Tuple[Optional[Grid], float]:
        """Weighted voting across solver results"""
        # Group predictions by hash
        votes: Dict[str, Tuple[Grid, float]] = {}
        
        for result in results:
            if result.prediction is None:
                continue
            
            grid_hash = self._grid_to_hash(result.prediction)
            solver = next((s for s in self.solvers if s.name == result.solver_name), None)
            weight = solver.weight if solver else 1.0
            weighted_confidence = result.confidence * weight
            
            if grid_hash in votes:
                _, current_conf = votes[grid_hash]
                votes[grid_hash] = (result.prediction, current_conf + weighted_confidence)
            else:
                votes[grid_hash] = (result.prediction, weighted_confidence)
        
        if not votes:
            return None, 0.0
        
        # Find best prediction
        best_hash = max(votes, key=lambda h: votes[h][1])
        best_grid, best_conf = votes[best_hash]
        
        # Normalize confidence
        total_conf = sum(v[1] for v in votes.values())
        normalized_conf = best_conf / total_conf if total_conf > 0 else 0.0
        
        return best_grid, normalized_conf
    
    def _majority_vote(self, results: List[SolverResult]) -> Tuple[Optional[Grid], float]:
        """Simple majority voting"""
        predictions = [r.prediction for r in results if r.prediction is not None]
        
        if not predictions:
            return None, 0.0
        
        # Count occurrences
        pred_hashes = [self._grid_to_hash(p) for p in predictions]
        counter = Counter(pred_hashes)
        
        most_common_hash, count = counter.most_common(1)[0]
        
        # Find the actual grid
        for pred, h in zip(predictions, pred_hashes):
            if h == most_common_hash:
                confidence = count / len(predictions)
                return pred, confidence
        
        return None, 0.0
    
    def _highest_confidence(self, results: List[SolverResult]) -> Tuple[Optional[Grid], float]:
        """Select prediction with highest confidence"""
        valid_results = [r for r in results if r.prediction is not None]
        
        if not valid_results:
            return None, 0.0
        
        best = max(valid_results, key=lambda r: r.confidence)
        return best.prediction, best.confidence
    
    def _calculate_agreement(self, results: List[SolverResult]) -> float:
        """Calculate agreement score across solvers"""
        predictions = [r.prediction for r in results if r.prediction is not None]
        
        if len(predictions) <= 1:
            return 1.0
        
        # Count unique predictions
        pred_hashes = [self._grid_to_hash(p) for p in predictions]
        unique_count = len(set(pred_hashes))
        
        # Agreement = 1 - (unique - 1) / (total - 1)
        agreement = 1.0 - (unique_count - 1) / (len(predictions) - 1)
        return max(0.0, agreement)
    
    def solve(
        self,
        task_id: str,
        input_grid: Grid,
        train_examples: List[Dict],
        voting_method: str = "weighted"
    ) -> EnsembleResult:
        """
        Solve a task using the ensemble
        
        Args:
            task_id: Task identifier
            input_grid: Input grid to transform
            train_examples: Training examples
            voting_method: "weighted", "majority", or "highest"
        
        Returns:
            EnsembleResult with final prediction and metadata
        """
        start_time = time.time()
        solver_results = []
        
        # Run all solvers
        for solver in self.solvers:
            solver_start = time.time()
            try:
                prediction, confidence = solver.solve(input_grid, train_examples)
                solver_results.append(SolverResult(
                    solver_name=solver.name,
                    prediction=prediction,
                    confidence=confidence,
                    execution_time=time.time() - solver_start
                ))
            except Exception as e:
                solver_results.append(SolverResult(
                    solver_name=solver.name,
                    prediction=None,
                    confidence=0.0,
                    execution_time=time.time() - solver_start,
                    error=str(e)
                ))
        
        # Vote on final prediction
        if voting_method == "weighted":
            final_prediction, final_confidence = self._weighted_vote(solver_results)
        elif voting_method == "majority":
            final_prediction, final_confidence = self._majority_vote(solver_results)
        else:  # highest
            final_prediction, final_confidence = self._highest_confidence(solver_results)
        
        # Calculate agreement
        agreement = self._calculate_agreement(solver_results)
        
        return EnsembleResult(
            task_id=task_id,
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            voting_method=voting_method,
            solver_results=solver_results,
            agreement_score=agreement,
            execution_time=time.time() - start_time
        )
    
    def evaluate(
        self,
        tasks: List[Tuple[str, Dict]],
        voting_method: str = "weighted",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble on multiple tasks
        
        Args:
            tasks: List of (task_id, task_data) tuples
            voting_method: Voting method to use
            verbose: Print progress
        
        Returns:
            Evaluation results dictionary
        """
        results = {
            "total_tasks": len(tasks),
            "correct": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "avg_agreement": 0.0,
            "solver_contributions": {s.name: 0 for s in self.solvers},
            "task_results": []
        }
        
        total_confidence = 0.0
        total_agreement = 0.0
        
        for i, (task_id, task_data) in enumerate(tasks):
            test_examples = task_data.get("test", [])
            train_examples = task_data.get("train", [])
            
            task_correct = True
            for test_ex in test_examples:
                input_grid = test_ex["input"]
                expected_output = test_ex["output"]
                
                result = self.solve(task_id, input_grid, train_examples, voting_method)
                
                if result.final_prediction == expected_output:
                    # Track which solver contributed
                    for sr in result.solver_results:
                        if sr.prediction == expected_output:
                            results["solver_contributions"][sr.solver_name] += 1
                else:
                    task_correct = False
                
                total_confidence += result.final_confidence
                total_agreement += result.agreement_score
            
            if task_correct:
                results["correct"] += 1
            
            results["task_results"].append({
                "task_id": task_id,
                "correct": task_correct,
                "confidence": result.final_confidence,
                "agreement": result.agreement_score
            })
            
            if verbose:
                status = "✅" if task_correct else "❌"
                print(f"[{i+1}/{len(tasks)}] {task_id}: {status} (conf: {result.final_confidence:.2f}, agree: {result.agreement_score:.2f})")
        
        results["accuracy"] = results["correct"] / len(tasks) if tasks else 0.0
        results["avg_confidence"] = total_confidence / (len(tasks) * len(test_examples)) if tasks else 0.0
        results["avg_agreement"] = total_agreement / (len(tasks) * len(test_examples)) if tasks else 0.0
        
        return results

def run_ensemble_evaluation():
    """Run ensemble evaluation on ARC-AGI"""
    import glob
    
    arc_training_dir = "/home/ubuntu/ARC-AGI/data/training"
    
    if not os.path.exists(arc_training_dir):
        print(f"Dataset not found at {arc_training_dir}")
        return None
    
    print("="*60)
    print("ARC-AGI ENSEMBLE FRAMEWORK EVALUATION")
    print("="*60)
    
    # Load tasks
    task_files = sorted(glob.glob(os.path.join(arc_training_dir, "*.json")))[:10]
    tasks = []
    for task_file in task_files:
        task_id = os.path.basename(task_file).replace('.json', '')
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        tasks.append((task_id, task_data))
    
    print(f"\nTasks: {len(tasks)}")
    
    # Create ensemble
    ensemble = EnsembleFramework()
    print(f"Solvers: {[s.name for s in ensemble.solvers]}")
    
    # Evaluate
    print("\n[Evaluating with weighted voting]\n")
    results = ensemble.evaluate(tasks, voting_method="weighted", verbose=True)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"Accuracy: {results['accuracy']*100:.1f}% ({results['correct']}/{results['total_tasks']})")
    print(f"Avg Confidence: {results['avg_confidence']:.2f}")
    print(f"Avg Agreement: {results['avg_agreement']:.2f}")
    print(f"\nSolver Contributions:")
    for solver, count in results["solver_contributions"].items():
        print(f"  {solver}: {count} correct predictions")
    
    return results

if __name__ == "__main__":
    results = run_ensemble_evaluation()
    
    if results:
        # Save results
        output_file = "/home/ubuntu/real-asi/ensemble_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
