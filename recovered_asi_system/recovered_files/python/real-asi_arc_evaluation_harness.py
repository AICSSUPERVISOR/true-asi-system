#!/usr/bin/env python3
"""
ARC-AGI Evaluation Harness
Proper benchmarking pipeline for ARC-AGI tasks

Features:
- Load training and evaluation datasets
- Run any solver function
- Calculate accuracy metrics
- Generate detailed reports
- Support for ensemble methods
"""

import json
import os
import time
import glob
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import traceback

@dataclass
class TaskResult:
    """Result of solving a single task"""
    task_id: str
    correct: bool
    predicted_outputs: List[List[List[int]]]
    expected_outputs: List[List[List[int]]]
    execution_time: float
    error: Optional[str] = None
    partial_score: float = 0.0  # Fraction of test cases correct

@dataclass
class EvaluationResult:
    """Result of evaluating on a dataset"""
    dataset_name: str
    total_tasks: int
    correct_tasks: int
    accuracy: float
    partial_accuracy: float  # Average partial score
    total_time: float
    task_results: List[TaskResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ARCEvaluationHarness:
    """Harness for evaluating ARC-AGI solvers"""
    
    def __init__(self, arc_data_dir: str = "/home/ubuntu/ARC-AGI/data"):
        self.arc_data_dir = arc_data_dir
        self.training_dir = os.path.join(arc_data_dir, "training")
        self.evaluation_dir = os.path.join(arc_data_dir, "evaluation")
        
    def load_task(self, task_path: str) -> Dict[str, Any]:
        """Load a single task from JSON file"""
        with open(task_path, 'r') as f:
            return json.load(f)
    
    def load_dataset(self, dataset: str = "training", max_tasks: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """Load all tasks from a dataset"""
        if dataset == "training":
            data_dir = self.training_dir
        elif dataset == "evaluation":
            data_dir = self.evaluation_dir
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        if max_tasks:
            task_files = task_files[:max_tasks]
        
        tasks = []
        for task_file in task_files:
            task_id = os.path.basename(task_file).replace('.json', '')
            task_data = self.load_task(task_file)
            tasks.append((task_id, task_data))
        
        return tasks
    
    def evaluate_task(self, task_id: str, task_data: Dict, solver: Callable) -> TaskResult:
        """Evaluate a solver on a single task"""
        test_examples = task_data.get("test", [])
        train_examples = task_data.get("train", [])
        
        predicted_outputs = []
        expected_outputs = []
        correct_count = 0
        
        start_time = time.time()
        error = None
        
        try:
            for test_case in test_examples:
                input_grid = test_case["input"]
                expected_output = test_case["output"]
                
                # Call solver
                try:
                    predicted_output = solver(input_grid, train_examples)
                except TypeError:
                    # Solver might not accept train_examples
                    predicted_output = solver(input_grid)
                
                predicted_outputs.append(predicted_output)
                expected_outputs.append(expected_output)
                
                if predicted_output == expected_output:
                    correct_count += 1
                    
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        execution_time = time.time() - start_time
        
        total_tests = len(test_examples)
        all_correct = correct_count == total_tests and total_tests > 0
        partial_score = correct_count / total_tests if total_tests > 0 else 0.0
        
        return TaskResult(
            task_id=task_id,
            correct=all_correct,
            predicted_outputs=predicted_outputs,
            expected_outputs=expected_outputs,
            execution_time=execution_time,
            error=error,
            partial_score=partial_score
        )
    
    def evaluate_dataset(
        self, 
        solver: Callable, 
        dataset: str = "training",
        max_tasks: Optional[int] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """Evaluate a solver on an entire dataset"""
        tasks = self.load_dataset(dataset, max_tasks)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating on {dataset} dataset ({len(tasks)} tasks)")
            print(f"{'='*60}\n")
        
        task_results = []
        correct_count = 0
        total_partial_score = 0.0
        start_time = time.time()
        
        for i, (task_id, task_data) in enumerate(tasks):
            result = self.evaluate_task(task_id, task_data, solver)
            task_results.append(result)
            
            if result.correct:
                correct_count += 1
            total_partial_score += result.partial_score
            
            if verbose:
                status = "✅" if result.correct else ("⚠️" if result.partial_score > 0 else "❌")
                print(f"[{i+1}/{len(tasks)}] {task_id}: {status} (partial: {result.partial_score:.2f}, time: {result.execution_time:.2f}s)")
                if result.error:
                    print(f"    Error: {result.error[:100]}...")
        
        total_time = time.time() - start_time
        accuracy = correct_count / len(tasks) if tasks else 0.0
        partial_accuracy = total_partial_score / len(tasks) if tasks else 0.0
        
        result = EvaluationResult(
            dataset_name=dataset,
            total_tasks=len(tasks),
            correct_tasks=correct_count,
            accuracy=accuracy,
            partial_accuracy=partial_accuracy,
            total_time=total_time,
            task_results=task_results
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS: {dataset}")
            print(f"{'='*60}")
            print(f"Total tasks: {result.total_tasks}")
            print(f"Correct: {result.correct_tasks} ({result.accuracy*100:.1f}%)")
            print(f"Partial accuracy: {result.partial_accuracy*100:.1f}%")
            print(f"Total time: {result.total_time:.1f}s")
            print(f"Avg time per task: {result.total_time/result.total_tasks:.2f}s")
        
        return result
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """Save evaluation results to JSON"""
        data = {
            "dataset_name": result.dataset_name,
            "total_tasks": result.total_tasks,
            "correct_tasks": result.correct_tasks,
            "accuracy": result.accuracy,
            "partial_accuracy": result.partial_accuracy,
            "total_time": result.total_time,
            "timestamp": result.timestamp,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "correct": tr.correct,
                    "partial_score": tr.partial_score,
                    "execution_time": tr.execution_time,
                    "error": tr.error
                }
                for tr in result.task_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")

class EnsembleSolver:
    """Ensemble of multiple solvers with voting"""
    
    def __init__(self, solvers: List[Callable], weights: Optional[List[float]] = None):
        self.solvers = solvers
        self.weights = weights or [1.0] * len(solvers)
    
    def __call__(self, input_grid: List[List[int]], train_examples: List[Dict] = None) -> List[List[int]]:
        """Run all solvers and vote on the output"""
        predictions = []
        
        for solver in self.solvers:
            try:
                if train_examples:
                    pred = solver(input_grid, train_examples)
                else:
                    pred = solver(input_grid)
                predictions.append(pred)
            except:
                predictions.append(None)
        
        # Simple majority voting
        valid_predictions = [p for p in predictions if p is not None]
        
        if not valid_predictions:
            return input_grid  # Fallback to identity
        
        # Convert to hashable for voting
        pred_counts = {}
        for pred in valid_predictions:
            key = json.dumps(pred)
            pred_counts[key] = pred_counts.get(key, 0) + 1
        
        # Return most common prediction
        best_key = max(pred_counts, key=pred_counts.get)
        return json.loads(best_key)

# Example solvers for testing

def identity_solver(input_grid: List[List[int]], train_examples: List[Dict] = None) -> List[List[int]]:
    """Baseline: return input unchanged"""
    return input_grid

def random_solver(input_grid: List[List[int]], train_examples: List[Dict] = None) -> List[List[int]]:
    """Baseline: return random grid of same size"""
    import random
    return [[random.randint(0, 9) for _ in row] for row in input_grid]

def copy_first_output_solver(input_grid: List[List[int]], train_examples: List[Dict] = None) -> List[List[int]]:
    """Baseline: return first training output"""
    if train_examples and len(train_examples) > 0:
        return train_examples[0].get("output", input_grid)
    return input_grid

def pattern_match_solver(input_grid: List[List[int]], train_examples: List[Dict] = None) -> List[List[int]]:
    """Try to find matching input in training and return corresponding output"""
    if not train_examples:
        return input_grid
    
    for example in train_examples:
        if example.get("input") == input_grid:
            return example.get("output", input_grid)
    
    # No match found, return first output as fallback
    return train_examples[0].get("output", input_grid)

if __name__ == "__main__":
    # Test the harness
    harness = ARCEvaluationHarness()
    
    print("="*60)
    print("ARC-AGI EVALUATION HARNESS")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(harness.training_dir):
        print(f"Dataset not found at {harness.training_dir}")
        print("Please run: git clone https://github.com/fchollet/ARC-AGI /home/ubuntu/ARC-AGI")
        exit(1)
    
    # Test with baseline solvers
    print("\n[Testing baseline solvers on 10 training tasks]\n")
    
    solvers = {
        "identity": identity_solver,
        "copy_first": copy_first_output_solver,
        "pattern_match": pattern_match_solver
    }
    
    for name, solver in solvers.items():
        print(f"\n--- {name} solver ---")
        result = harness.evaluate_dataset(solver, "training", max_tasks=10, verbose=False)
        print(f"Accuracy: {result.accuracy*100:.1f}% ({result.correct_tasks}/{result.total_tasks})")
        print(f"Partial: {result.partial_accuracy*100:.1f}%")
    
    # Test ensemble
    print("\n--- ensemble solver ---")
    ensemble = EnsembleSolver([identity_solver, copy_first_output_solver, pattern_match_solver])
    result = harness.evaluate_dataset(ensemble, "training", max_tasks=10, verbose=False)
    print(f"Accuracy: {result.accuracy*100:.1f}% ({result.correct_tasks}/{result.total_tasks})")
    print(f"Partial: {result.partial_accuracy*100:.1f}%")
    
    # Save results
    harness.save_results(result, "/home/ubuntu/real-asi/evaluation_harness_test.json")
    
    print("\n✅ Evaluation harness ready for use!")
