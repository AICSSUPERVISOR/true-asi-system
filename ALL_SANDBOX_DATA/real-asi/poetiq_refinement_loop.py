#!/usr/bin/env python3
"""
Poetiq-Style Refinement Loop for ARC-AGI
Based on: https://github.com/poetiq-ai/poetiq-arc-agi-solver

This implements the core refinement loop that improved Gemini 3 Pro from 31% to 54% on ARC-AGI-2.

Key Concepts:
1. Generate multiple candidate solutions
2. Verify each against training examples
3. Refine based on feedback
4. Iterate until convergence or max iterations
"""

import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import urllib.request
import urllib.error

# Configuration
AIML_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AIML_API_URL = "https://api.anthropic.com/v1/messages"
MAX_REFINEMENT_ITERATIONS = 5
CANDIDATES_PER_ITERATION = 3
TEMPERATURE = 0.7

class SolutionStatus(Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class Candidate:
    """A candidate solution for an ARC task"""
    code: str
    description: str
    confidence: float = 0.0
    status: SolutionStatus = SolutionStatus.PENDING
    test_results: List[bool] = field(default_factory=list)
    iteration: int = 0

@dataclass
class RefinementState:
    """State of the refinement loop"""
    task_id: str
    task_data: Dict[str, Any]
    candidates: List[Candidate] = field(default_factory=list)
    best_candidate: Optional[Candidate] = None
    iteration: int = 0
    converged: bool = False
    feedback_history: List[str] = field(default_factory=list)

def call_llm(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Call LLM API for generation"""
    if not AIML_API_KEY:
        return "# Mock response - API key not configured\ndef solve(grid): return grid"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": AIML_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if system:
        data["system"] = system
    
    try:
        req = urllib.request.Request(
            AIML_API_URL,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("content", [{}])[0].get("text", "")
    except Exception as e:
        return f"# Error: {e}\ndef solve(grid): return grid"

def format_grid(grid: List[List[int]]) -> str:
    """Format a grid for display"""
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def generate_initial_candidates(state: RefinementState) -> List[Candidate]:
    """Generate initial candidate solutions"""
    task = state.task_data
    train_examples = task.get("train", [])
    
    # Format examples
    examples_str = ""
    for i, ex in enumerate(train_examples):
        examples_str += f"\n### Example {i+1}\nInput:\n{format_grid(ex['input'])}\nOutput:\n{format_grid(ex['output'])}\n"
    
    prompt = f"""You are an expert at solving ARC-AGI puzzles. Analyze these input-output examples and write a Python function to solve the pattern.

## Task: {state.task_id}
{examples_str}

## Instructions:
1. Analyze the transformation pattern between inputs and outputs
2. Write a Python function `def solve(grid: List[List[int]]) -> List[List[int]]`
3. The function should work for ANY valid input, not just these examples
4. Be precise and handle edge cases

## Your Solution:
```python
def solve(grid: List[List[int]]) -> List[List[int]]:
    # Your implementation here
```

Provide {CANDIDATES_PER_ITERATION} different solution approaches, each in a separate code block.
"""
    
    system = "You are an expert ARC-AGI solver. Generate precise, working Python code."
    
    response = call_llm(prompt, system, temperature=TEMPERATURE)
    
    # Extract code blocks
    candidates = []
    code_blocks = extract_code_blocks(response)
    
    for i, code in enumerate(code_blocks[:CANDIDATES_PER_ITERATION]):
        candidates.append(Candidate(
            code=code,
            description=f"Candidate {i+1} from initial generation",
            confidence=0.5,
            iteration=0
        ))
    
    # Ensure we have at least one candidate
    if not candidates:
        candidates.append(Candidate(
            code="def solve(grid): return grid",
            description="Fallback identity function",
            confidence=0.1,
            iteration=0
        ))
    
    return candidates

def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text"""
    blocks = []
    in_block = False
    current_block = []
    
    for line in text.split('\n'):
        if line.strip().startswith('```python'):
            in_block = True
            current_block = []
        elif line.strip() == '```' and in_block:
            in_block = False
            if current_block:
                blocks.append('\n'.join(current_block))
        elif in_block:
            current_block.append(line)
    
    # Also try to find standalone functions
    if not blocks:
        lines = text.split('\n')
        current_func = []
        in_func = False
        
        for line in lines:
            if line.strip().startswith('def solve'):
                in_func = True
                current_func = [line]
            elif in_func:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if current_func:
                        blocks.append('\n'.join(current_func))
                    in_func = False
                    current_func = []
                else:
                    current_func.append(line)
        
        if current_func:
            blocks.append('\n'.join(current_func))
    
    return blocks

def verify_candidate(candidate: Candidate, task_data: Dict[str, Any]) -> Tuple[bool, List[bool], str]:
    """Verify a candidate against training examples"""
    train_examples = task_data.get("train", [])
    results = []
    feedback = []
    
    # Create execution environment
    local_env = {"List": list}
    
    try:
        exec(candidate.code, local_env)
        solve_func = local_env.get("solve")
        
        if not solve_func:
            return False, [], "No 'solve' function found in code"
        
        for i, example in enumerate(train_examples):
            try:
                input_grid = example["input"]
                expected_output = example["output"]
                actual_output = solve_func([row[:] for row in input_grid])  # Deep copy
                
                if actual_output == expected_output:
                    results.append(True)
                else:
                    results.append(False)
                    feedback.append(f"Example {i+1}: Expected {expected_output}, got {actual_output}")
            except Exception as e:
                results.append(False)
                feedback.append(f"Example {i+1}: Runtime error - {e}")
        
        all_passed = all(results)
        return all_passed, results, '\n'.join(feedback) if feedback else "All examples passed!"
        
    except SyntaxError as e:
        return False, [], f"Syntax error: {e}"
    except Exception as e:
        return False, [], f"Execution error: {e}"

def refine_candidate(candidate: Candidate, feedback: str, state: RefinementState) -> Candidate:
    """Refine a candidate based on feedback"""
    task = state.task_data
    train_examples = task.get("train", [])
    
    examples_str = ""
    for i, ex in enumerate(train_examples):
        examples_str += f"\n### Example {i+1}\nInput:\n{format_grid(ex['input'])}\nOutput:\n{format_grid(ex['output'])}\n"
    
    prompt = f"""You are refining an ARC-AGI solution that didn't work correctly.

## Task: {state.task_id}
{examples_str}

## Previous Solution:
```python
{candidate.code}
```

## Feedback (what went wrong):
{feedback}

## Previous Refinement History:
{chr(10).join(state.feedback_history[-3:]) if state.feedback_history else 'None'}

## Instructions:
1. Analyze why the previous solution failed
2. Fix the issues based on the feedback
3. Write an improved Python function

## Your Improved Solution:
```python
def solve(grid: List[List[int]]) -> List[List[int]]:
    # Your improved implementation
```
"""
    
    system = "You are an expert ARC-AGI solver. Fix the issues and provide working code."
    
    response = call_llm(prompt, system, temperature=TEMPERATURE * 0.8)  # Lower temp for refinement
    
    code_blocks = extract_code_blocks(response)
    
    if code_blocks:
        return Candidate(
            code=code_blocks[0],
            description=f"Refined from iteration {candidate.iteration}",
            confidence=candidate.confidence * 0.9,  # Slightly lower confidence
            iteration=state.iteration + 1
        )
    
    return candidate  # Return original if refinement failed

def run_refinement_loop(task_id: str, task_data: Dict[str, Any]) -> RefinementState:
    """Run the complete refinement loop for a task"""
    state = RefinementState(task_id=task_id, task_data=task_data)
    
    print(f"\n{'='*60}")
    print(f"Starting refinement loop for task: {task_id}")
    print(f"{'='*60}")
    
    # Phase 1: Generate initial candidates
    print("\n[Phase 1] Generating initial candidates...")
    state.candidates = generate_initial_candidates(state)
    print(f"Generated {len(state.candidates)} candidates")
    
    # Phase 2: Verify and refine
    for iteration in range(MAX_REFINEMENT_ITERATIONS):
        state.iteration = iteration
        print(f"\n[Iteration {iteration + 1}/{MAX_REFINEMENT_ITERATIONS}]")
        
        # Verify all candidates
        for i, candidate in enumerate(state.candidates):
            passed, results, feedback = verify_candidate(candidate, task_data)
            candidate.test_results = results
            
            if passed:
                candidate.status = SolutionStatus.VERIFIED
                candidate.confidence = 1.0
                state.best_candidate = candidate
                state.converged = True
                print(f"  ✅ Candidate {i+1}: VERIFIED (all examples passed)")
            elif any(results):
                candidate.status = SolutionStatus.PARTIAL
                candidate.confidence = sum(results) / len(results)
                print(f"  ⚠️ Candidate {i+1}: PARTIAL ({sum(results)}/{len(results)} passed)")
            else:
                candidate.status = SolutionStatus.FAILED
                candidate.confidence = 0.0
                print(f"  ❌ Candidate {i+1}: FAILED")
            
            # Store feedback
            if feedback and feedback != "All examples passed!":
                state.feedback_history.append(f"Iteration {iteration+1}, Candidate {i+1}: {feedback[:200]}")
        
        # Check for convergence
        if state.converged:
            print(f"\n✅ CONVERGED at iteration {iteration + 1}!")
            break
        
        # Find best candidate for refinement
        best_partial = max(state.candidates, key=lambda c: c.confidence)
        
        if best_partial.confidence > 0:
            print(f"\n  Refining best candidate (confidence: {best_partial.confidence:.2f})...")
            _, _, feedback = verify_candidate(best_partial, task_data)
            refined = refine_candidate(best_partial, feedback, state)
            
            # Replace worst candidate with refined version
            worst_idx = min(range(len(state.candidates)), key=lambda i: state.candidates[i].confidence)
            state.candidates[worst_idx] = refined
        
        # Update best candidate
        current_best = max(state.candidates, key=lambda c: c.confidence)
        if state.best_candidate is None or current_best.confidence > state.best_candidate.confidence:
            state.best_candidate = current_best
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Refinement complete for task: {task_id}")
    print(f"Iterations: {state.iteration + 1}")
    print(f"Converged: {state.converged}")
    if state.best_candidate:
        print(f"Best confidence: {state.best_candidate.confidence:.2f}")
        print(f"Best status: {state.best_candidate.status.value}")
    print(f"{'='*60}")
    
    return state

def solve_arc_task(task_path: str) -> Dict[str, Any]:
    """Solve an ARC task from a JSON file"""
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    task_id = os.path.basename(task_path).replace('.json', '')
    state = run_refinement_loop(task_id, task_data)
    
    result = {
        "task_id": task_id,
        "converged": state.converged,
        "iterations": state.iteration + 1,
        "best_confidence": state.best_candidate.confidence if state.best_candidate else 0,
        "best_code": state.best_candidate.code if state.best_candidate else None,
        "status": state.best_candidate.status.value if state.best_candidate else "failed"
    }
    
    return result

def evaluate_on_dataset(data_dir: str, max_tasks: int = 10) -> Dict[str, Any]:
    """Evaluate on multiple ARC tasks"""
    import glob
    
    task_files = glob.glob(os.path.join(data_dir, "*.json"))[:max_tasks]
    
    results = {
        "total_tasks": len(task_files),
        "solved": 0,
        "partial": 0,
        "failed": 0,
        "task_results": []
    }
    
    for task_file in task_files:
        print(f"\n{'#'*60}")
        print(f"Processing: {os.path.basename(task_file)}")
        print(f"{'#'*60}")
        
        result = solve_arc_task(task_file)
        results["task_results"].append(result)
        
        if result["converged"]:
            results["solved"] += 1
        elif result["best_confidence"] > 0:
            results["partial"] += 1
        else:
            results["failed"] += 1
    
    results["accuracy"] = results["solved"] / results["total_tasks"] if results["total_tasks"] > 0 else 0
    
    return results

if __name__ == "__main__":
    import sys
    
    # Check for ARC-AGI dataset
    arc_training_dir = "/home/ubuntu/ARC-AGI/data/training"
    arc_eval_dir = "/home/ubuntu/ARC-AGI/data/evaluation"
    
    if os.path.exists(arc_training_dir):
        print("="*60)
        print("POETIQ-STYLE REFINEMENT LOOP FOR ARC-AGI")
        print("="*60)
        print(f"\nAPI Key configured: {'Yes' if AIML_API_KEY else 'No (using mock responses)'}")
        print(f"Max iterations: {MAX_REFINEMENT_ITERATIONS}")
        print(f"Candidates per iteration: {CANDIDATES_PER_ITERATION}")
        
        # Run on a few training tasks
        results = evaluate_on_dataset(arc_training_dir, max_tasks=5)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Total tasks: {results['total_tasks']}")
        print(f"Solved: {results['solved']} ({results['accuracy']*100:.1f}%)")
        print(f"Partial: {results['partial']}")
        print(f"Failed: {results['failed']}")
        
        # Save results
        output_file = "/home/ubuntu/real-asi/poetiq_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    else:
        print(f"ARC-AGI dataset not found at {arc_training_dir}")
        print("Please clone: git clone https://github.com/fchollet/ARC-AGI")
