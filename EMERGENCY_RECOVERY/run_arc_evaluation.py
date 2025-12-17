#!/usr/bin/env python3
"""
ASI ARC-AGI EVALUATION SYSTEM
Target: 85%+ Accuracy (Superhuman)

This script runs all 4 models on ARC-AGI benchmark and combines
them using ensemble voting to achieve maximum accuracy.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    name: str
    path: str
    expected_accuracy: float
    weight: float  # Ensemble weight based on expected performance
    max_tokens: int = 4096
    temperature: float = 0.0

MODELS = {
    "marc-8B": ModelConfig(
        name="MARC-8B (MIT TTT)",
        path="/workspace/models/marc-8B",
        expected_accuracy=0.628,
        weight=0.35  # Highest weight - best performer
    ),
    "nvarc": ModelConfig(
        name="NVARC (NVIDIA)",
        path="/workspace/models/nvarc",
        expected_accuracy=0.55,
        weight=0.25
    ),
    "qwen3-8B": ModelConfig(
        name="Qwen3-8B",
        path="/workspace/models/qwen3-8B",
        expected_accuracy=0.45,
        weight=0.20
    ),
    "deepseek-coder": ModelConfig(
        name="DeepSeek-Coder-V2",
        path="/workspace/models/deepseek-coder-v2",
        expected_accuracy=0.40,
        weight=0.20
    )
}

ARC_DATA_PATH = "/workspace/ARC-AGI/data"
RESULTS_PATH = "/workspace/arc_agi_results.json"

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(config: ModelConfig):
    """Load a model for inference."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading {config.name} from {config.path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"✅ {config.name} loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load {config.name}: {e}")
        return None, None

# ============================================================
# ARC-AGI TASK PROCESSING
# ============================================================

def load_arc_tasks(split: str = "evaluation") -> List[Dict]:
    """Load ARC-AGI tasks from dataset."""
    tasks = []
    task_dir = Path(ARC_DATA_PATH) / split
    
    if not task_dir.exists():
        # Try alternative path
        task_dir = Path("/workspace/ARC-AGI-1/data") / split
    
    if not task_dir.exists():
        logger.error(f"Task directory not found: {task_dir}")
        return tasks
    
    for task_file in sorted(task_dir.glob("*.json")):
        with open(task_file) as f:
            task = json.load(f)
            task["id"] = task_file.stem
            tasks.append(task)
    
    logger.info(f"Loaded {len(tasks)} tasks from {split}")
    return tasks

def format_arc_prompt(task: Dict) -> str:
    """Format ARC task as prompt for LLM."""
    prompt = """You are an expert at solving ARC-AGI puzzles. 

Given input-output examples, find the transformation pattern and apply it to the test input.

Training Examples:
"""
    
    for i, example in enumerate(task.get("train", [])):
        prompt += f"\nExample {i+1}:\n"
        prompt += f"Input:\n{json.dumps(example['input'])}\n"
        prompt += f"Output:\n{json.dumps(example['output'])}\n"
    
    prompt += "\nTest Input:\n"
    prompt += f"{json.dumps(task['test'][0]['input'])}\n"
    
    prompt += """
Analyze the pattern and provide ONLY the output grid as a JSON array.
Output:"""
    
    return prompt

def parse_grid_response(response: str) -> Optional[List[List[int]]]:
    """Parse LLM response to extract grid."""
    try:
        # Try to find JSON array in response
        import re
        
        # Look for [[...]] pattern
        match = re.search(r'\[\s*\[.*?\]\s*\]', response, re.DOTALL)
        if match:
            grid = json.loads(match.group())
            # Validate it's a proper grid
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                return grid
        
        return None
    except:
        return None

def evaluate_prediction(predicted: List[List[int]], expected: List[List[int]]) -> bool:
    """Check if prediction matches expected output."""
    if predicted is None:
        return False
    return predicted == expected

# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, tokenizer, prompt: str, config: ModelConfig) -> str:
    """Run inference on a single prompt."""
    try:
        import torch
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return ""

# ============================================================
# ENSEMBLE VOTING
# ============================================================

def ensemble_vote(predictions: Dict[str, List[List[int]]], 
                  weights: Dict[str, float]) -> Optional[List[List[int]]]:
    """Combine predictions using weighted voting."""
    if not predictions:
        return None
    
    # Convert grids to hashable format for voting
    grid_votes = {}
    
    for model_name, grid in predictions.items():
        if grid is None:
            continue
        
        grid_key = json.dumps(grid)
        weight = weights.get(model_name, 0.25)
        
        if grid_key not in grid_votes:
            grid_votes[grid_key] = {"grid": grid, "weight": 0, "count": 0}
        
        grid_votes[grid_key]["weight"] += weight
        grid_votes[grid_key]["count"] += 1
    
    if not grid_votes:
        return None
    
    # Return grid with highest weighted vote
    best = max(grid_votes.values(), key=lambda x: (x["weight"], x["count"]))
    return best["grid"]

# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_all_models():
    """Run full evaluation on all models."""
    logger.info("=" * 60)
    logger.info("ASI ARC-AGI EVALUATION - TARGET: 85%+ ACCURACY")
    logger.info("=" * 60)
    
    # Load tasks
    tasks = load_arc_tasks("evaluation")
    if not tasks:
        logger.error("No tasks loaded!")
        return
    
    # Load all models
    loaded_models = {}
    for model_key, config in MODELS.items():
        if Path(config.path).exists():
            model, tokenizer = load_model(config)
            if model is not None:
                loaded_models[model_key] = (model, tokenizer, config)
    
    if not loaded_models:
        logger.error("No models loaded!")
        return
    
    logger.info(f"Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")
    
    # Evaluate
    results = {
        "total_tasks": len(tasks),
        "models": {},
        "ensemble": {"correct": 0, "total": 0},
        "per_task": []
    }
    
    weights = {k: MODELS[k].weight for k in loaded_models}
    
    for task_idx, task in enumerate(tasks):
        task_id = task["id"]
        expected = task["test"][0].get("output")
        
        if expected is None:
            continue
        
        prompt = format_arc_prompt(task)
        predictions = {}
        
        # Get prediction from each model
        for model_key, (model, tokenizer, config) in loaded_models.items():
            try:
                response = run_inference(model, tokenizer, prompt, config)
                grid = parse_grid_response(response)
                predictions[model_key] = grid
                
                # Track individual model performance
                if model_key not in results["models"]:
                    results["models"][model_key] = {"correct": 0, "total": 0}
                
                results["models"][model_key]["total"] += 1
                if evaluate_prediction(grid, expected):
                    results["models"][model_key]["correct"] += 1
                    
            except Exception as e:
                logger.error(f"Error with {model_key} on task {task_id}: {e}")
                predictions[model_key] = None
        
        # Ensemble prediction
        ensemble_pred = ensemble_vote(predictions, weights)
        ensemble_correct = evaluate_prediction(ensemble_pred, expected)
        
        results["ensemble"]["total"] += 1
        if ensemble_correct:
            results["ensemble"]["correct"] += 1
        
        # Log progress
        if (task_idx + 1) % 10 == 0:
            acc = results["ensemble"]["correct"] / results["ensemble"]["total"] * 100
            logger.info(f"Progress: {task_idx + 1}/{len(tasks)} - Ensemble accuracy: {acc:.1f}%")
        
        results["per_task"].append({
            "task_id": task_id,
            "predictions": {k: v is not None for k, v in predictions.items()},
            "ensemble_correct": ensemble_correct
        })
    
    # Calculate final accuracies
    for model_key in results["models"]:
        m = results["models"][model_key]
        m["accuracy"] = m["correct"] / m["total"] if m["total"] > 0 else 0
    
    results["ensemble"]["accuracy"] = (
        results["ensemble"]["correct"] / results["ensemble"]["total"]
        if results["ensemble"]["total"] > 0 else 0
    )
    
    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    
    for model_key, stats in results["models"].items():
        logger.info(f"{model_key}: {stats['correct']}/{stats['total']} = {stats['accuracy']*100:.1f}%")
    
    logger.info("-" * 40)
    logger.info(f"ENSEMBLE: {results['ensemble']['correct']}/{results['ensemble']['total']} = {results['ensemble']['accuracy']*100:.1f}%")
    logger.info("=" * 60)
    
    if results["ensemble"]["accuracy"] >= 0.85:
        logger.info("🎉 SUPERHUMAN PERFORMANCE ACHIEVED! (85%+)")
    elif results["ensemble"]["accuracy"] >= 0.70:
        logger.info("✅ Excellent performance (70%+)")
    elif results["ensemble"]["accuracy"] >= 0.50:
        logger.info("⚠️ Good performance (50%+)")
    else:
        logger.info("❌ Below target - needs improvement")
    
    return results

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        results = evaluate_all_models()
        if results:
            print(f"\nResults saved to: {RESULTS_PATH}")
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
