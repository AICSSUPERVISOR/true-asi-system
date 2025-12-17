#!/usr/bin/env python3
"""
ARC-AGI Training Data Pipeline
Augmentation and synthetic data generation for improved training

Features:
1. Data augmentation (rotation, flip, color permutation)
2. Synthetic task generation
3. Curriculum learning ordering
4. Train/validation splits
5. Export to various formats (JSON, HuggingFace, PyTorch)
"""

import json
import os
import random
import glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import hashlib

Grid = List[List[int]]

@dataclass
class AugmentedTask:
    """An augmented ARC task"""
    original_id: str
    augmentation: str
    train: List[Dict[str, Grid]]
    test: List[Dict[str, Grid]]
    difficulty: float = 0.5
    
    @property
    def task_id(self) -> str:
        return f"{self.original_id}_{self.augmentation}"

class DataAugmenter:
    """Augmentation operations for ARC grids"""
    
    @staticmethod
    def rotate_90(grid: Grid) -> Grid:
        """Rotate grid 90 degrees clockwise"""
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]
    
    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        """Rotate grid 180 degrees"""
        return [row[::-1] for row in grid[::-1]]
    
    @staticmethod
    def rotate_270(grid: Grid) -> Grid:
        """Rotate grid 270 degrees clockwise"""
        rows, cols = len(grid), len(grid[0])
        return [[grid[j][cols-1-i] for j in range(rows)] for i in range(cols)]
    
    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        """Flip grid vertically"""
        return grid[::-1]
    
    @staticmethod
    def transpose(grid: Grid) -> Grid:
        """Transpose grid"""
        return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
    
    @staticmethod
    def permute_colors(grid: Grid, permutation: Dict[int, int]) -> Grid:
        """Apply color permutation"""
        return [[permutation.get(c, c) for c in row] for row in grid]
    
    @staticmethod
    def generate_color_permutation(exclude_zero: bool = True) -> Dict[int, int]:
        """Generate a random color permutation"""
        colors = list(range(1, 10)) if exclude_zero else list(range(10))
        shuffled = colors[:]
        random.shuffle(shuffled)
        perm = {c: s for c, s in zip(colors, shuffled)}
        if exclude_zero:
            perm[0] = 0  # Keep background unchanged
        return perm

class TrainingDataPipeline:
    """Pipeline for preparing ARC-AGI training data"""
    
    def __init__(self, arc_data_dir: str = "/home/ubuntu/ARC-AGI/data"):
        self.arc_data_dir = arc_data_dir
        self.training_dir = os.path.join(arc_data_dir, "training")
        self.evaluation_dir = os.path.join(arc_data_dir, "evaluation")
        self.augmenter = DataAugmenter()
        
    def load_task(self, task_path: str) -> Dict[str, Any]:
        """Load a single task"""
        with open(task_path, 'r') as f:
            return json.load(f)
    
    def load_all_tasks(self, split: str = "training") -> List[Tuple[str, Dict]]:
        """Load all tasks from a split"""
        data_dir = self.training_dir if split == "training" else self.evaluation_dir
        task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        tasks = []
        for task_file in task_files:
            task_id = os.path.basename(task_file).replace('.json', '')
            task_data = self.load_task(task_file)
            tasks.append((task_id, task_data))
        
        return tasks
    
    def augment_example(self, example: Dict[str, Grid], augmentation: str) -> Dict[str, Grid]:
        """Apply augmentation to a single example"""
        input_grid = example["input"]
        output_grid = example["output"]
        
        if augmentation == "original":
            return example
        elif augmentation == "rot90":
            return {
                "input": self.augmenter.rotate_90(input_grid),
                "output": self.augmenter.rotate_90(output_grid)
            }
        elif augmentation == "rot180":
            return {
                "input": self.augmenter.rotate_180(input_grid),
                "output": self.augmenter.rotate_180(output_grid)
            }
        elif augmentation == "rot270":
            return {
                "input": self.augmenter.rotate_270(input_grid),
                "output": self.augmenter.rotate_270(output_grid)
            }
        elif augmentation == "flip_h":
            return {
                "input": self.augmenter.flip_horizontal(input_grid),
                "output": self.augmenter.flip_horizontal(output_grid)
            }
        elif augmentation == "flip_v":
            return {
                "input": self.augmenter.flip_vertical(input_grid),
                "output": self.augmenter.flip_vertical(output_grid)
            }
        elif augmentation == "transpose":
            return {
                "input": self.augmenter.transpose(input_grid),
                "output": self.augmenter.transpose(output_grid)
            }
        elif augmentation.startswith("color_"):
            # Color permutation
            seed = int(augmentation.split("_")[1])
            random.seed(seed)
            perm = self.augmenter.generate_color_permutation()
            return {
                "input": self.augmenter.permute_colors(input_grid, perm),
                "output": self.augmenter.permute_colors(output_grid, perm)
            }
        else:
            return example
    
    def augment_task(self, task_id: str, task_data: Dict, augmentations: List[str]) -> List[AugmentedTask]:
        """Apply multiple augmentations to a task"""
        augmented_tasks = []
        
        for aug in augmentations:
            train = [self.augment_example(ex, aug) for ex in task_data.get("train", [])]
            test = [self.augment_example(ex, aug) for ex in task_data.get("test", [])]
            
            augmented_tasks.append(AugmentedTask(
                original_id=task_id,
                augmentation=aug,
                train=train,
                test=test,
                difficulty=self.estimate_difficulty(task_data)
            ))
        
        return augmented_tasks
    
    def estimate_difficulty(self, task_data: Dict) -> float:
        """Estimate task difficulty based on various factors"""
        train = task_data.get("train", [])
        
        if not train:
            return 0.5
        
        # Factors
        avg_input_size = sum(len(ex["input"]) * len(ex["input"][0]) for ex in train) / len(train)
        avg_output_size = sum(len(ex["output"]) * len(ex["output"][0]) for ex in train) / len(train)
        num_examples = len(train)
        
        # Unique colors
        all_colors = set()
        for ex in train:
            for row in ex["input"]:
                all_colors.update(row)
            for row in ex["output"]:
                all_colors.update(row)
        num_colors = len(all_colors)
        
        # Size change ratio
        size_ratio = avg_output_size / avg_input_size if avg_input_size > 0 else 1.0
        
        # Difficulty score (0-1)
        difficulty = 0.0
        difficulty += min(avg_input_size / 900, 0.25)  # Max 30x30
        difficulty += min(num_colors / 10, 0.25)
        difficulty += min(abs(size_ratio - 1.0), 0.25)
        difficulty += max(0, (4 - num_examples) / 4) * 0.25  # Fewer examples = harder
        
        return min(difficulty, 1.0)
    
    def create_augmented_dataset(
        self,
        split: str = "training",
        augmentations: Optional[List[str]] = None,
        num_color_permutations: int = 3
    ) -> List[AugmentedTask]:
        """Create augmented dataset"""
        if augmentations is None:
            augmentations = [
                "original",
                "rot90", "rot180", "rot270",
                "flip_h", "flip_v",
                "transpose"
            ]
            # Add color permutations
            for i in range(num_color_permutations):
                augmentations.append(f"color_{i}")
        
        tasks = self.load_all_tasks(split)
        all_augmented = []
        
        for task_id, task_data in tasks:
            augmented = self.augment_task(task_id, task_data, augmentations)
            all_augmented.extend(augmented)
        
        return all_augmented
    
    def create_curriculum(self, tasks: List[AugmentedTask]) -> List[AugmentedTask]:
        """Order tasks by difficulty for curriculum learning"""
        return sorted(tasks, key=lambda t: t.difficulty)
    
    def split_train_val(
        self,
        tasks: List[AugmentedTask],
        val_ratio: float = 0.1
    ) -> Tuple[List[AugmentedTask], List[AugmentedTask]]:
        """Split into training and validation sets"""
        # Group by original task to prevent data leakage
        by_original = defaultdict(list)
        for task in tasks:
            by_original[task.original_id].append(task)
        
        original_ids = list(by_original.keys())
        random.shuffle(original_ids)
        
        val_count = int(len(original_ids) * val_ratio)
        val_ids = set(original_ids[:val_count])
        
        train_tasks = []
        val_tasks = []
        
        for task in tasks:
            if task.original_id in val_ids:
                val_tasks.append(task)
            else:
                train_tasks.append(task)
        
        return train_tasks, val_tasks
    
    def export_to_json(self, tasks: List[AugmentedTask], output_path: str):
        """Export to JSON format"""
        data = []
        for task in tasks:
            data.append({
                "task_id": task.task_id,
                "original_id": task.original_id,
                "augmentation": task.augmentation,
                "difficulty": task.difficulty,
                "train": task.train,
                "test": task.test
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Exported {len(data)} tasks to {output_path}")
    
    def export_to_huggingface(self, tasks: List[AugmentedTask], output_dir: str):
        """Export to HuggingFace datasets format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset files
        train_data = []
        for task in tasks:
            for i, example in enumerate(task.train):
                train_data.append({
                    "task_id": task.task_id,
                    "example_idx": i,
                    "input": json.dumps(example["input"]),
                    "output": json.dumps(example["output"]),
                    "difficulty": task.difficulty
                })
        
        # Save as JSONL
        train_path = os.path.join(output_dir, "train.jsonl")
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
        
        # Create dataset info
        info = {
            "dataset_name": "arc-agi-augmented",
            "num_examples": len(train_data),
            "num_tasks": len(tasks),
            "features": ["task_id", "example_idx", "input", "output", "difficulty"]
        }
        
        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Exported {len(train_data)} examples to {output_dir}")
    
    def generate_statistics(self, tasks: List[AugmentedTask]) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {
            "total_tasks": len(tasks),
            "unique_original_tasks": len(set(t.original_id for t in tasks)),
            "augmentations": defaultdict(int),
            "difficulty_distribution": {
                "easy": 0,
                "medium": 0,
                "hard": 0
            },
            "avg_train_examples": 0,
            "avg_test_examples": 0,
            "grid_size_stats": {
                "min_input": float('inf'),
                "max_input": 0,
                "avg_input": 0,
                "min_output": float('inf'),
                "max_output": 0,
                "avg_output": 0
            }
        }
        
        total_train = 0
        total_test = 0
        total_input_size = 0
        total_output_size = 0
        count = 0
        
        for task in tasks:
            stats["augmentations"][task.augmentation] += 1
            
            if task.difficulty < 0.33:
                stats["difficulty_distribution"]["easy"] += 1
            elif task.difficulty < 0.66:
                stats["difficulty_distribution"]["medium"] += 1
            else:
                stats["difficulty_distribution"]["hard"] += 1
            
            total_train += len(task.train)
            total_test += len(task.test)
            
            for ex in task.train:
                input_size = len(ex["input"]) * len(ex["input"][0])
                output_size = len(ex["output"]) * len(ex["output"][0])
                
                stats["grid_size_stats"]["min_input"] = min(stats["grid_size_stats"]["min_input"], input_size)
                stats["grid_size_stats"]["max_input"] = max(stats["grid_size_stats"]["max_input"], input_size)
                stats["grid_size_stats"]["min_output"] = min(stats["grid_size_stats"]["min_output"], output_size)
                stats["grid_size_stats"]["max_output"] = max(stats["grid_size_stats"]["max_output"], output_size)
                
                total_input_size += input_size
                total_output_size += output_size
                count += 1
        
        if len(tasks) > 0:
            stats["avg_train_examples"] = total_train / len(tasks)
            stats["avg_test_examples"] = total_test / len(tasks)
        
        if count > 0:
            stats["grid_size_stats"]["avg_input"] = total_input_size / count
            stats["grid_size_stats"]["avg_output"] = total_output_size / count
        
        stats["augmentations"] = dict(stats["augmentations"])
        
        return stats

def run_pipeline():
    """Run the complete training data pipeline"""
    print("="*60)
    print("ARC-AGI TRAINING DATA PIPELINE")
    print("="*60)
    
    pipeline = TrainingDataPipeline()
    
    # Check if data exists
    if not os.path.exists(pipeline.training_dir):
        print(f"Dataset not found at {pipeline.training_dir}")
        return None
    
    # Create augmented dataset
    print("\n[1/5] Creating augmented dataset...")
    augmented = pipeline.create_augmented_dataset(
        split="training",
        num_color_permutations=3
    )
    print(f"  Created {len(augmented)} augmented tasks")
    
    # Create curriculum
    print("\n[2/5] Creating curriculum ordering...")
    curriculum = pipeline.create_curriculum(augmented)
    print(f"  Ordered {len(curriculum)} tasks by difficulty")
    
    # Split train/val
    print("\n[3/5] Splitting train/validation...")
    train_tasks, val_tasks = pipeline.split_train_val(curriculum, val_ratio=0.1)
    print(f"  Training: {len(train_tasks)} tasks")
    print(f"  Validation: {len(val_tasks)} tasks")
    
    # Generate statistics
    print("\n[4/5] Generating statistics...")
    stats = pipeline.generate_statistics(augmented)
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Unique original: {stats['unique_original_tasks']}")
    print(f"  Difficulty: Easy={stats['difficulty_distribution']['easy']}, Medium={stats['difficulty_distribution']['medium']}, Hard={stats['difficulty_distribution']['hard']}")
    
    # Export
    print("\n[5/5] Exporting datasets...")
    output_dir = "/home/ubuntu/real-asi/training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline.export_to_json(train_tasks, os.path.join(output_dir, "train_augmented.json"))
    pipeline.export_to_json(val_tasks, os.path.join(output_dir, "val_augmented.json"))
    pipeline.export_to_huggingface(train_tasks, os.path.join(output_dir, "huggingface"))
    
    # Save statistics
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved to {stats_path}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Training tasks: {len(train_tasks)}")
    print(f"Validation tasks: {len(val_tasks)}")
    print(f"Augmentation factor: {len(augmented) / stats['unique_original_tasks']:.1f}x")
    
    return {
        "train_count": len(train_tasks),
        "val_count": len(val_tasks),
        "total_augmented": len(augmented),
        "statistics": stats,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    results = run_pipeline()
    
    if results:
        # Save results summary
        summary_path = "/home/ubuntu/real-asi/training_pipeline_results.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {summary_path}")
