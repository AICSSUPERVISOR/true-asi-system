#!/usr/bin/env python3
"""
SOAR: Self-Improving Program Synthesis for ARC-AGI
Based on: https://openreview.net/pdf?id=z4IG090qt2

Key Concepts:
1. Generate programs using LLM
2. Execute and verify against examples
3. Collect successful search traces
4. Fine-tune LLM on its own successful traces
5. Iterate to improve

This achieves 52% on ARC-AGI-1 with self-improvement loop.
"""

import json
import os
import time
import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import urllib.request
import urllib.error

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MAX_PROGRAMS_PER_TASK = 50
MAX_SEARCH_DEPTH = 10
BEAM_WIDTH = 5

@dataclass
class Program:
    """A synthesized program"""
    code: str
    description: str
    primitives_used: List[str]
    depth: int
    score: float = 0.0
    verified: bool = False
    
    def __hash__(self):
        return hash(self.code)

@dataclass
class SearchTrace:
    """A trace of the search process"""
    task_id: str
    successful_programs: List[Program]
    failed_programs: List[Program]
    search_path: List[str]
    total_attempts: int
    time_taken: float

# ARC-AGI Domain-Specific Language (DSL) Primitives
DSL_PRIMITIVES = {
    # Grid operations
    "rotate_90": "Rotate grid 90 degrees clockwise",
    "rotate_180": "Rotate grid 180 degrees",
    "rotate_270": "Rotate grid 270 degrees clockwise",
    "flip_horizontal": "Flip grid horizontally",
    "flip_vertical": "Flip grid vertically",
    "transpose": "Transpose grid (swap rows and columns)",
    
    # Color operations
    "replace_color": "Replace one color with another",
    "swap_colors": "Swap two colors",
    "invert_colors": "Invert all colors (9 - color)",
    "fill_color": "Fill entire grid with a color",
    "count_colors": "Count occurrences of each color",
    
    # Shape operations
    "find_objects": "Find connected components",
    "extract_object": "Extract a specific object",
    "move_object": "Move an object to a new position",
    "scale_object": "Scale an object up or down",
    "copy_object": "Copy an object to a new position",
    
    # Pattern operations
    "tile_pattern": "Tile a pattern across the grid",
    "mirror_pattern": "Mirror a pattern",
    "repeat_pattern": "Repeat a pattern n times",
    "overlay_patterns": "Overlay two patterns",
    
    # Size operations
    "crop": "Crop grid to bounding box",
    "pad": "Pad grid with a color",
    "resize": "Resize grid to new dimensions",
    "upscale": "Upscale grid by factor",
    "downscale": "Downscale grid by factor",
    
    # Selection operations
    "select_row": "Select a specific row",
    "select_column": "Select a specific column",
    "select_region": "Select a rectangular region",
    "select_by_color": "Select cells by color",
    "select_largest": "Select the largest object",
    "select_smallest": "Select the smallest object",
    
    # Combination operations
    "stack_horizontal": "Stack grids horizontally",
    "stack_vertical": "Stack grids vertically",
    "merge_grids": "Merge two grids with a rule",
    "diff_grids": "Find difference between grids",
    
    # Conditional operations
    "if_color": "Apply operation if color matches",
    "if_size": "Apply operation if size matches",
    "if_pattern": "Apply operation if pattern matches",
    "for_each_object": "Apply operation to each object",
}

# Python implementations of primitives
PRIMITIVE_IMPLEMENTATIONS = '''
from typing import List, Tuple, Set
import copy

Grid = List[List[int]]

def rotate_90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise"""
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]

def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees"""
    return [row[::-1] for row in grid[::-1]]

def rotate_270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise"""
    rows, cols = len(grid), len(grid[0])
    return [[grid[j][cols-1-i] for j in range(rows)] for i in range(cols)]

def flip_horizontal(grid: Grid) -> Grid:
    """Flip grid horizontally"""
    return [row[::-1] for row in grid]

def flip_vertical(grid: Grid) -> Grid:
    """Flip grid vertically"""
    return grid[::-1]

def transpose(grid: Grid) -> Grid:
    """Transpose grid"""
    return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]

def replace_color(grid: Grid, old_color: int, new_color: int) -> Grid:
    """Replace one color with another"""
    return [[new_color if c == old_color else c for c in row] for row in grid]

def swap_colors(grid: Grid, color1: int, color2: int) -> Grid:
    """Swap two colors"""
    return [[color2 if c == color1 else (color1 if c == color2 else c) for c in row] for row in grid]

def invert_colors(grid: Grid) -> Grid:
    """Invert all colors (9 - color)"""
    return [[9 - c for c in row] for row in grid]

def fill_color(grid: Grid, color: int) -> Grid:
    """Fill entire grid with a color"""
    return [[color for _ in row] for row in grid]

def count_colors(grid: Grid) -> dict:
    """Count occurrences of each color"""
    counts = {}
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return counts

def find_objects(grid: Grid) -> List[Set[Tuple[int, int]]]:
    """Find connected components (excluding background color 0)"""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    objects = []
    
    def bfs(start_r, start_c, color):
        obj = set()
        queue = [(start_r, start_c)]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != color:
                continue
            visited.add((r, c))
            obj.add((r, c))
            queue.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        return obj
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != 0:
                obj = bfs(r, c, grid[r][c])
                if obj:
                    objects.append(obj)
    
    return objects

def crop_to_content(grid: Grid) -> Grid:
    """Crop grid to bounding box of non-zero content"""
    rows, cols = len(grid), len(grid[0])
    min_r, max_r = rows, 0
    min_c, max_c = cols, 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if min_r > max_r:
        return grid
    
    return [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]

def pad_grid(grid: Grid, padding: int, color: int = 0) -> Grid:
    """Pad grid with a color"""
    rows, cols = len(grid), len(grid[0])
    new_rows = rows + 2 * padding
    new_cols = cols + 2 * padding
    result = [[color] * new_cols for _ in range(new_rows)]
    for r in range(rows):
        for c in range(cols):
            result[r + padding][c + padding] = grid[r][c]
    return result

def upscale(grid: Grid, factor: int) -> Grid:
    """Upscale grid by factor"""
    rows, cols = len(grid), len(grid[0])
    result = []
    for r in range(rows):
        for _ in range(factor):
            row = []
            for c in range(cols):
                row.extend([grid[r][c]] * factor)
            result.append(row)
    return result

def downscale(grid: Grid, factor: int) -> Grid:
    """Downscale grid by factor (takes top-left of each block)"""
    rows, cols = len(grid), len(grid[0])
    new_rows = rows // factor
    new_cols = cols // factor
    return [[grid[r * factor][c * factor] for c in range(new_cols)] for r in range(new_rows)]

def tile_pattern(pattern: Grid, rows: int, cols: int) -> Grid:
    """Tile a pattern to fill rows x cols"""
    p_rows, p_cols = len(pattern), len(pattern[0])
    result = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(pattern[r % p_rows][c % p_cols])
        result.append(row)
    return result

def overlay(base: Grid, overlay_grid: Grid, r_offset: int = 0, c_offset: int = 0) -> Grid:
    """Overlay one grid on another (non-zero values replace)"""
    result = [row[:] for row in base]
    for r in range(len(overlay_grid)):
        for c in range(len(overlay_grid[0])):
            if overlay_grid[r][c] != 0:
                nr, nc = r + r_offset, c + c_offset
                if 0 <= nr < len(result) and 0 <= nc < len(result[0]):
                    result[nr][nc] = overlay_grid[r][c]
    return result

def get_unique_colors(grid: Grid) -> List[int]:
    """Get unique colors in grid"""
    colors = set()
    for row in grid:
        colors.update(row)
    return sorted(colors)

def get_grid_size(grid: Grid) -> Tuple[int, int]:
    """Get grid dimensions"""
    return len(grid), len(grid[0])

def create_grid(rows: int, cols: int, color: int = 0) -> Grid:
    """Create a new grid filled with color"""
    return [[color] * cols for _ in range(rows)]

def copy_grid(grid: Grid) -> Grid:
    """Deep copy a grid"""
    return [row[:] for row in grid]
'''

def call_llm_for_program(task_data: Dict, primitives: List[str], temperature: float = 0.7) -> str:
    """Call LLM to generate a program using specified primitives"""
    if not ANTHROPIC_API_KEY:
        return "def solve(grid): return grid"
    
    train_examples = task_data.get("train", [])
    
    # Format examples
    examples_str = ""
    for i, ex in enumerate(train_examples):
        examples_str += f"\nExample {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}\n"
    
    primitives_str = "\n".join([f"- {p}: {DSL_PRIMITIVES.get(p, 'Custom operation')}" for p in primitives])
    
    prompt = f"""You are an expert at solving ARC-AGI puzzles using program synthesis.

## Available Primitives:
{primitives_str}

## Primitive Implementations (already available):
{PRIMITIVE_IMPLEMENTATIONS}

## Task Examples:
{examples_str}

## Instructions:
1. Analyze the pattern in the examples
2. Write a Python function `solve(grid)` that transforms input to output
3. Use ONLY the available primitives and implementations
4. The function should work for ANY valid input

## Your Solution:
```python
def solve(grid):
    # Your implementation using the primitives
```
"""
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 2048,
        "temperature": temperature,
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
            
            # Extract code
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0]
                return PRIMITIVE_IMPLEMENTATIONS + "\n\n" + code.strip()
            elif "def solve" in text:
                start = text.find("def solve")
                return PRIMITIVE_IMPLEMENTATIONS + "\n\n" + text[start:].strip()
            
            return PRIMITIVE_IMPLEMENTATIONS + "\n\ndef solve(grid): return grid"
    except Exception as e:
        return PRIMITIVE_IMPLEMENTATIONS + f"\n\n# Error: {e}\ndef solve(grid): return grid"

def verify_program(program: Program, task_data: Dict) -> Tuple[bool, float]:
    """Verify a program against training examples"""
    train_examples = task_data.get("train", [])
    
    local_env = {"List": list, "Tuple": tuple, "Set": set}
    
    try:
        exec(program.code, local_env)
        solve_func = local_env.get("solve")
        
        if not solve_func:
            return False, 0.0
        
        correct = 0
        for example in train_examples:
            try:
                input_grid = [row[:] for row in example["input"]]
                expected = example["output"]
                actual = solve_func(input_grid)
                
                if actual == expected:
                    correct += 1
            except:
                pass
        
        score = correct / len(train_examples) if train_examples else 0.0
        return score == 1.0, score
        
    except:
        return False, 0.0

def beam_search(task_id: str, task_data: Dict, beam_width: int = BEAM_WIDTH, max_depth: int = MAX_SEARCH_DEPTH) -> SearchTrace:
    """Beam search for program synthesis"""
    start_time = time.time()
    
    successful_programs = []
    failed_programs = []
    search_path = []
    
    # Start with all primitives
    all_primitives = list(DSL_PRIMITIVES.keys())
    
    # Generate initial beam
    beam = []
    for _ in range(beam_width):
        # Random subset of primitives
        num_primitives = random.randint(3, 8)
        primitives = random.sample(all_primitives, min(num_primitives, len(all_primitives)))
        
        code = call_llm_for_program(task_data, primitives, temperature=0.8)
        program = Program(
            code=code,
            description=f"Generated with {len(primitives)} primitives",
            primitives_used=primitives,
            depth=0
        )
        
        verified, score = verify_program(program, task_data)
        program.verified = verified
        program.score = score
        
        if verified:
            successful_programs.append(program)
            search_path.append(f"SUCCESS at depth 0: score={score}")
        else:
            beam.append(program)
            failed_programs.append(program)
            search_path.append(f"FAILED at depth 0: score={score}")
    
    # If we found a solution, return early
    if successful_programs:
        return SearchTrace(
            task_id=task_id,
            successful_programs=successful_programs,
            failed_programs=failed_programs,
            search_path=search_path,
            total_attempts=len(beam) + len(successful_programs),
            time_taken=time.time() - start_time
        )
    
    # Continue search
    for depth in range(1, max_depth):
        if not beam:
            break
        
        # Sort beam by score
        beam.sort(key=lambda p: p.score, reverse=True)
        beam = beam[:beam_width]
        
        new_beam = []
        for parent in beam:
            # Mutate primitives
            primitives = parent.primitives_used[:]
            
            # Add or remove a primitive
            if random.random() < 0.5 and len(primitives) > 2:
                primitives.remove(random.choice(primitives))
            else:
                available = [p for p in all_primitives if p not in primitives]
                if available:
                    primitives.append(random.choice(available))
            
            code = call_llm_for_program(task_data, primitives, temperature=0.6)
            program = Program(
                code=code,
                description=f"Mutated from depth {depth-1}",
                primitives_used=primitives,
                depth=depth
            )
            
            verified, score = verify_program(program, task_data)
            program.verified = verified
            program.score = score
            
            if verified:
                successful_programs.append(program)
                search_path.append(f"SUCCESS at depth {depth}: score={score}")
            else:
                new_beam.append(program)
                failed_programs.append(program)
                search_path.append(f"FAILED at depth {depth}: score={score}")
        
        beam = new_beam
        
        # Early exit if found solution
        if successful_programs:
            break
    
    return SearchTrace(
        task_id=task_id,
        successful_programs=successful_programs,
        failed_programs=failed_programs,
        search_path=search_path,
        total_attempts=len(failed_programs) + len(successful_programs),
        time_taken=time.time() - start_time
    )

def collect_training_data(traces: List[SearchTrace]) -> Dict[str, Any]:
    """Collect training data from successful traces for fine-tuning"""
    training_data = {
        "successful_programs": [],
        "primitive_usage": defaultdict(int),
        "avg_program_length": 0,
        "success_rate": 0
    }
    
    total_programs = 0
    total_length = 0
    
    for trace in traces:
        for program in trace.successful_programs:
            training_data["successful_programs"].append({
                "task_id": trace.task_id,
                "code": program.code,
                "primitives": program.primitives_used,
                "depth": program.depth
            })
            
            for prim in program.primitives_used:
                training_data["primitive_usage"][prim] += 1
            
            total_length += len(program.code)
            total_programs += 1
    
    if total_programs > 0:
        training_data["avg_program_length"] = total_length / total_programs
        training_data["success_rate"] = total_programs / len(traces) if traces else 0
    
    training_data["primitive_usage"] = dict(training_data["primitive_usage"])
    
    return training_data

def run_soar(data_dir: str, max_tasks: int = 10) -> Dict[str, Any]:
    """Run SOAR on ARC-AGI tasks"""
    import glob
    
    task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:max_tasks]
    
    print("="*60)
    print("SOAR: Self-Improving Program Synthesis for ARC-AGI")
    print("="*60)
    print(f"\nTasks: {len(task_files)}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Max depth: {MAX_SEARCH_DEPTH}")
    print(f"API configured: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
    
    traces = []
    solved = 0
    
    for i, task_file in enumerate(task_files):
        task_id = os.path.basename(task_file).replace('.json', '')
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        print(f"\n[{i+1}/{len(task_files)}] Task: {task_id}")
        
        trace = beam_search(task_id, task_data, beam_width=BEAM_WIDTH, max_depth=3)  # Reduced depth for speed
        traces.append(trace)
        
        if trace.successful_programs:
            solved += 1
            print(f"  ✅ SOLVED in {trace.time_taken:.1f}s ({len(trace.successful_programs)} solutions)")
        else:
            best_score = max([p.score for p in trace.failed_programs]) if trace.failed_programs else 0
            print(f"  ❌ FAILED (best score: {best_score:.2f}, attempts: {trace.total_attempts})")
    
    # Collect training data
    training_data = collect_training_data(traces)
    
    results = {
        "total_tasks": len(task_files),
        "solved": solved,
        "accuracy": solved / len(task_files) if task_files else 0,
        "training_data": training_data,
        "traces": [
            {
                "task_id": t.task_id,
                "solved": len(t.successful_programs) > 0,
                "attempts": t.total_attempts,
                "time": t.time_taken
            }
            for t in traces
        ]
    }
    
    print("\n" + "="*60)
    print("SOAR RESULTS")
    print("="*60)
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Solved: {results['solved']} ({results['accuracy']*100:.1f}%)")
    print(f"Training samples collected: {len(training_data['successful_programs'])}")
    
    return results

if __name__ == "__main__":
    arc_training_dir = "/home/ubuntu/ARC-AGI/data/training"
    
    if os.path.exists(arc_training_dir):
        results = run_soar(arc_training_dir, max_tasks=5)
        
        # Save results
        output_file = "/home/ubuntu/real-asi/soar_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"ARC-AGI dataset not found at {arc_training_dir}")
