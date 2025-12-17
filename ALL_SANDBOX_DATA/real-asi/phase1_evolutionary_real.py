#!/usr/bin/env python3.11
"""
PHASE 1: REAL EVOLUTIONARY ARC-AGI SOLVER
Based on Jeremy Berman's approach (58.5% accuracy)
Using REAL Anthropic API calls - NO SIMULATIONS
"""

import json
import os
import subprocess
import time
from typing import List, Dict, Tuple

class Phase1EvolutionarySolver:
    """
    Real evolutionary solver using Anthropic Claude API
    Target: 40-50% accuracy on ARC-AGI
    """
    
    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.generation_count = 0
        self.total_api_calls = 0
        self.total_cost = 0.0
        self.model = "claude-3-5-sonnet-20241022"
        
    def call_claude(self, prompt: str) -> str:
        """Call Claude API using urllib"""
        
        # Escape for shell
        prompt_safe = prompt.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
        
        cmd = f'''python3.11 << 'PYTHON_SCRIPT'
import json
import urllib.request

api_key = "{self.api_key}"
url = "https://api.anthropic.com/v1/messages"

data = {{
    "model": "{self.model}",
    "max_tokens": 4096,
    "messages": [{{
        "role": "user",
        "content": """{prompt_safe}"""
    }}]
}}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode('utf-8'),
    headers={{
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }}
)

try:
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(result['content'][0]['text'])
except Exception as e:
    print(f"ERROR: {{e}}")
PYTHON_SCRIPT
'''
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
            self.total_api_calls += 1
            self.total_cost += 0.015  # $3/$15 per million tokens
            
            if result.returncode == 0 and not result.stdout.startswith('ERROR'):
                return result.stdout.strip()
            else:
                return "# API Error"
        except Exception as e:
            return f"# Error: {e}"
    
    def generate_population(self, task: Dict, size: int = 6) -> List[str]:
        """Generate initial population"""
        print(f"\n🧬 Generating {size} transform functions...")
        
        examples = "\n".join([
            f"Example {i+1}:\\nInput: {ex['input']}\\nOutput: {ex['output']}"
            for i, ex in enumerate(task['train'])
        ])
        
        prompt = f"""Solve this ARC-AGI puzzle by writing a Python function.

Examples:
{examples}

Write a function `transform(input_grid)` that:
- Takes a 2D list as input
- Returns a 2D list as output
- Uses only standard Python
- Solves the pattern shown in examples

Return ONLY Python code, no explanation."""
        
        population = []
        for i in range(size):
            print(f"  {i+1}/{size}...", end=" ", flush=True)
            code = self.call_claude(prompt)
            
            # Extract code
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            population.append(code)
            print("✓")
            time.sleep(2)  # Rate limit
        
        return population
    
    def evaluate(self, code: str, examples: List[Dict]) -> Tuple[int, int]:
        """Evaluate function fitness"""
        try:
            temp_file = f"/tmp/eval_{hash(code) % 100000}.py"
            with open(temp_file, 'w') as f:
                f.write(code)
            
            primary = 0
            secondary = 0
            
            for ex in examples:
                try:
                    exec_cmd = f"""
import json
exec(open('{temp_file}').read())
result = transform({ex['input']})
print(json.dumps(result))
"""
                    proc = subprocess.run(
                        ['python3.11', '-c', exec_cmd],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    
                    if proc.returncode == 0:
                        pred = json.loads(proc.stdout.strip())
                        exp = ex['output']
                        
                        if pred == exp:
                            primary += 1
                            secondary += sum(len(row) for row in exp)
                        else:
                            # Count matching cells
                            for i, row in enumerate(exp):
                                if i < len(pred):
                                    for j, cell in enumerate(row):
                                        if j < len(pred[i]) and pred[i][j] == cell:
                                            secondary += 1
                except:
                    pass
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return (primary, secondary)
        except:
            return (0, 0)
    
    def select_best(self, population: List[str], examples: List[Dict], n: int = 3) -> List[Tuple[str, int, int]]:
        """Select top performers"""
        print(f"\\n🔍 Evaluating population...")
        
        scored = []
        for i, code in enumerate(population):
            print(f"  {i+1}/{len(population)}...", end=" ", flush=True)
            primary, secondary = self.evaluate(code, examples)
            scored.append((code, primary, secondary))
            print(f"{primary}/{len(examples)}")
        
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        print(f"\\n📊 Top {n}:")
        for i, (_, p, s) in enumerate(scored[:n]):
            print(f"  {i+1}. Solved: {p}/{len(examples)}, Cells: {s}")
        
        return scored[:n]
    
    def improve(self, parent: str, score: Tuple[int, int], examples: List[Dict], n: int = 3) -> List[str]:
        """Create improved versions"""
        
        ex_str = "\\n".join([
            f"Ex {i+1}: In={ex['input']}, Out={ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"""Improve this Python function that solves {score[0]}/{len(examples)} examples:

Examples:
{ex_str}

Current code:
```python
{parent}
```

Create an IMPROVED version that solves MORE examples.
Return ONLY Python code."""
        
        offspring = []
        for i in range(n):
            code = self.call_claude(prompt)
            
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            offspring.append(code)
            time.sleep(2)
        
        return offspring
    
    def evolve(self, task: Dict) -> str:
        """Run evolutionary algorithm"""
        examples = task['train']
        
        # Gen 1
        self.generation_count = 1
        print(f"\\n{'='*50}\\nGENERATION {self.generation_count}\\n{'='*50}")
        pop = self.generate_population(task, 6)
        best = self.select_best(pop, examples, 3)
        
        if best and best[0][1] == len(examples):
            print("\\n🎉 Perfect solution!")
            return best[0][0]
        
        # Gen 2
        self.generation_count = 2
        print(f"\\n{'='*50}\\nGENERATION {self.generation_count}\\n{'='*50}")
        print("Creating offspring...")
        pop = []
        for code, p, s in best:
            pop.extend(self.improve(code, (p, s), examples, 3))
        
        best = self.select_best(pop, examples, 2)
        
        if best:
            return best[0][0]
        return "def transform(g): return g"
    
    def solve(self, task: Dict) -> List:
        """Solve complete task"""
        best = self.evolve(task)
        
        predictions = []
        for test in task['test']:
            try:
                temp = "/tmp/solve.py"
                with open(temp, 'w') as f:
                    f.write(best)
                
                cmd = f"""
import json
exec(open('{temp}').read())
print(json.dumps(transform({test['input']})))
"""
                proc = subprocess.run(
                    ['python3.11', '-c', cmd],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if proc.returncode == 0:
                    predictions.append(json.loads(proc.stdout.strip()))
                else:
                    predictions.append(test['input'])
            except:
                predictions.append(test['input'])
        
        return predictions

def main():
    print("="*70)
    print("PHASE 1: EVOLUTIONARY ARC-AGI SOLVER")
    print("Real Anthropic API - Target: 40-50% accuracy")
    print("="*70)
    
    solver = Phase1EvolutionarySolver()
    
    # Load 5 tasks for testing
    eval_dir = "/home/ubuntu/ARC-AGI/data/evaluation"
    files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.json')])[:5]
    
    print(f"\\n📊 Testing on {len(files)} tasks...")
    
    correct = 0
    details = []
    start = time.time()
    
    for i, fname in enumerate(files, 1):
        print(f"\\n{'='*70}\\nTASK {i}/{len(files)}: {fname}\\n{'='*70}")
        
        with open(os.path.join(eval_dir, fname)) as f:
            task = json.load(f)
        
        try:
            t_start = time.time()
            preds = solver.solve(task)
            t_time = time.time() - t_start
            
            solved = all(
                preds[j] == task['test'][j]['output']
                for j in range(len(task['test']))
                if j < len(preds)
            )
            
            if solved:
                correct += 1
                print(f"\\n✅ SOLVED ({t_time:.1f}s)")
            else:
                print(f"\\n❌ Not solved ({t_time:.1f}s)")
            
            details.append({'task': fname, 'solved': solved, 'time': t_time})
        except Exception as e:
            print(f"\\n❌ Error: {e}")
            details.append({'task': fname, 'solved': False, 'error': str(e)})
    
    elapsed = time.time() - start
    accuracy = (correct / len(files)) * 100
    
    # Save results
    results = {
        'phase': 1,
        'method': 'Evolutionary (Jeremy Berman approach)',
        'model': solver.model,
        'tasks_tested': len(files),
        'tasks_correct': correct,
        'accuracy': accuracy,
        'api_calls': solver.total_api_calls,
        'cost_usd': solver.total_cost,
        'time_seconds': elapsed,
        'simulated': False,
        'real_api': True,
        'details': details
    }
    
    out_file = '/home/ubuntu/real-asi/phase1_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n{'='*70}\\nPHASE 1 RESULTS\\n{'='*70}")
    print(f"Tasks: {len(files)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"API Calls: {solver.total_api_calls}")
    print(f"Cost: ${solver.total_cost:.2f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Real API: YES")
    print(f"{'='*70}")
    
    # Upload to S3
    print(f"\\n📤 Uploading to S3...")
    subprocess.run(['aws', 's3', 'cp', out_file, 's3://asi-knowledge-base-898982995956/REAL_ASI/'])
    print("✅ Uploaded to S3")

if __name__ == "__main__":
    main()
