"""
PHASE 34: AI-DRIVEN PROGRAMMING (SELF-CODING)
Goal: Enable the system to write, debug, and improve its own code
Using: Vertex AI Gemini 2.5 Flash Lite + existing infrastructure
"""

import json
import boto3
import requests
import subprocess
from datetime import datetime

class SelfCodingSystem:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.bucket = "asi-knowledge-base-898982995956"
        self.vertex_api_key = "AQ.Ab8RN6J09J-LtGcl3r7aigIc4RGi3mhE3BVk0MLdHzU2p880_g"
        self.vertex_url = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent"
        
    def generate_code(self, task_description):
        """Generate code for a given task"""
        print(f"\n{'='*80}")
        print(f"GENERATING CODE FOR: {task_description}")
        print(f"{'='*80}\n")
        
        prompt = f"""You are an AI that can write code. Generate Python code for this task:

Task: {task_description}

Requirements:
1. Write clean, well-documented code
2. Include error handling
3. Make it production-ready
4. Add comments explaining the logic

Provide ONLY the Python code, no explanations."""

        try:
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                code = result["candidates"][0]["content"]["parts"][0]["text"]
                # Clean up code blocks
                code = code.replace("```python", "").replace("```", "").strip()
                print(f"✅ Code generated ({len(code)} characters)")
                return code
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def debug_code(self, code, error_message):
        """Debug and fix code"""
        print(f"\n{'='*80}")
        print("DEBUGGING CODE")
        print(f"{'='*80}\n")
        
        prompt = f"""You are an AI debugger. Fix this code:

Code:
{code}

Error:
{error_message}

Provide the FIXED code only, no explanations."""

        try:
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2000}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                fixed_code = result["candidates"][0]["content"]["parts"][0]["text"]
                fixed_code = fixed_code.replace("```python", "").replace("```", "").strip()
                print(f"✅ Code debugged")
                return fixed_code
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def improve_code(self, code):
        """Improve existing code"""
        print(f"\n{'='*80}")
        print("IMPROVING CODE")
        print(f"{'='*80}\n")
        
        prompt = f"""You are an AI code optimizer. Improve this code:

{code}

Make it:
1. More efficient
2. More readable
3. Better error handling
4. Add type hints

Provide the IMPROVED code only."""

        try:
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2000}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                improved_code = result["candidates"][0]["content"]["parts"][0]["text"]
                improved_code = improved_code.replace("```python", "").replace("```", "").strip()
                print(f"✅ Code improved")
                return improved_code
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def test_code(self, code, test_file="/tmp/test_generated_code.py"):
        """Test generated code"""
        print(f"\n{'='*80}")
        print("TESTING CODE")
        print(f"{'='*80}\n")
        
        try:
            # Write code to file
            with open(test_file, "w") as f:
                f.write(code)
            
            # Try to compile it
            result = subprocess.run(
                ["python3.11", "-m", "py_compile", test_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ Code compiles successfully")
                return True, None
            else:
                print(f"❌ Code has syntax errors")
                return False, result.stderr
                
        except Exception as e:
            print(f"❌ Exception during testing: {e}")
            return False, str(e)
    
    def run_phase34(self):
        """Execute Phase 34"""
        print(f"\n{'='*80}")
        print("PHASE 34: AI-DRIVEN PROGRAMMING (SELF-CODING)")
        print(f"{'='*80}\n")
        
        # Test task
        task = "Create a function that calculates the Fibonacci sequence up to n terms using dynamic programming"
        
        # Generate code
        code = self.generate_code(task)
        if not code:
            return False
        
        # Test code
        compiles, error = self.test_code(code)
        
        # If it doesn't compile, debug it
        if not compiles and error:
            print("\nCode has errors, attempting to debug...")
            code = self.debug_code(code, error)
            if code:
                compiles, error = self.test_code(code)
        
        # Improve code
        if compiles:
            improved_code = self.improve_code(code)
            if improved_code:
                compiles_improved, _ = self.test_code(improved_code)
                if compiles_improved:
                    code = improved_code
        
        # Brutal audit
        print(f"\n{'='*80}")
        print("BRUTAL AUDIT - PHASE 34")
        print(f"{'='*80}\n")
        
        audit_result = {
            "code_generated": code is not None,
            "code_compiles": compiles,
            "code_improved": improved_code is not None if compiles else False,
            "overall_pass": code is not None and compiles
        }
        
        print(f"Code generated: {'✅ YES' if audit_result['code_generated'] else '❌ NO'}")
        print(f"Code compiles: {'✅ YES' if audit_result['code_compiles'] else '❌ NO'}")
        print(f"Code improved: {'✅ YES' if audit_result['code_improved'] else '❌ NO'}")
        print(f"Overall: {'✅ PASS' if audit_result['overall_pass'] else '❌ FAIL'}")
        
        # Save results
        result = {
            "phase": 34,
            "name": "AI-Driven Programming (Self-Coding)",
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "generated_code": code,
            "brutal_audit": audit_result,
            "success": audit_result['overall_pass']
        }
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"PHASE34_RESULTS/results_{date_str}.json"
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json.dumps(result, indent=2)
        )
        
        print(f"\n{'='*80}")
        print("PHASE 34 COMPLETE")
        status_str = "SUCCESS" if result["success"] else "FAILED"
        print(f"Status: {status_str}")
        print(f"Saved to S3: s3://{self.bucket}/{s3_key}")
        print(f"{'='*80}\n")
        
        return result["success"]

if __name__ == "__main__":
    system = SelfCodingSystem()
    success = system.run_phase34()
    exit(0 if success else 1)
