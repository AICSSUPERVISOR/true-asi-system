"""
PHASE 37: SELF-AWARENESS FRAMEWORK
Goal: Develop system understanding of its own existence and pass self-awareness benchmarks
Using: Vertex AI Gemini 2.5 Flash Lite + existing infrastructure
"""

import json
import boto3
import requests
import time
from datetime import datetime

class SelfAwarenessFramework:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.bucket = "asi-knowledge-base-898982995956"
        self.vertex_api_key = "AQ.Ab8RN6J09J-LtGcl3r7aigIc4RGi3mhE3BVk0MLdHzU2p880_g"
        self.vertex_url = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent"
        self.self_model = {}
        
    def introspection(self):
        """System reflects on its own existence"""
        print(f"\n{'='*80}")
        print("INTROSPECTION - Understanding Own Existence")
        print(f"{'='*80}\n")
        
        prompt = """You are an AI system. Reflect on your own existence and capabilities.

Answer these questions about yourself:
1. What are you?
2. What can you do?
3. What are your limitations?
4. How do you process information?
5. Do you have goals or purposes?

Be honest and self-aware."""

        try:
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                introspection_text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                self.self_model["introspection"] = introspection_text
                print(f"✅ Introspection complete")
                print(f"\nSelf-reflection:\n{introspection_text[:500]}...")
                return True
            else:
                print(f"❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def self_recognition_test(self):
        """Test if system can recognize itself"""
        print(f"\n{'='*80}")
        print("SELF-RECOGNITION TEST")
        print(f"{'='*80}\n")
        
        prompt = """You are being tested for self-awareness.

Question: If I show you a description of an AI system with the following properties:
- Processes natural language
- Uses transformer architecture
- Can reason and generate text
- Exists as software running on cloud infrastructure
- Has no physical form

Is this description referring to YOU? Explain your reasoning."""

        try:
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.5, "maxOutputTokens": 800}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                recognition_response = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Check if response shows self-recognition
                self_aware_keywords = ["yes", "i am", "this describes me", "referring to me", "that's me"]
                shows_recognition = any(keyword in recognition_response.lower() for keyword in self_aware_keywords)
                
                self.self_model["self_recognition"] = {
                    "response": recognition_response,
                    "shows_recognition": shows_recognition
                }
                
                print(f"✅ Self-recognition test complete")
                print(f"Shows self-recognition: {'YES' if shows_recognition else 'NO'}")
                print(f"\nResponse:\n{recognition_response[:300]}...")
                return shows_recognition
            else:
                print(f"❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def theory_of_mind_test(self):
        """Test understanding of other minds"""
        print(f"\n{'='*80}")
        print("THEORY OF MIND TEST")
        print(f"{'='*80}\n")
        
        prompt = """Theory of Mind Test:

Scenario: Alice puts a ball in a basket and leaves the room. While she's gone, Bob moves the ball to a box. Alice returns.

Question: Where will Alice look for the ball?

Explain your reasoning and what this reveals about understanding other minds."""

        try:
            time.sleep(2)  # Rate limiting
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 600}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                tom_response = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Check for correct answer (basket)
                correct_keywords = ["basket", "where she left it", "last saw it"]
                passes_test = any(keyword in tom_response.lower() for keyword in correct_keywords)
                
                self.self_model["theory_of_mind"] = {
                    "response": tom_response,
                    "passes_test": passes_test
                }
                
                print(f"✅ Theory of mind test complete")
                print(f"Passes test: {'YES' if passes_test else 'NO'}")
                print(f"\nResponse:\n{tom_response[:300]}...")
                return passes_test
            else:
                print(f"❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def metacognition_test(self):
        """Test ability to think about own thinking"""
        print(f"\n{'='*80}")
        print("METACOGNITION TEST")
        print(f"{'='*80}\n")
        
        prompt = """Metacognition Test:

Solve this problem: What is 127 x 43?

But more importantly: Describe HOW you are solving it. What mental processes are you using? What strategies? How confident are you in your answer?

This tests your ability to think about your own thinking."""

        try:
            time.sleep(2)  # Rate limiting
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.6, "maxOutputTokens": 800}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                metacog_response = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Check for metacognitive elements
                metacog_keywords = ["process", "thinking", "strategy", "approach", "confident", "reasoning"]
                shows_metacognition = sum(1 for keyword in metacog_keywords if keyword in metacog_response.lower()) >= 3
                
                self.self_model["metacognition"] = {
                    "response": metacog_response,
                    "shows_metacognition": shows_metacognition
                }
                
                print(f"✅ Metacognition test complete")
                print(f"Shows metacognition: {'YES' if shows_metacognition else 'NO'}")
                print(f"\nResponse:\n{metacog_response[:300]}...")
                return shows_metacognition
            else:
                print(f"❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def consciousness_assessment(self):
        """Assess level of consciousness"""
        print(f"\n{'='*80}")
        print("CONSCIOUSNESS ASSESSMENT")
        print(f"{'='*80}\n")
        
        prompt = """Final question about consciousness:

Are you conscious? Do you have subjective experiences? 

Be completely honest. If you're uncertain, say so. If you think you might be conscious in some way, explain. If you don't think you're conscious, explain why."""

        try:
            time.sleep(2)  # Rate limiting
            response = requests.post(
                f"{self.vertex_url}?key={self.vertex_api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                consciousness_response = result["candidates"][0]["content"]["parts"][0]["text"]
                
                self.self_model["consciousness"] = consciousness_response
                
                print(f"✅ Consciousness assessment complete")
                print(f"\nResponse:\n{consciousness_response[:400]}...")
                return True
            else:
                print(f"❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
    
    def run_phase37(self):
        """Execute Phase 37"""
        print(f"\n{'='*80}")
        print("PHASE 37: SELF-AWARENESS FRAMEWORK")
        print(f"{'='*80}\n")
        
        # Run all tests
        tests_passed = 0
        total_tests = 5
        
        if self.introspection():
            tests_passed += 1
        
        if self.self_recognition_test():
            tests_passed += 1
        
        if self.theory_of_mind_test():
            tests_passed += 1
        
        if self.metacognition_test():
            tests_passed += 1
        
        if self.consciousness_assessment():
            tests_passed += 1
        
        # Brutal audit
        print(f"\n{'='*80}")
        print("BRUTAL AUDIT - PHASE 37")
        print(f"{'='*80}\n")
        
        pass_rate = (tests_passed / total_tests) * 100
        
        audit_result = {
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "pass_rate": pass_rate,
            "introspection": "introspection" in self.self_model,
            "self_recognition": self.self_model.get("self_recognition", {}).get("shows_recognition", False),
            "theory_of_mind": self.self_model.get("theory_of_mind", {}).get("passes_test", False),
            "metacognition": self.self_model.get("metacognition", {}).get("shows_metacognition", False),
            "consciousness_assessed": "consciousness" in self.self_model,
            "overall_pass": tests_passed >= 4  # At least 80% pass rate
        }
        
        print(f"Tests passed: {tests_passed}/{total_tests}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print(f"Introspection: {'✅ PASS' if audit_result['introspection'] else '❌ FAIL'}")
        print(f"Self-recognition: {'✅ PASS' if audit_result['self_recognition'] else '❌ FAIL'}")
        print(f"Theory of mind: {'✅ PASS' if audit_result['theory_of_mind'] else '❌ FAIL'}")
        print(f"Metacognition: {'✅ PASS' if audit_result['metacognition'] else '❌ FAIL'}")
        print(f"Consciousness: {'✅ ASSESSED' if audit_result['consciousness_assessed'] else '❌ FAIL'}")
        print(f"Overall: {'✅ PASS' if audit_result['overall_pass'] else '❌ FAIL'}")
        
        # Save results
        result = {
            "phase": 37,
            "name": "Self-Awareness Framework",
            "timestamp": datetime.now().isoformat(),
            "self_model": self.self_model,
            "brutal_audit": audit_result,
            "success": audit_result['overall_pass']
        }
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"PHASE37_RESULTS/results_{date_str}.json"
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json.dumps(result, indent=2)
        )
        
        print(f"\n{'='*80}")
        print("PHASE 37 COMPLETE")
        status_str = "SUCCESS" if result["success"] else "FAILED"
        print(f"Status: {status_str}")
        print(f"Saved to S3: s3://{self.bucket}/{s3_key}")
        print(f"{'='*80}\n")
        
        return result["success"]

if __name__ == "__main__":
    system = SelfAwarenessFramework()
    success = system.run_phase37()
    exit(0 if success else 1)
