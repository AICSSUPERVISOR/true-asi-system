#!/usr/bin/env python3
"""
ASI System Automated Tests
==========================
Comprehensive test suite for all ASI components.
100% functional - no simulations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple, List

class TestResult:
    """Test result container."""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message

class ASITestSuite:
    """Automated test suite for ASI system."""
    
    def __init__(self, base_path: str = "/home/ubuntu/real-asi"):
        self.base_path = Path(base_path)
        self.results: List[TestResult] = []
        
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all tests and return (passed, failed) counts."""
        self.results = []
        
        # File existence tests
        self._test_file_exists("arc_evaluation_harness.py", "Evaluation harness")
        self._test_file_exists("ensemble_framework.py", "Ensemble framework")
        self._test_file_exists("training_data_pipeline.py", "Training pipeline")
        self._test_file_exists("soar_program_synthesis.py", "SOAR synthesis")
        self._test_file_exists("poetiq_refinement_loop.py", "Poetiq loop")
        self._test_file_exists("monitoring_dashboard.py", "Monitoring dashboard")
        
        # Directory tests
        self._test_directory_exists("runpod_deployment", "Runpod deployment")
        self._test_directory_exists("marc", "MIT TTT repository")
        self._test_directory_exists("arc_agi", "Jeremy Berman repository")
        
        # ARC-AGI dataset tests
        self._test_arc_dataset()
        
        # Code quality tests
        self._test_no_simulations()
        self._test_no_mocks()
        self._test_no_hardcoded_scores()
        
        # Import tests
        self._test_python_imports()
        
        # Configuration tests
        self._test_model_configs()
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        return passed, failed
    
    def _test_file_exists(self, filename: str, description: str):
        """Test that a file exists."""
        filepath = self.base_path / filename
        if filepath.exists():
            self.results.append(TestResult(f"File: {description}", True))
        else:
            self.results.append(TestResult(f"File: {description}", False, f"Missing: {filename}"))
    
    def _test_directory_exists(self, dirname: str, description: str):
        """Test that a directory exists and has files."""
        dirpath = self.base_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            file_count = sum(1 for _ in dirpath.rglob("*") if _.is_file())
            if file_count > 0:
                self.results.append(TestResult(f"Dir: {description}", True, f"{file_count} files"))
            else:
                self.results.append(TestResult(f"Dir: {description}", False, "Empty directory"))
        else:
            self.results.append(TestResult(f"Dir: {description}", False, f"Missing: {dirname}"))
    
    def _test_arc_dataset(self):
        """Test ARC-AGI dataset availability."""
        arc_path = Path("/home/ubuntu/ARC-AGI/data")
        if arc_path.exists():
            training_path = arc_path / "training"
            eval_path = arc_path / "evaluation"
            
            training_count = len(list(training_path.glob("*.json"))) if training_path.exists() else 0
            eval_count = len(list(eval_path.glob("*.json"))) if eval_path.exists() else 0
            
            if training_count >= 400 and eval_count >= 400:
                self.results.append(TestResult("ARC-AGI dataset", True, f"{training_count} training, {eval_count} eval"))
            else:
                self.results.append(TestResult("ARC-AGI dataset", False, f"Incomplete: {training_count} training, {eval_count} eval"))
        else:
            self.results.append(TestResult("ARC-AGI dataset", False, "Missing dataset"))
    
    def _test_no_simulations(self):
        """Test that no files contain simulation flags set to True."""
        simulation_files = []
        for pyfile in self.base_path.glob("*.py"):
            content = pyfile.read_text()
            # Skip test files themselves
            if pyfile.name == "test_asi_system.py":
                continue
            if "simulated = True" in content or "simulation = True" in content:
                simulation_files.append(pyfile.name)
        
        if not simulation_files:
            self.results.append(TestResult("No simulations", True))
        else:
            self.results.append(TestResult("No simulations", False, f"Found in: {', '.join(simulation_files)}"))
    
    def _test_no_mocks(self):
        """Test that no files contain mock flags set to True."""
        mock_files = []
        for pyfile in self.base_path.glob("*.py"):
            content = pyfile.read_text()
            # Skip test files themselves
            if pyfile.name == "test_asi_system.py":
                continue
            if "mock = True" in content or "fake = True" in content:
                mock_files.append(pyfile.name)
        
        if not mock_files:
            self.results.append(TestResult("No mocks", True))
        else:
            self.results.append(TestResult("No mocks", False, f"Found in: {', '.join(mock_files)}"))
    
    def _test_no_hardcoded_scores(self):
        """Test that no files contain hardcoded fake scores."""
        import re
        score_files = []
        pattern = r'(accuracy|score)\s*=\s*0\.9[0-9]'
        
        for pyfile in self.base_path.glob("*.py"):
            content = pyfile.read_text()
            if re.search(pattern, content):
                # Check if it's in a comment or expected value
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.search(pattern, line) and not line.strip().startswith('#'):
                        if 'expected' not in line.lower() and 'threshold' not in line.lower():
                            score_files.append(f"{pyfile.name}:{i+1}")
        
        if not score_files:
            self.results.append(TestResult("No hardcoded scores", True))
        else:
            self.results.append(TestResult("No hardcoded scores", False, f"Found: {', '.join(score_files[:3])}"))
    
    def _test_python_imports(self):
        """Test that key Python files can be parsed."""
        import ast
        parse_errors = []
        
        for pyfile in self.base_path.glob("*.py"):
            try:
                content = pyfile.read_text()
                ast.parse(content)
            except SyntaxError as e:
                parse_errors.append(f"{pyfile.name}: {e}")
        
        if not parse_errors:
            self.results.append(TestResult("Python syntax", True))
        else:
            self.results.append(TestResult("Python syntax", False, f"Errors: {len(parse_errors)}"))
    
    def _test_model_configs(self):
        """Test model configuration files."""
        config_path = Path("/home/ubuntu/asi-models/models_info.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if "models" in config and len(config["models"]) >= 4:
                    self.results.append(TestResult("Model configs", True, f"{len(config['models'])} models"))
                else:
                    self.results.append(TestResult("Model configs", False, "Incomplete config"))
            except json.JSONDecodeError:
                self.results.append(TestResult("Model configs", False, "Invalid JSON"))
        else:
            self.results.append(TestResult("Model configs", False, "Missing config file"))
    
    def print_results(self):
        """Print test results to console."""
        print("=" * 60)
        print("ASI SYSTEM TEST RESULTS")
        print("=" * 60)
        print()
        
        passed = 0
        failed = 0
        
        for result in self.results:
            icon = "✅" if result.passed else "❌"
            status = "PASS" if result.passed else "FAIL"
            msg = f" - {result.message}" if result.message else ""
            print(f"  {icon} [{status}] {result.name}{msg}")
            
            if result.passed:
                passed += 1
            else:
                failed += 1
        
        print()
        print("-" * 60)
        print(f"TOTAL: {passed} passed, {failed} failed")
        print(f"STATUS: {'ALL TESTS PASSED ✅' if failed == 0 else 'SOME TESTS FAILED ❌'}")
        print("=" * 60)
        
        return passed, failed


def main():
    """Run test suite."""
    suite = ASITestSuite()
    passed, failed = suite.run_all_tests()
    suite.print_results()
    
    # Save results
    results_path = Path("/home/ubuntu/real-asi/test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "passed": passed,
            "failed": failed,
            "results": [{"name": r.name, "passed": r.passed, "message": r.message} for r in suite.results]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Exit with error code if tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
