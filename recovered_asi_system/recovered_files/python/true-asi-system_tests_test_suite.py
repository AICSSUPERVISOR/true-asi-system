
import unittest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from master_integration import MasterIntegration

class TestTrueASISystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class with a MasterIntegration instance."""
        cls.master = MasterIntegration()

    def test_01_system_initialization(self):
        """Test that the system initializes without errors."""
        self.assertIsNotNone(self.master)
        self.assertIsNotNone(self.master.gpu_system)
        self.assertIsNotNone(self.master.s3_loader)
        self.assertIsNotNone(self.master.unified_bridge)
        self.assertIsNotNone(self.master.collaboration)
        self.assertIsNotNone(self.master.super_machine)
        self.assertIsNotNone(self.master.symbiosis)
        self.assertIsNotNone(self.master.power_bridge)
        self.assertIsNotNone(self.master.asi_coordinator)

    def test_02_get_available_models(self):
        """Test that the system can retrieve a list of available models."""
        models = self.master.get_available_models()
        self.assertIsInstance(models, dict)
        self.assertIn("local_models", models)
        self.assertIn("api_models", models)
        self.assertGreater(models["total"], 0)

    def test_03_single_model_execution(self):
        """Test that the system can execute a single model."""
        response = self.master.execute_single_model(
            model_name="qwen:7b-chat-v1.5-q4_K_M",
            prompt="What is the capital of France?"
        )
        self.assertIsInstance(response, dict)
        self.assertEqual(response["status"], "success")
        self.assertIn("Paris", response["response"])

    def test_04_multi_model_consensus(self):
        """Test that the system can execute multiple models with consensus."""
        response = self.master.execute_multi_model_consensus(
            model_names=["qwen:7b-chat-v1.5-q4_K_M", "llama2:7b-chat-q4_K_M"],
            prompt="What is the capital of France?"
        )
        self.assertIsInstance(response, dict)
        self.assertIn("consensus_response", response)
        self.assertIn("Paris", response["consensus_response"])

    def test_05_collaboration_pattern(self):
        """Test that the system can execute a collaboration pattern."""
        response = self.master.execute_collaboration_pattern(
            pattern="pipeline",
            model_names=["qwen:7b-chat-v1.5-q4_K_M", "llama2:7b-chat-q4_K_M"],
            task="Write a short story about a robot who discovers music."
        )
        self.assertIsInstance(response, dict)
        self.assertIn("final_result", response)

    def test_06_asi_task_execution(self):
        """Test that the system can execute a task with the full ASI system."""
        response = self.master.execute_asi_task(
            task="Analyze the current state of the stock market and provide a summary."
        )
        self.assertIsInstance(response, dict)
        self.assertIn("summary", response)

    def test_07_power_bridge_execution(self):
        """Test that the system can execute a task with the power bridge."""
        response = self.master.execute_power_bridge(
            task="Generate 10 creative ideas for a new mobile app.",
            num_models=5
        )
        self.assertIsInstance(response, dict)
        self.assertIn("results", response)
        self.assertEqual(len(response["results"]), 5)

if __name__ == "__main__":
    unittest.main()
