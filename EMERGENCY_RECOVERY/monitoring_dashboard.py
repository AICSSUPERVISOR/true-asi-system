#!/usr/bin/env python3
"""
ASI Monitoring Dashboard
========================
Real-time monitoring of ASI system components and performance.
100% functional - no simulations.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

class ASIMonitoringDashboard:
    """Real-time monitoring dashboard for ASI system."""
    
    def __init__(self, base_path: str = "/home/ubuntu/real-asi"):
        self.base_path = Path(base_path)
        self.metrics = {}
        self.alerts = []
        
    def check_component_health(self) -> dict:
        """Check health of all ASI components."""
        components = {
            "arc_agi_dataset": self._check_arc_dataset(),
            "evaluation_harness": self._check_file("arc_evaluation_harness.py"),
            "ensemble_framework": self._check_file("ensemble_framework.py"),
            "training_pipeline": self._check_file("training_data_pipeline.py"),
            "runpod_deployment": self._check_directory("runpod_deployment"),
            "mit_ttt_repo": self._check_directory("marc"),
            "jeremy_repo": self._check_directory("arc_agi"),
            "soar_synthesis": self._check_file("soar_program_synthesis.py"),
            "poetiq_loop": self._check_file("poetiq_refinement_loop.py"),
        }
        return components
    
    def _check_arc_dataset(self) -> dict:
        """Check ARC-AGI dataset availability."""
        arc_path = Path("/home/ubuntu/ARC-AGI/data")
        if arc_path.exists():
            training = len(list((arc_path / "training").glob("*.json"))) if (arc_path / "training").exists() else 0
            evaluation = len(list((arc_path / "evaluation").glob("*.json"))) if (arc_path / "evaluation").exists() else 0
            return {
                "status": "healthy" if training >= 400 else "degraded",
                "training_tasks": training,
                "evaluation_tasks": evaluation
            }
        return {"status": "missing", "training_tasks": 0, "evaluation_tasks": 0}
    
    def _check_file(self, filename: str) -> dict:
        """Check if a file exists and get its stats."""
        filepath = self.base_path / filename
        if filepath.exists():
            stat = filepath.stat()
            return {
                "status": "healthy",
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        return {"status": "missing", "size_bytes": 0, "modified": None}
    
    def _check_directory(self, dirname: str) -> dict:
        """Check if a directory exists and count files."""
        dirpath = self.base_path / dirname
        if dirpath.exists():
            file_count = sum(1 for _ in dirpath.rglob("*") if _.is_file())
            return {
                "status": "healthy",
                "file_count": file_count
            }
        return {"status": "missing", "file_count": 0}
    
    def check_aws_connectivity(self) -> dict:
        """Check AWS S3 connectivity."""
        try:
            import subprocess
            result = subprocess.run(
                ["aws", "s3", "ls", "s3://asi-knowledge-base-898982995956/", "--max-items", "1"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return {"status": "connected", "bucket": "asi-knowledge-base-898982995956"}
            return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_system_metrics(self) -> dict:
        """Get current system metrics."""
        import subprocess
        
        metrics = {}
        
        # Disk usage
        try:
            result = subprocess.run(["df", "-h", "/home/ubuntu"], capture_output=True, text=True)
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split()
                metrics["disk"] = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "percent_used": parts[4]
                }
        except:
            metrics["disk"] = {"status": "error"}
        
        # Memory
        try:
            result = subprocess.run(["free", "-h"], capture_output=True, text=True)
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split()
                metrics["memory"] = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[6] if len(parts) > 6 else "N/A"
                }
        except:
            metrics["memory"] = {"status": "error"}
        
        return metrics
    
    def generate_dashboard_report(self) -> dict:
        """Generate complete dashboard report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system": "ASI Monitoring Dashboard",
            "version": "1.0.0",
            "components": self.check_component_health(),
            "aws": self.check_aws_connectivity(),
            "system_metrics": self.get_system_metrics(),
            "overall_status": "healthy"
        }
        
        # Determine overall status
        unhealthy = sum(1 for c in report["components"].values() if c.get("status") != "healthy")
        if unhealthy > 0:
            report["overall_status"] = "degraded" if unhealthy < 3 else "critical"
        
        if report["aws"].get("status") != "connected":
            report["overall_status"] = "degraded"
        
        return report
    
    def print_dashboard(self):
        """Print dashboard to console."""
        report = self.generate_dashboard_report()
        
        print("=" * 60)
        print("ASI MONITORING DASHBOARD")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['overall_status'].upper()}")
        print()
        
        print("COMPONENT HEALTH:")
        print("-" * 40)
        for name, status in report["components"].items():
            status_icon = "✅" if status.get("status") == "healthy" else "❌"
            print(f"  {status_icon} {name}: {status.get('status', 'unknown')}")
        
        print()
        print("AWS CONNECTIVITY:")
        print("-" * 40)
        aws = report["aws"]
        status_icon = "✅" if aws.get("status") == "connected" else "❌"
        print(f"  {status_icon} S3: {aws.get('status', 'unknown')}")
        
        print()
        print("SYSTEM METRICS:")
        print("-" * 40)
        metrics = report["system_metrics"]
        if "disk" in metrics:
            print(f"  Disk: {metrics['disk'].get('used', 'N/A')} / {metrics['disk'].get('total', 'N/A')} ({metrics['disk'].get('percent_used', 'N/A')})")
        if "memory" in metrics:
            print(f"  Memory: {metrics['memory'].get('used', 'N/A')} / {metrics['memory'].get('total', 'N/A')}")
        
        print()
        print("=" * 60)
        
        return report


def main():
    """Run monitoring dashboard."""
    dashboard = ASIMonitoringDashboard()
    report = dashboard.print_dashboard()
    
    # Save report to file
    output_path = Path("/home/ubuntu/real-asi/dashboard_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    main()
