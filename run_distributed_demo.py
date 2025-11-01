#!/usr/bin/env python3
"""
TRUE ASI System - Distributed Framework Demo
=============================================

Demonstrate distributed computing capabilities.

Author: TRUE ASI System
Date: November 1, 2025
Version: 1.0.0
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import components
from distributed.distributed_framework import DistributedFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main demonstration function"""
    print("="*70)
    print("TRUE ASI SYSTEM - DISTRIBUTED COMPUTING FRAMEWORK DEMONSTRATION")
    print("="*70)
    print()
    
    # Create distributed framework
    framework = DistributedFramework()
    
    print("🚀 Submitting 100 tasks to distributed system...")
    print()
    
    # Submit tasks
    for i in range(100):
        priority = (i % 10) + 1  # Priorities 1-10
        await framework.submit_task(
            task_type="repository_processing",
            payload={"repo_id": i, "action": "analyze"},
            priority=priority
        )
    
    print(f"✅ Submitted 100 tasks")
    print()
    
    print("🔄 Processing tasks with auto-scaling and load balancing...")
    print()
    
    # Process tasks
    await framework.process_tasks()
    
    print()
    print("="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print()
    
    # Generate and display report
    report = framework.generate_report()
    print(report)
    
    # Save report
    report_file = Path("DISTRIBUTED_FRAMEWORK_REPORT.txt")
    report_file.write_text(report)
    print()
    print(f"✅ Report saved: {report_file}")
    
    print()
    print("="*70)
    print("DISTRIBUTED FRAMEWORK DEMONSTRATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
