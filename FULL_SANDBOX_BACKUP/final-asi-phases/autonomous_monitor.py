#!/usr/bin/env python3.11
'''
Autonomous Self-Monitoring System
Continuously monitors ASI system health and performance
'''

import json
import time
import subprocess
from datetime import datetime
import os

class AutonomousMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    def check_system_health(self):
        '''Check overall system health'''
        print("🏥 Checking system health...")
        
        health = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check disk space
        try:
            result = subprocess.run(['df', '-h', '/home'], capture_output=True, text=True)
            health['checks']['disk_space'] = 'OK' if result.returncode == 0 else 'FAIL'
        except:
            health['checks']['disk_space'] = 'UNKNOWN'
        
        # Check S3 connectivity
        try:
            result = subprocess.run(
                ['aws', 's3', 'ls', 's3://asi-knowledge-base-898982995956/'],
                capture_output=True,
                timeout=10
            )
            health['checks']['s3_connectivity'] = 'OK' if result.returncode == 0 else 'FAIL'
        except:
            health['checks']['s3_connectivity'] = 'FAIL'
        
        # Check Python processes
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            python_procs = len([l for l in result.stdout.split('\n') if 'python' in l])
            health['checks']['python_processes'] = f'{python_procs} running'
        except:
            health['checks']['python_processes'] = 'UNKNOWN'
        
        # Overall status
        failed = sum(1 for v in health['checks'].values() if 'FAIL' in str(v))
        health['overall_status'] = 'HEALTHY' if failed == 0 else f'DEGRADED ({failed} failures)'
        
        return health
    
    def check_performance_metrics(self):
        '''Check performance metrics'''
        print("📊 Checking performance metrics...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Check file count
        try:
            result = subprocess.run(
                ['find', '/home/ubuntu/final-asi-phases', '-type', 'f'],
                capture_output=True,
                text=True
            )
            file_count = len(result.stdout.strip().split('\n'))
            metrics['metrics']['total_files'] = file_count
        except:
            metrics['metrics']['total_files'] = 'UNKNOWN'
        
        # Check total size
        try:
            result = subprocess.run(
                ['du', '-sh', '/home/ubuntu/final-asi-phases'],
                capture_output=True,
                text=True
            )
            size = result.stdout.split()[0]
            metrics['metrics']['total_size'] = size
        except:
            metrics['metrics']['total_size'] = 'UNKNOWN'
        
        return metrics
    
    def autonomous_monitoring_cycle(self):
        '''Run autonomous monitoring cycle'''
        print("\n🤖 AUTONOMOUS MONITORING CYCLE")
        print("="*50)
        
        # Check health
        health = self.check_system_health()
        print(f"\n  Overall Status: {health['overall_status']}")
        for check, status in health['checks'].items():
            symbol = '✅' if 'OK' in str(status) or 'running' in str(status) else '⚠️'
            print(f"  {symbol} {check}: {status}")
        
        # Check performance
        metrics = self.check_performance_metrics()
        print(f"\n  Performance Metrics:")
        for metric, value in metrics['metrics'].items():
            print(f"    - {metric}: {value}")
        
        # Save monitoring report
        report = {
            'timestamp': datetime.now().isoformat(),
            'health': health,
            'metrics': metrics
        }
        
        with open('/home/ubuntu/final-asi-phases/monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Monitoring report saved")
        
        # Upload to S3
        try:
            subprocess.run([
                'aws', 's3', 'cp',
                '/home/ubuntu/final-asi-phases/monitoring_report.json',
                's3://asi-knowledge-base-898982995956/FINAL_ASI_PHASES/'
            ], capture_output=True, timeout=10)
            print("  ✅ Report uploaded to S3")
        except:
            print("  ⚠️ Could not upload to S3")
        
        return health['overall_status']

if __name__ == '__main__':
    monitor = AutonomousMonitor()
    status = monitor.autonomous_monitoring_cycle()
    print(f"\n✅ Monitoring complete - Status: {status}")
